# src/sampson_active_learning/active_learning.py

from __future__ import annotations

import math
import os
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sedoNNa.utils import construct_run
from sedoNNa.dataloader import FastSupernovaDataset, NormalizeSpectralData
from helpers import mkdir, write_to_file, read_file
from .utils import evaluate_model

from .config import AppConfig


@dataclass
class NormalizationStats:
    fluxes_mean: torch.Tensor
    fluxes_std: torch.Tensor
    time_mean: torch.Tensor
    time_std: torch.Tensor
    descriptor_mean: torch.Tensor
    descriptor_std: torch.Tensor


def load_normalization_stats(cfg: AppConfig, device: torch.device) -> NormalizationStats:
    '''
    Normalization stats hold the information needed to normalize the fluxes, wavelengths, time, 
    and physical parameters. Emulator only deals with normalized data because it's a NN. 
    '''
    mean_std_dict = torch.load(cfg.data.normalization_stats_file, weights_only=False)
    return NormalizationStats(
        fluxes_mean=mean_std_dict["fluxes_mean"],
        fluxes_std=mean_std_dict["fluxes_std"],
        time_mean=mean_std_dict["time_mean"],
        time_std=mean_std_dict["time_std"],
        descriptor_mean=mean_std_dict["descriptor_mean"].float().to(device),
        descriptor_std=mean_std_dict["descriptor_std"].float().to(device),
    )


def load_preprocessed_dataset(cfg: AppConfig) -> List[Dict]:
    '''
    Training and test set are preprocessed in to a .pth file. The .pth file holds all the spectra
    that the emulator will be trained on. This way, the file IO bottleneck is removed during training. 
    '''
    all_data = torch.load(cfg.data.preprocessed_spectra_file, weights_only=False)
    return all_data


def build_test_loader(
    all_data: List[Dict],
    norm_stats: NormalizationStats,
    device: torch.device,
) -> tuple[DataLoader, List[int]]:
    """
    Read in the dataset, pass out only the test sample ids. This will be used to evaluate the 
    trained emulator to find how well it's performing. 
    """
    all_sample_ids = sorted({entry["sample_id"] for entry in all_data})
    train_sample_ids, test_sample_ids = train_test_split(
        all_sample_ids, train_size=0.8, random_state=42
    )
    sample_ids = set(test_sample_ids)

    data = [entry for entry in all_data if entry["sample_id"] in sample_ids]
    transform = NormalizeSpectralData(
        {
            "fluxes_mean": norm_stats.fluxes_mean,
            "fluxes_std": norm_stats.fluxes_std,
            "time_mean": norm_stats.time_mean,
            "time_std": norm_stats.time_std,
            "descriptor_mean": norm_stats.descriptor_mean.cpu(),
            "descriptor_std": norm_stats.descriptor_std.cpu(),
        }
    )
    dataset = FastSupernovaDataset(samples=data, transform=transform)
    loader = DataLoader(
        dataset, batch_size=64, shuffle=True, drop_last=True, num_workers=4, pin_memory=True
    )

    # just to get wav for evaluate_model
    batch = next(iter(loader))
    wav = batch["wav"][0].to(device)
    return loader, wav, test_sample_ids


def read_lc_properties(cfg: AppConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    to_read = cfg.data.lc_properties_file.format(
        model=cfg.model.name, epochs=cfg.model.epochs
    )
    lc_prop_dict = read_file(to_read)["dict"]
    sids = np.asarray(lc_prop_dict[0], dtype=float)
    avg_mag_diff = np.asarray(lc_prop_dict[1], dtype=float)
    max_mag_diff = np.asarray(lc_prop_dict[2], dtype=float)

    # sort in descending order by avg_mag_diff (as in your script)
    sorted_indices = np.argsort(avg_mag_diff)[::-1]
    sids = sids[sorted_indices]
    avg_mag_diff = avg_mag_diff[sorted_indices]
    max_mag_diff = max_mag_diff[sorted_indices]
    return sids, avg_mag_diff, max_mag_diff


def get_sample_location(
    sid: int,
    cfg: AppConfig,
    norm_stats: NormalizationStats,
    device: torch.device,
) -> torch.Tensor | None:
    """
    Read sample.txt for this sample id and return normalized physical parameters.
    """
    path = cfg.data.sample_file_template.format(sid=sid)
    if not os.path.exists(path):
        print(f"sample txt not found for sample id: {sid}")
        return None

    sample_df = pd.read_csv(path, sep=",")
    sample_loc = torch.tensor(
        [
            sample_df["D"].values[0],
            sample_df["R_2"].values[0],
            sample_df["R_28"].values[0],
            sample_df["R_opacity"].values[0],
            sample_df["min_vel"].values[0],
            sample_df["max_vel"].values[0],
            sample_df["total_2"].values[0],
            sample_df["total_28"].values[0],
            sample_df["total_opacity"].values[0],
        ],
        dtype=torch.float32,
        device=device,
    )

    return (sample_loc - norm_stats.descriptor_mean) / norm_stats.descriptor_std


def propose_new_samples(
    cfg: AppConfig,
    sids_sorted: np.ndarray,
    norm_stats: NormalizationStats,
    device: torch.device,
) -> Dict[str, List[float]]:
    """
    Makes M number of jumps from each of the top N worst-fit samples in the test set. 
    Each new sample to which we jump is added to the new training set, to augment the train set. 
    In theory, once these new data points are added to the training set, the emulator will perform
    better over the N samples in the dataset, and they will no longer be badly-fit samples.
    The size of each jump is sampled from a half-gaussian with standard deviation d in 
    normalized units. The direction of each jump is in the direction of the vector that points
    from poorly-fit point i towards the (i+1)-th worst-fit point. 
    This will generate N*M new points to augment the original training set. 
    """
    N = cfg.active_learning.N
    M = cfg.active_learning.M
    d = cfg.active_learning.step_sigma

    new_sample_points = {
        "D": [],
        "R_2": [],
        "R_28": [],
        "R_opacity": [],
        "min_vel": [],
        "max_vel": [],
        "total_2": [],
        "total_28": [],
        "total_opacity": [],
    }

    for i in range(N):
        sid = int(sids_sorted[i])
        next_sample = int(sids_sorted[i + 1])
        print(f"Working on sample id: {sid}")

        for j in range(M):
            l = abs(np.random.normal(loc=0.0, scale=d))

            sample_loc = get_sample_location(sid, cfg, norm_stats, device)
            next_sample_loc = get_sample_location(next_sample, cfg, norm_stats, device)
            if sample_loc is None or next_sample_loc is None:
                print(
                    f"Could not get sample location for sample id: {sid} or {next_sample}. Skipping..."
                )
                continue

            direction = next_sample_loc - sample_loc
            direction = direction / torch.norm(direction)

            new_sample_loc = sample_loc + l * direction
            # unnormalize
            new_sample_loc = new_sample_loc * norm_stats.descriptor_std + norm_stats.descriptor_mean

            new_sample_points["D"].append(new_sample_loc[0].item())
            new_sample_points["R_2"].append(new_sample_loc[1].item())
            new_sample_points["R_28"].append(new_sample_loc[2].item())
            new_sample_points["R_opacity"].append(new_sample_loc[3].item())
            new_sample_points["min_vel"].append(new_sample_loc[4].item())
            new_sample_points["max_vel"].append(new_sample_loc[5].item())
            new_sample_points["total_2"].append(new_sample_loc[6].item())
            new_sample_points["total_28"].append(new_sample_loc[7].item())
            new_sample_points["total_opacity"].append(new_sample_loc[8].item())

    return new_sample_points


def submit_sedona_jobs(
    cfg: AppConfig,
    new_sample_points: Dict[str, List[float]],
    sids_sorted: np.ndarray,
) -> None:
    """
    Loop through N*M new sample points, create supplemental_grid dirs, call construct_run,
    and submit one Sedona batch job per new run. Behavior matches your original script.
    """
    mkdir(str(cfg.data.supplemental_grid_root))

    files = glob.glob(str(cfg.data.supplemental_grid_root / "*/mod.mod"))
    max_existing_sample = (
        np.max([int(f.split("/")[-2]) for f in files]) if len(files) > 0 else -1
    )

    ran = 0
    to_run = cfg.active_learning.to_run

    for i in range(len(new_sample_points["D"])):
        if ran >= to_run:
            break

        out_dir = cfg.data.supplemental_grid_root / str(max_existing_sample + 1 + i)
        out_dir_str = str(out_dir) + "/"

        D = new_sample_points["D"][i]
        R_2 = new_sample_points["R_2"][i]
        R_28 = new_sample_points["R_28"][i]
        R_opacity = new_sample_points["R_opacity"][i]
        min_vel = new_sample_points["min_vel"][i]
        max_vel = new_sample_points["max_vel"][i]
        total_2 = new_sample_points["total_2"][i]
        total_28 = new_sample_points["total_28"][i]
        total_opacity = new_sample_points["total_opacity"][i]

        construct_run(
            out_dir_str,
            D,
            R_2,
            R_28,
            R_opacity,
            min_vel,
            max_vel,
            total_2,
            total_28,
            total_opacity,
            start_time=5.0,
            stop_time=50,
            dt=0.2,
            hours=3,
            mass_dict=None,
            vel=None,
            job_name=f"supp_{max_existing_sample+1+i}",
        )

        write_to_file(
            out_dir_str + "info.txt",
            f"This is supplemental sample {i % len(new_sample_points['D'])} "
            f"for original sample {int(sids_sorted[math.floor(i / cfg.active_learning.M)])}",
        )

        print(f"Starting job for sample id: {max_existing_sample+1+i}")
        os.system(f"cd {out_dir_str}; sbatch run_batch.sub")
        ran += 1


def get_training_sbatch_code(
    idx: str,
    cfg: AppConfig,
    d_model: int,
    nhead: int,
    num_layers: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    MLP: bool = True,
) -> tuple[str, str]:
    """
    Create an sbatch script to submit a slurm job that trains the emulator over the newly 
    augmented training dataset. Everything is read-in from the fancy config file; 
    We live lavishly in this house. We use config files. 
    """
    logs_dir = cfg.training.logs_dir
    mkdir(str(logs_dir))


    header = f"""#!/bin/sh
#SBATCH --cpus-per-task {cfg.training.sbatch_cpus_per_task}
#SBATCH --nodes {cfg.training.sbatch_nodes}
#SBATCH -t {cfg.training.sbatch_time}
#SBATCH -p {cfg.training.sbatch_partition}
#SBATCH --gres={cfg.training.sbatch_gres}
#SBATCH --constraint="{cfg.training.sbatch_constraint}"
#SBATCH --mem={cfg.training.sbatch_mem}
#SBATCH --output={logs_dir}/{idx}.out
#SBATCH --error={logs_dir}/{idx}.err
#SBATCH --job-name=grid_{idx}
"""

    module_lines = "\n".join(f"module load {m}" for m in cfg.training.sbatch_modules)

    body = f"""
source ~/.bashrc
{module_lines}
mamba activate {cfg.training.conda_env}

normalization_stats_file={cfg.data.normalization_stats_file}
preprocessed_spectra_file={cfg.data.preprocessed_spectra_file}
supp_preprocessed_spectra_file={cfg.data.supp_preprocessed_spectra_file}
train_small_file={cfg.training.train_small_file}

python $train_small_file \\
    --d_model {d_model} \\
    --nhead {nhead} \\
    --num_layers {num_layers} \\
    --learnedPE True \\
    --lr {lr} \\
    --weight_decay {weight_decay} \\
    --batch_size {batch_size} \\
    --epochs 300 \\
    --normalization_stats_file $normalization_stats_file \\
    --preprocessed_spectra_file $preprocessed_spectra_file \\
    --idx {idx} \\
    --supplemental_spectra_file $supp_preprocessed_spectra_file
"""

    script = header + body
    job_signature = (
        f"{idx}_dim{d_model}_nhead{nhead}_numlayers{num_layers}_"
        f"learnedPETrue_lr{lr}_weightdecay{weight_decay}_batchsize{batch_size}"
    )
    return script, job_signature


def launch_training_job(cfg: AppConfig) -> None:
    """
    Creates and launches a new training job. 
    """

    dir1 = cfg.training.emulator_job_dir
    mkdir(str(dir1))

    trained_ems = glob.glob(str(dir1 / "run_em_*.sh"))
    if len(trained_ems) > 0:
        idx = str(
            1
            + np.max(
                [int(Path(i).stem.split("_")[-1]) for i in trained_ems]
            )
        )
    else:
        idx = "0"

    script, job_signature = get_training_sbatch_code(
        idx,
        cfg,
        d_model=512,
        nhead=64,
        num_layers=7,
        lr=0.008,
        weight_decay=0.07,
        batch_size=512,
    )

    script_path = dir1 / f"run_em_{idx}.sh"
    script_path.write_text(script)
    os.system(f"cd {dir1}; sbatch run_em_{idx}.sh; cd ..")


def recalculate_normalization(cfg: AppConfig) -> None:
    """
    This is the job that rebuilds normalization_stats.pt,
    preprocessed_spectra.pt, and supp_preprocessed_spectra.pt.
    This should be run after new sedona sims are finished running. 
    """

    os.system(f"sbatch {cfg.data.recalc_batch_script}")


def run_propose_round(cfg: AppConfig) -> None:
    """
    Stage A: evaluate (optionally), hoose worst samples, propose new points,
    and submit Sedona jobs. Does NOT launch preprocessing or emulator training.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    norm_stats = load_normalization_stats(cfg, device)
    all_data = load_preprocessed_dataset(cfg)
    loader, wav, test_sample_ids = build_test_loader(all_data, norm_stats, device)

    if cfg.flags.eval_model:
        evaluate_model(
            cfg,
            cfg.model.name,
            cfg.model.epochs,
            set(test_sample_ids),
            wav,
        )

    sids, avg_mag_diff, max_mag_diff = read_lc_properties(cfg)

    new_sample_points = propose_new_samples(cfg, sids, norm_stats, device)
    submit_sedona_jobs(cfg, new_sample_points, sids)
    # No normalization or training here.


def run_preprocess_round(cfg: AppConfig) -> None:
    """
    Stage B: submit the job that recomputes normalization and preprocessed spectra
    after new Sedona simulations have finished.

    This should be run once the new Sedona runs are complete.
    """
    recalculate_normalization(cfg)


def run_training_round(cfg: AppConfig) -> None:
    """
    Stage C: launch emulator training job, assuming:
      - all Sedona runs (including newly proposed ones) are done, and
      - normalization / preprocessed spectra files are already updated.
    """
    launch_training_job(cfg)