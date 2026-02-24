# src/sampson_active_learning/config.py

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import yaml


@dataclass
class DataConfig:
    normalization_stats_file: Path
    preprocessed_spectra_file: Path
    supp_preprocessed_spectra_file: Path
    lc_properties_file: str          # has {model} and {epochs}
    sample_file_template: str        # has {sid}
    supplemental_grid_root: Path
    recalc_batch_script: Path        # e.g. run1_batch.sub


@dataclass
class ModelConfig:
    ckpt_dir: Path
    name: str
    epochs: int


@dataclass
class ActiveLearningConfig:
    N: int
    M: int
    step_sigma: float
    to_run: int
    random_seed: int


@dataclass
class FlagsConfig:
    eval_model: bool


@dataclass
class TrainingConfig:
    emulator_job_dir: Path
    train_small_file: Path
    sbatch_cpus_per_task: int
    sbatch_nodes: int
    sbatch_time: str
    sbatch_partition: str
    sbatch_gres: str
    sbatch_constraint: str
    sbatch_mem: str
    sbatch_modules: List[str]
    conda_env: str
    logs_dir: Path
    eval_plots_dir: Path



@dataclass
class AppConfig:
    data: DataConfig
    model: ModelConfig
    active_learning: ActiveLearningConfig
    flags: FlagsConfig
    training: TrainingConfig


def _to_path(v: Any) -> Path:
    return Path(v).expanduser().resolve()


def load_config(path: str | Path) -> AppConfig:
    with open(path, "r") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    data = DataConfig(
        normalization_stats_file=_to_path(raw["data"]["normalization_stats_file"]),
        preprocessed_spectra_file=_to_path(raw["data"]["preprocessed_spectra_file"]),
        supp_preprocessed_spectra_file=_to_path(
            raw["data"]["supp_preprocessed_spectra_file"]
        ),
        lc_properties_file=raw["data"]["lc_properties_file"],
        sample_file_template=raw["data"]["sample_file_template"],
        supplemental_grid_root=_to_path(raw["data"]["supplemental_grid_root"]),
        recalc_batch_script=_to_path(raw["data"]["recalc_batch_script"]),
    )

    model = ModelConfig(
        ckpt_dir=_to_path(raw["model"]["ckpt_dir"]),
        name=raw["model"]["name"],
        epochs=int(raw["model"]["epochs"]),
    )

    al = ActiveLearningConfig(
        N=int(raw["active_learning"]["N"]),
        M=int(raw["active_learning"]["M"]),
        step_sigma=float(raw["active_learning"]["step_sigma"]),
        to_run=int(raw["active_learning"]["to_run"]),
        random_seed=int(raw["active_learning"]["random_seed"]),
    )

    flags = FlagsConfig(
        eval_model=bool(raw["flags"]["eval_model"]),
    )

    training = TrainingConfig(
        emulator_job_dir=_to_path(raw["training"]["emulator_job_dir"]),
        train_small_file=_to_path(raw["training"]["train_small_file"]),
        sbatch_cpus_per_task=int(raw["training"]["sbatch_cpus_per_task"]),
        sbatch_nodes=int(raw["training"]["sbatch_nodes"]),
        sbatch_time=raw["training"]["sbatch_time"],
        sbatch_partition=raw["training"]["sbatch_partition"],
        sbatch_gres=raw["training"]["sbatch_gres"],
        sbatch_constraint=raw["training"]["sbatch_constraint"],
        sbatch_mem=raw["training"]["sbatch_mem"],
        sbatch_modules=list(raw["training"]["sbatch_modules"]),
        conda_env=raw["training"]["conda_env"],
        logs_dir=_to_path(raw["training"]["logs_dir"]),
        eval_plots_dir=_to_path(raw["training"]["eval_plots_dir"]),
    )

    return AppConfig(
        data=data,
        model=model,
        active_learning=al,
        flags=flags,
        training=training,
    )