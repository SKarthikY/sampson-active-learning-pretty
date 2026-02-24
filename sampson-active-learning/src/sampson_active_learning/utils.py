# src/sampson_active_learning/utils.py

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence

import glob
import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
from .simple_MLP import *
from sedoNNa.utils import (compute_photometry_torch, 
                           get_lc_from_file,
                           get_rise_time,
                           get_fall_time)
from astropy import units as u


'''sedoNNa is a package I built for a related project a while back. 
It's unfortunate that file is also called utils.

'''

from helpers import *
'''
helpers is another package I've built just as a series of helper functions 
that I've amassed throughout grad school. In particular, I invented up a
bunch of colors and color schemes. sKy_colors is a variable that holds my invented
colors (sky is my initials), and I'm very happy to share these colors, but that is 
definitely not a technically relevant or useful piece of code, haha

u

'''
from .config import AppConfig


jansky = 1.0E-23 * u.erg/(u.second*(u.cm)**2*hz)

def get_lc_properties(LC_file: str, filt: str = "Lbol(erg/s)"):
    """
    Compute basic light-curve properties from a Sedona lightcurve file:
      - peak luminosity
      - time of peak
      - rise time
      - fall time

    Parameters
    ----------
    LC_file : str
        Path to the lightcurve file.
    filt : str
        Filter name, e.g. 'Lbol(erg/s)' or 'SDSS_r'.

    Returns
    -------
    peak_lum, peak_time, rise_time, fall_time : floats or None
    """
    lc, times = get_lc_from_file(LC_file, filt=filt)
    if lc is None:
        return None, None, None, None

    if "SDSS" in filt:
        # Convert magnitudes to luminosity density
        lc = (
            10 ** (-1.0 * lc / 2.5)
            * 3631
            * jansky
            * 4
            * np.pi
            * (10 * u.pc) ** 2
        ).decompose().to(u.erg / (u.second * u.Hz)).value

    peak_lum = np.max(lc)
    peak_time = times[find_nearest(lc, peak_lum)]
    rise_time = get_rise_time(times, lc)
    fall_time = get_fall_time(times, lc)

    return peak_lum, peak_time, rise_time, fall_time


def evaluate_model(
    cfg: AppConfig,
    ckpt: str,
    epochs: int,
    sample_ids: Iterable,
    wav: torch.Tensor,
) -> None:
    """
    Evaluate a trained emulator checkpoint on a set of sample_ids and:

      - write per-sample light curves (predicted)
      - compute light-curve properties
      - write lc_properties.txt
      - generate diagnostic plots

    All outputs go under:
        cfg.training.eval_plots_dir / ckpt / epochs

    lc_properties.txt is written at:
        cfg.data.lc_properties_file.format(model=ckpt, epochs=epochs)

    Parameters
    ----------
    cfg : AppConfig
        Global configuration object.
    ckpt : str
        Checkpoint name (subdirectory under cfg.model.ckpt_dir).
    epochs : int
        Epoch number (used in filename "{epochs}.pth").
    sample_ids : iterable
        List or set of sample IDs to evaluate.
    wav : torch.Tensor
        Wavelength grid (1D tensor) for the spectra.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Load normalization statistics
    # ------------------------------------------------------------------
    mean_std_dict = torch.load(cfg.data.normalization_stats_file, weights_only=False)
    fluxes_mean = mean_std_dict["fluxes_mean"]
    fluxes_std = mean_std_dict["fluxes_std"]
    time_mean = mean_std_dict["time_mean"]
    time_std = mean_std_dict["time_std"]
    descriptor_mean = mean_std_dict["descriptor_mean"].float().to(device)
    descriptor_std = mean_std_dict["descriptor_std"].float().to(device)

    # ------------------------------------------------------------------
    # Load model checkpoint
    # ------------------------------------------------------------------
    ckpt_path = Path(cfg.model.ckpt_dir) / ckpt / f"{epochs}.pth"
    model = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.eval()
    criterion = nn.MSELoss()  # currently unused, but kept for clarity

    # ------------------------------------------------------------------
    # Photometry filters and constants
    # ------------------------------------------------------------------
    d = (10 * u.parsec).to(u.cm).value
    denom = 4 * np.pi * d**2.0

    filter_prof_dir = "/n/home07/kyadavalli/scratch/NeuralNetworks/for_github/filter_profs/"
    filters = {
        "u": "SLOAN_SDSS.u",
        "g": "SLOAN_SDSS.g",
        "r": "SLOAN_SDSS.r",
        "i": "SLOAN_SDSS.i",
        "z": "SLOAN_SDSS.z",
    }
    filter_tensors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for fil, fname in filters.items():
        filter_data = np.loadtxt(f"{filter_prof_dir}/{fname}.dat")
        filter_wavs = filter_data[:, 0]
        filter_transmission = filter_data[:, 1]
        fw_t = torch.as_tensor(filter_wavs, dtype=torch.float64, device=device)
        ft_t = torch.as_tensor(filter_transmission, dtype=torch.float64, device=device)
        filter_tensors[fil] = (fw_t, ft_t)

    all_filters = ["u", "g", "r", "i", "z"]
    plot_filters = ["g", "r", "i", "z"]


    bbox_props = dict(
        boxstyle="round,pad=0.2",
        facecolor="white",
        edgecolor="black",
        linewidth=0.5,
        alpha=1.0,
    )

    # ------------------------------------------------------------------
    # Output locations
    # ------------------------------------------------------------------
    eval_root = cfg.training.eval_plots_dir  # e.g. "output/eval"
    save_dir = eval_root / ckpt / str(epochs)
    mkdir(str(eval_root/ckpt))
    mkdir(str(save_dir))

    # lc_properties path driven by config pattern "output/eval/{model}/{epochs}/lc_properties.txt"
    lc_properties_path = Path(
        cfg.data.lc_properties_file.format(model=ckpt, epochs=epochs)
    )
    mkdir(str(lc_properties_path.parent))

    figs_dir = save_dir / "figs"
    # Early exit if we've already done this evaluation
    if lc_properties_path.exists() and (figs_dir / "max_ft.pdf").exists():
        print(f"Evaluation already done for ckpt={ckpt}, epoch={epochs}. Skipping...")
        return

    print(f"Working on ckpt: {ckpt}, epoch: {epochs}")

    # ------------------------------------------------------------------
    # Loop over samples, generate light curves and per-sample plots
    # ------------------------------------------------------------------
    write_to_file(
        str(lc_properties_path),
        "# Sample\tAvg_Mag_Diff\tMax_Mag_Diff\tPeak_Lum\tPeak_Time\tRise_Time\tFall_Time\n",
        append=False,
    )

    for sid in tqdm(list(sample_ids)):
        # per-sample output directory
        sample_dir = save_dir / str(sid)
        mkdir(str(sample_dir))

        out_file = sample_dir / "lc.txt"

        # Header row
        header = "# Time (Days)\tLbol (erg/s)\tMbol"
        for fil in all_filters:
            header += f"\tSDSS_{fil}"
        header += "\n"
        write_to_file(str(out_file), header, append=False)

        # Locate Sedona sample.txt
        sample_txt_path = f"/n/home07/kyadavalli/scratch/aCOperation/models_4d_0.2dt/{sid}/sample.txt"
        if "supp" in str(sid):
            sample_txt_path = (
                "/n/home07/kyadavalli/scratch/NeuralNetworks/NN_grid/"
                f"active_learning/supplemental_grid/{str(sid)[5:]}/sample.txt"
            )
        if not os.path.exists(sample_txt_path):
            print(f"sample txt not found for sample id: {sid}")
            continue

        sample_df = pd.read_csv(sample_txt_path, sep=",")

        # ------------------------------------------------------------------
        # Predict light curve with emulator
        # ------------------------------------------------------------------
        with torch.no_grad():
            param_tensor = torch.tensor(
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
            ).unsqueeze(0)

            param_tensor = (param_tensor - descriptor_mean) / descriptor_std

            for t in np.arange(7.5 * 86400, 40.5 * 86400, 1.0 * 86400):
                time_tensor = torch.tensor([[t]], dtype=torch.float32, device=device)
                time_tensor = (time_tensor - time_mean) / time_std
                input_tensor = torch.cat((param_tensor, time_tensor), dim=1)

                output = model(input_tensor).to(torch.float64)
                spec_pred = 10.0 ** (output * fluxes_std + fluxes_mean)

                lbol = torch.trapz(spec_pred.squeeze(), wav)
                mbol = 4.74 - 2.5 * torch.log10(lbol / 3.83e33)

                spec_pred /= denom

                row = f"{t/86400}\t{lbol}\t{mbol}"
                for fil in all_filters:
                    pred_mag_batch = compute_photometry_torch(
                        wav, spec_pred, fw_t, ft_t
                    )
                    row += f"\t{pred_mag_batch.item()}"
                row += "\n"
                write_to_file(str(out_file), row, append=True)

        # ------------------------------------------------------------------
        # Compare emulator vs Sedona light curves and make plots
        # ------------------------------------------------------------------
        import matplotlib.pyplot as plt

        dir1 = "/n/home07/kyadavalli/scratch/aCOperation/models_4d_0.2dt/"
        if "supp" in str(sid):
            dir1 = (
                "/n/home07/kyadavalli/scratch/NeuralNetworks/NN_grid/"
                "active_learning/supplemental_grid/"
            )

        # 1) Multi-band LC comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = np.array(axes).reshape(2, 2)

        for idx_f, filt in enumerate(plot_filters):
            row = idx_f // 2
            col = idx_f % 2
            ax = axes[row, col]

            if "supp" in str(sid):
                sed_lc, sed_times = get_lc_from_file(
                    dir1 + f"{str(sid)[5:]}/lightcurve.out", filt=f"SDSS_{filt}"
                )
            else:
                sed_lc, sed_times = get_lc_from_file(
                    dir1 + f"{sid}/lightcurve.out", filt=f"SDSS_{filt}"
                )
            if sed_lc is None:
                print(f"Light curve not found for sample id: {sid}")
                continue
            lc, times = get_lc_from_file(str(out_file), filt=f"SDSS_{filt}")
            if lc is None:
                print(f"Predicted light curve not found for sample id: {sid}")
                continue

            idx = np.where((times > 7) & (times < 40))
            sed_idx = np.where((sed_times > 7) & (sed_times < 40))
            avg_mag_diff = str(round(np.abs(lc[idx] - sed_lc[sed_idx]).mean(), 2))
            max_mag_diff = str(round(np.abs(lc[idx] - sed_lc[sed_idx]).max(), 2))

            ax.plot(
                sed_times,
                sed_lc,
                label=f"True {filt} LC",
                color=sKy_colors["blue"],
                alpha=0.7,
                linewidth=4,
            )
            ax.plot(
                times,
                lc,
                label=f"Predicted {filt} LC",
                color=sKy_colors["green"],
                alpha=0.7,
                linewidth=4,
            )

            if row == 0:
                ax.set_xlabel("")
                ax.set_xticks([])
            else:
                ax.set_xlabel("Time (Days)", fontsize=18)

            ax.set_ylabel("", fontsize=18)

            ul = np.min([np.min(sed_lc), np.min(lc)])
            ax.set_ylim([ul + 5, ul - 0.3])

            ax.set_title(
                f"{filt}-band, Avg|Max Mag Diff: {avg_mag_diff} | {max_mag_diff}",
                fontsize=25,
            )
            ax.legend(fontsize=12)

            if filt == "r":
                if "supp" in str(sid):
                    lc_file = dir1 + f"{str(sid)[5:]}/lightcurve.out"
                else:
                    lc_file = dir1 + f"{sid}/lightcurve.out"

                peak_lum, peak_time, rise_time, fall_time = get_lc_properties(
                    lc_file, filt="SDSS_r"
                )

                # write sample-level metrics
                write_to_file(
                    str(lc_properties_path),
                    f"{sid}\t{avg_mag_diff}\t{max_mag_diff}\t"
                    f"{peak_lum}\t{peak_time}\t{rise_time}\t{fall_time}\n",
                    append=True,
                )

        fig.suptitle(f"Sample ID: {sid}", fontsize=24)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        fig_path = sample_dir / "multi_band_lc.pdf"
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)

        # 2) Mag-diff plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = np.array(axes).reshape(2, 2)

        for idx_f, filt in enumerate(plot_filters):
            row = idx_f // 2
            col = idx_f % 2
            ax = axes[row, col]

            if "supp" in str(sid):
                sed_lc, sed_times = get_lc_from_file(
                    dir1 + f"{str(sid)[5:]}/lightcurve.out", filt=f"SDSS_{filt}"
                )
            else:
                sed_lc, sed_times = get_lc_from_file(
                    dir1 + f"{sid}/lightcurve.out", filt=f"SDSS_{filt}"
                )
            if sed_lc is None:
                continue

            lc, times = get_lc_from_file(str(out_file), filt=f"SDSS_{filt}")
            if lc is None:
                continue

            idx = np.where((times > 7) & (times < 40))
            sed_idx = np.where((sed_times > 7) & (sed_times < 40))
            avg_mag_diff = str(round((lc[idx] - sed_lc[sed_idx]).mean(), 2))
            max_mag_diff = str(round(np.abs(lc[idx] - sed_lc[sed_idx]).max(), 2))
            mag_diff = lc[idx] - sed_lc[sed_idx]

            ax.plot(
                sed_times[sed_idx],
                mag_diff,
                label=f"Mag Diff {filt} LC",
                color=sKy_colors["blue"],
                alpha=0.7,
                linewidth=4,
            )

            if row == 0:
                ax.set_xlabel("")
                ax.set_xticks([])
            else:
                ax.set_xlabel("Time (Days)", fontsize=18)

            ax.set_ylabel("", fontsize=18)

            ax.set_title(
                f"{filt}-band, Avg|Max Mag Diff: {avg_mag_diff} | {max_mag_diff}",
                fontsize=25,
            )
            ax.legend(fontsize=12)

        fig.suptitle(f"Sample ID: {sid}", fontsize=24)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        fig_path = sample_dir / "mag_diff.pdf"
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # After processing all samples, make summary plots over lc_properties.txt
    # ------------------------------------------------------------------
    lc_prop_dict = read_file(str(lc_properties_path))["dict"]

    sids = np.asarray(lc_prop_dict[0])
    avg_mag_diff = np.asarray(lc_prop_dict[1], dtype=float)
    max_mag_diff = np.asarray(lc_prop_dict[2], dtype=float)
    rise_times = np.asarray(lc_prop_dict[5], dtype=float)
    fall_times = np.asarray(lc_prop_dict[6], dtype=float)

    mkdir(str(figs_dir))

    import matplotlib.pyplot as plt

    # avg mag diff vs fall time
    plt, _, _ = get_pretty_plot()
    plt.xlabel("Fall Time (d)", fontsize=35)
    plt.ylabel("Average Mag Diff", fontsize=35)
    for i in range(len(fall_times)):
        if np.abs(avg_mag_diff[i]) > 0.25:
            plt.text(
                fall_times[i],
                avg_mag_diff[i],
                f"{sids[i]}",
                fontsize=12,
                ha="center",
                va="bottom",
                weight="bold",
                bbox=bbox_props,
            )
    plt.scatter(fall_times, avg_mag_diff, s=300, color=sKy_colors["blue"])
    plt.savefig(figs_dir / "avg_ft.pdf", bbox_inches="tight")
    plt.close()

    # max mag diff vs fall time
    plt, _, _ = get_pretty_plot()
    plt.xlabel("Fall Time (d)", fontsize=35)
    plt.ylabel("Maximum Mag Diff", fontsize=35)
    for i in range(len(fall_times)):
        if np.abs(max_mag_diff[i]) > 0.25:
            plt.text(
                fall_times[i],
                max_mag_diff[i],
                f"{sids[i]}",
                fontsize=12,
                ha="center",
                va="bottom",
                weight="bold",
                zorder=100,
                bbox=bbox_props,
            )
    plt.scatter(fall_times, max_mag_diff, s=300, color=sKy_colors["blue"])
    plt.savefig(figs_dir / "max_ft.pdf", bbox_inches="tight")
    plt.close()

    # avg mag diff vs rise time
    plt, _, _ = get_pretty_plot()
    plt.xlabel("Rise Time (d)", fontsize=35)
    plt.ylabel("Average Mag Diff", fontsize=35)
    for i in range(len(rise_times)):
        if np.abs(avg_mag_diff[i]) > 0.25 or rise_times[i] < 10:
            plt.text(
                rise_times[i],
                avg_mag_diff[i],
                f"{sids[i]}",
                fontsize=12,
                ha="center",
                va="bottom",
                weight="bold",
                zorder=100,
                bbox=bbox_props,
            )
    plt.scatter(rise_times, avg_mag_diff, s=300, color=sKy_colors["blue"])
    plt.savefig(figs_dir / "avg_rt.pdf", bbox_inches="tight")
    plt.close()

    # max mag diff vs rise time
    plt, _, _ = get_pretty_plot()
    plt.xlabel("Rise Time (d)", fontsize=35)
    plt.ylabel("Maximum Mag Diff", fontsize=35)
    for i in range(len(rise_times)):
        if np.abs(max_mag_diff[i]) > 0.25:
            plt.text(
                rise_times[i],
                max_mag_diff[i],
                f"{sids[i]}",
                fontsize=12,
                ha="center",
                va="bottom",
                weight="bold",
                zorder=100,
                bbox=bbox_props,
            )
    plt.scatter(rise_times, max_mag_diff, s=300, color=sKy_colors["blue"])
    plt.savefig(figs_dir / "max_rt.pdf", bbox_inches="tight")
    plt.close()

    # Histogram of fall times
    plt, _, _ = get_pretty_plot()
    plt.xlabel("Fall Time (d)", fontsize=35)
    plt.hist(fall_times, color=sKy_colors["blue"], density=True)
    plt.savefig(figs_dir / "ft_hist.pdf", bbox_inches="tight")
    plt.close()

    # Histogram of rise times
    plt, _, _ = get_pretty_plot()
    plt.xlabel("Rise Time (d)", fontsize=35)
    plt.hist(rise_times, color=sKy_colors["blue"], density=True)
    plt.savefig(figs_dir / "rt_hist.pdf", bbox_inches="tight")
    plt.close()