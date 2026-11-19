"""Frames from three failed attempts that the general Robometer-4B passed with a score of 0.95 or more.

Videos are the openpi eval rollouts in results/libero_eval/; scores are from results/orientation_check/ (general model
on the stored view, and the LIBERO fine-tune on its mirrored training view). Four frames per attempt, start to end.

    uv run --no-project --with matplotlib --with imageio --with imageio-ffmpeg python analysis/plot_confident_mistakes.py
"""

import pathlib

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
V = ROOT / "results/libero_eval/pi05_libero_10trials"
CASES = [
    ("Open the middle drawer of the cabinet", V / "videos_libero_goal/rollout_open_the_middle_drawer_of_the_cabinet_failure.mp4",
     0.95, 0.23, "A drawer ends up open, but LIBERO's check says the middle one is not."),
    ("Put the yellow and white mug in the microwave and close it",
     V / "videos_libero_10/rollout_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_failure.mp4",
     0.96, 0.05, "The mug is off the table, but LIBERO's check says the door is not closed."),
    ("Put both moka pots on the stove", V / "videos_libero_10/rollout_put_both_moka_pots_on_the_stove_failure.mp4",
     0.96, 1.00, "Both pots end up crowded at the stove, but LIBERO's check says they are not both on it."),
]
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def main():
    fig, axes = plt.subplots(len(CASES), 4, figsize=(10, 9.4), facecolor=SURFACE,
                             gridspec_kw={"hspace": 0.72, "wspace": 0.04})
    for row, (task, path, general, tuned, what) in zip(axes, CASES):
        frames = iio.imread(path)
        for j, (ax, i) in enumerate(zip(row, np.linspace(0, len(frames) - 1, 4).astype(int))):
            ax.imshow(frames[i])
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ax.spines.values():
                side.set_visible(False)
            ax.set_xlabel(["Start", "", "", "End"][j], color=INK2, fontsize=9, labelpad=3)
        row[0].set_title(f'"{task}"', loc="left", color=INK, fontsize=10.5, fontweight="bold", pad=34)
        row[0].text(0, 1.15, f"LIBERO: failed.   Robometer: {general:.2f}.   LIBERO fine-tune: {tuned:.2f}.",
                    transform=row[0].transAxes, color=INK, fontsize=9.5, va="bottom")
        row[0].text(0, 1.03, what, transform=row[0].transAxes, color=INK2, fontsize=9.5, va="bottom")
    fig.suptitle("Failed attempts Robometer-4B scored as near-certain successes", x=0.125, ha="left", y=0.985,
                 color=INK, fontsize=13, fontweight="bold")
    fig.text(0.125, 0.012, "Scores are success probabilities from 0 to 1; above 0.5 counts as a pass. Robometer: general "
             "model, stored view. Fine-tune: jesbu1/robometer-4b-fft-libero, mirrored view.", color=INK2, fontsize=8)
    out = ROOT / "results/orientation_check/confident_mistakes.png"
    fig.savefig(out, dpi=170, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
