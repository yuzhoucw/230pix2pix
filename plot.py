import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import argparse

def plot(dict, figname="./checkpoints/results.png"):
    fig, (train_ax, val_ax) = plt.subplots(nrows=2, ncols=3, figsize=(30,10))

    plot_sub(train_ax, stats["train_loss"], "Train")
    plot_sub(val_ax, stats["val_loss"], "Val")

    fig.tight_layout()
    plt.savefig(figname)

def plot_sub(train_ax, losses, mode="Train"):
    ax0, ax1, ax2 = train_ax
    ax0.plot(losses["G"], label="G")
    ax0.plot(losses["G_L1"], label="G_L1")
    ax0.plot(losses["G_gan"], label="G_gan")
    ax0.set_title("%s G Loss" % mode)
    ax0.set_xlabel("Iterations")
    ax0.set_ylabel("Loss")
    ax0.legend()

    ax1.plot(losses["D"], label="D")
    ax1.plot(losses["D_real"], label="D_real")
    ax1.plot(losses["D_fake"], label="D_fake")
    ax1.set_title("%s D Loss" % mode)
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(losses["G"], label="G")
    ax2.plot(losses["D"], label="D")
    ax2.set_title("%s G, D Loss" % mode)
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Loss")
    ax2.legend()

parser = argparse.ArgumentParser()
parser.add_argument('--dir', default="", type=str)
args = parser.parse_args()

fname = os.path.join(args.dir, "train.json")
outname = os.path.join(args.dir, "result.png")
with open(fname) as f:
    stats = json.load(f)
    plot(stats, outname)
