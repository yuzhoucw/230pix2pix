import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import argparse
import numpy as np

def plot(dict, figname="./checkpoints/results.png"):
    fig, (train_ax, val_ax) = plt.subplots(nrows=2, ncols=2, figsize=(20,10))

    plot_sub(train_ax, dict["train_loss"], "Train")
    plot_sub(val_ax, dict["val_loss"], "Val")

    fig.tight_layout()
    plt.savefig(figname)

def smooth(y, box_pts = 20):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def plot_sub(train_ax, losses, mode="Train"):
    ax0, ax1 = train_ax
    length = len(losses["G"])
    if mode == "Train":
        ite = np.arange(length)/1096.0
    else:
        ite = np.arange(length)/100.0

    ax0.plot(ite[0::10],smooth(losses["G"])[0::10], label="G")
    ax0.plot(ite[0::10],smooth(losses["G_L1"])[0::10], label="G_L1")
    ax0.plot(ite[0::10],smooth(losses["G_gan"])[0::10], label="G_gan")
    ax0.set_title("%s G Loss" % mode, fontsize=16, fontweight='bold')
    ax0.set_xlabel("Epoch",fontsize=16, fontweight='bold')
    ax0.set_ylabel("Loss",fontsize=16, fontweight='bold')
    ax0.set_ylim([-2,20])
    ax0.tick_params(axis='both', which='major', labelsize=16)
    ax0.legend(fontsize=16)

    ax1.plot(ite[0::10],smooth(losses["D"])[0::10], label="D")
    ax1.plot(ite[0::10],smooth(losses["D_real"])[0::10], label="D_real")
    ax1.plot(ite[0::10],smooth(losses["D_fake"])[0::10], label="D_fake")
    ax1.set_title("%s D Loss" % mode, fontsize=16, fontweight='bold')
    ax1.set_xlabel("Epoch", fontsize=16, fontweight='bold')
    ax1.set_ylabel("Loss", fontsize=16, fontweight='bold')
    ax1.set_ylim([-0.2,2])
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.legend(fontsize=16)


def print_epoch_ave(log_file):
    for set in ave_stats:
        for loss in ave_stats[set]:
            print("set = %s, %s = %f" %(set, loss, ave_stats[set][loss][-1]))

    # with open(log_file, "w") as f:
    #             json.dump(ave_stats, f)

parser = argparse.ArgumentParser()
parser.add_argument('--dir', default="", nargs='+', type=str)
args = parser.parse_args()

new_stats = {}
ave_stats = {}


for folder in args.dir:
    fname = os.path.join(folder, "train.json")
    with open(fname) as f:
        stats = json.load(f)
        sets = stats.keys()
        for set in sets:
            if set not in new_stats.keys():
                new_stats[set] = {}
                ave_stats[set] = {}
            for loss in stats[set].keys():
                if loss not in new_stats[set].keys():
                    new_stats[set][loss] = []
                    ave_stats[set][loss] = []

                new_stats[set][loss] += stats[set][loss]

                # calculate average loss in each epoch
                if set == "train_loss":
                    ave_loss = np.mean([ stats[set][loss][i:i + 1096] for i in range(0, len(stats[set][loss]), 1096) ], axis=1).tolist()
                if set == "val_loss":
                    ave_loss = np.mean([ stats[set][loss][i:i + 100] for i in range(0, len(stats[set][loss]), 100) ], axis=1).tolist()
                ave_stats[set][loss] += ave_loss


outname = os.path.join(args.dir[0], "result.png")
log_file = os.path.join(args.dir[0], "epoch_ave.json")
plot(new_stats, outname)
print_epoch_ave(log_file)

