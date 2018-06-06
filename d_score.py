import matplotlib.pyplot as plt
import json

with open("checkpoints/d_scores.json") as f:
    j = json.load(f)
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(13,5))

    x = j['scores_gen']
    # x = sorted(j['scores_gen'], reverse=True)
    ax1.plot(x, '.')
    ax1.set_title('Train D Scores gen')
    ax1.axhline(y=0.5, color='g')

    x = j['scores_gt']
    # x = sorted(j['scores_gt'], reverse=True)
    ax2.plot(x, '.')
    ax2.set_title('Train D Scores gt')
    ax2.axhline(y=0.5, color='g')

    fig.tight_layout()
    # plt.show()

    plt.savefig("checkpoints/d_scores_1.png")