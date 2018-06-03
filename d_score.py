import matplotlib.pyplot as plt
import json

with open("checkpoints/d_scores.json") as f:
    j = json.load(f)
    plt.plot(j['scores_gen'], '.')
    plt.title('D Scores gen')
    plt.show()

    plt.plot(j['scores_gt'], '.')
    plt.title('D Scores gt')
    plt.show()