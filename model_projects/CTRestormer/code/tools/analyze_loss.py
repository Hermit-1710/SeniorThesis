import argparse
from pathlib import Path

import numpy as np


def mean_tail(values, window):
    if len(values) < window:
        return float(np.mean(values))
    return float(np.mean(values[-window:]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("loss_file", type=Path)
    args = parser.parse_args()

    values = np.load(args.loss_file).astype(np.float32)
    if len(values) == 0:
        raise SystemExit("empty loss file")

    print("loss_file:", args.loss_file)
    print("count:", len(values))
    print("first_500_mean:", mean_tail(values[:500], min(500, len(values))))
    for window in (2000, 1000, 500, 200, 100):
        if len(values) >= window:
            print("last_{}_mean:".format(window), mean_tail(values, window))
    if len(values) >= 1000:
        last = float(np.mean(values[-500:]))
        prev = float(np.mean(values[-1000:-500]))
        print("delta_last500_vs_prev500:", last - prev)
    if len(values) >= 200:
        last = float(np.mean(values[-100:]))
        prev = float(np.mean(values[-200:-100]))
        print("delta_last100_vs_prev100:", last - prev)
    print("min_loss:", float(np.min(values)))
    print("last_loss:", float(values[-1]))


if __name__ == "__main__":
    main()
