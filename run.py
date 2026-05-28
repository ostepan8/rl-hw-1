import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from env import ApartmentEnv
from policies import OptimalPolicy, RandomPolicy, ThresholdPolicy

T, K = 4, 4
N = 10_000


class Result:
    def __init__(self, name, returns):
        self.name = name
        self.returns = returns

    @property
    def mean(self):
        return float(self.returns.mean())

    @property
    def se(self):
        return float(self.returns.std(ddof=1) / np.sqrt(len(self.returns)))

    @property
    def frac_rejected_all(self):
        return float((self.returns == 0.0).mean())


def run_policy(policy, n, noise_std, base_seed):
    """Run `policy` for n episodes; return the array of episode returns."""
    env = ApartmentEnv(T=T, K=K, noise_std=noise_std, seed=base_seed)
    returns = np.empty(n, dtype=np.float64)
    for i in range(n):
        obs, _ = env.reset(seed=base_seed + 1 + i)
        total = 0.0
        done = False
        while not done:
            action = policy.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        returns[i] = total
    return returns


def make_policies(noise_std):
    return {
        "Random (p=1/T)": RandomPolicy(T=T, seed=123),
        "Threshold(u_min=3)": ThresholdPolicy(u_min=3),
        "Optimal (1c)": OptimalPolicy(),
    }


def part_c():
    results = [
        Result(name, run_policy(pol, N, noise_std=0.0, base_seed=1000 + j * N))
        for j, (name, pol) in enumerate(make_policies(0.0).items())
    ]

    print(f"\n{'policy':>22}  {'mean':>7}  {'SE':>6}  {'rej-all':>8}")
    for r in results:
        print(f"{r.name:>22}  {r.mean:>7.4f}  {r.se:>6.4f}  {r.frac_rejected_all:>7.2%}")

    # Threshold sweep
    print("\nThresholdPolicy sweep (u_min):")
    print(f"{'u_min':>6}  {'mean':>7}  {'SE':>6}  {'rej-all':>8}")
    best_umin, best_mean = None, -np.inf
    for u_min in (1, 2, 3, 4):
        r = Result(
            f"thr{u_min}",
            run_policy(ThresholdPolicy(u_min), N, 0.0, base_seed=5000 + u_min * N),
        )
        print(f"{u_min:>6}  {r.mean:>7.4f}  {r.se:>6.4f}  {r.frac_rejected_all:>7.2%}")
        if r.mean > best_mean:
            best_mean, best_umin = r.mean, u_min
    print(f"-> best fixed threshold: u_min={best_umin} (mean {best_mean:.4f})")

    # Figure: three histograms side by side
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True)
    bins = np.arange(-0.5, K + 1.5, 1.0)
    for ax, r in zip(axes, results):
        ax.hist(r.returns, bins=bins, color="#4C72B0", edgecolor="white")
        ax.set_title(f"{r.name}\nmean={r.mean:.3f}")
        ax.set_xlabel("episode return")
        ax.set_xticks(range(0, K + 1))
    axes[0].set_ylabel("count")
    fig.suptitle(f"Return distributions, N={N}, T={T}, K={K}")
    fig.tight_layout()
    fig.savefig("returns.png", dpi=130)
    print("\nsaved figure -> returns.png")
    return results


def part_d():
    sigmas = (0.0, 0.5, 1.0, 2.0)
    rows = []
    header = f"{'sigma':>6}  " + "  ".join(f"{n:>20}" for n in make_policies(0.0))
    print(header)
    for sigma in sigmas:
        means = []
        for j, (name, pol) in enumerate(make_policies(sigma).items()):
            r = Result(name, run_policy(pol, N, noise_std=sigma, base_seed=9000 + j * N))
            means.append(r.mean)
        rows.append((sigma, *means))
        print(f"{sigma:>6.1f}  " + "  ".join(f"{m:>20.4f}" for m in means))

    with open("noise_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sigma", *make_policies(0.0).keys()])
        w.writerows(rows)
    print("saved table -> noise_table.csv")


if __name__ == "__main__":
    part_c()
    part_d()
