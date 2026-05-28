import numpy as np

from env import ACCEPT, REJECT, ApartmentEnv

ACTION_NAME = {REJECT: "reject", ACCEPT: "accept"}


def main(seed=0):
    env = ApartmentEnv(T=4, K=4, seed=seed)
    rng = np.random.default_rng(seed)

    obs, info = env.reset(seed=seed)
    print(f"{'t':>2}  {'U_t':>4}  {'action':>7}  {'reward':>6}  done")
    done = False
    while not done:
        t = int(round(float(obs[0])))
        u = info["true_u"]
        action = int(rng.integers(0, 2))
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(f"{t:>2}  {u:>4}  {ACTION_NAME[action]:>7}  {reward:>6.1f}  {done}")


if __name__ == "__main__":
    main()
