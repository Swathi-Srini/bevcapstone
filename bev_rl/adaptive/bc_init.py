"""Behaviour-clone the privileged IDM expert on the teacher's observation, to warm-start PPO (avoids the stand-still optimum)."""
import numpy as np
import torch
from torch import nn


def main(n_eps=150, out="bev_rl/adaptive/bc_init.pt"):
    import ultralytics  # noqa
    from bev_rl.adaptive.env import AdaptiveEnv
    conds = [("clear", 0.0), ("fog", 0.3), ("fog", 0.6), ("fog", 1.0), ("rain", 0.3), ("rain", 0.6), ("rain", 1.0)]
    e = AdaptiveEnv(range(2000, 3000), conds, expert=True, sample_seed=7, horizon=500)
    X, Y = [], []
    for ep in range(n_eps):
        o, _ = e.reset()
        while True:
            a_obs = o
            o, r, te, tr, i = e.step([0.0, 0.0])
            X.append(a_obs); Y.append(e.prev_action.copy())
            if te or tr:
                break
    X, Y = np.array(X, np.float32), np.array(Y, np.float32)
    print("samples", len(X), flush=True)
    net = nn.Sequential(nn.Linear(12, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 2))
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    Xt, Yt = torch.tensor(X), torch.tensor(Y)
    idx = np.random.default_rng(0).permutation(len(X)); nv = len(X) // 10
    va, tr_ = idx[:nv], idx[nv:]
    for epoch in range(60):
        np.random.shuffle(tr_)
        for b in range(0, len(tr_), 256):
            j = tr_[b:b + 256]
            loss = ((net(Xt[j]) - Yt[j]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if epoch % 10 == 9:
            with torch.no_grad():
                print(epoch, "val mse", ((net(Xt[va]) - Yt[va]) ** 2).mean().item(), flush=True)
    torch.save(net.state_dict(), out)
    e.close()


if __name__ == "__main__":
    main()
