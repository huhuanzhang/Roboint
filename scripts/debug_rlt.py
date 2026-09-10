"""RLT 学习诊断：打印训练过程中的成功率/q 值/alpha/aux_loss 轨迹。"""

import numpy as np
import torch

from robo_int.algos.rlt import RLT
from robo_int.models.mock import MockVLA
from robo_int.tasks.mock_reach import MockReachTask

torch.manual_seed(3)
model = MockVLA(state_dim=4, action_dim=2, hidden_dim=128)
algo = RLT(model, action_limit=0.15, seed=3, hidden_dim=128, batch_size=128)
task = MockReachTask()
rng = np.random.default_rng(100)

steps = 0
hist = []
m = {}
while steps < 3500:
    obs = task.reset(seed=int(rng.integers(1 << 31)))
    ep_ret, success = 0.0, False
    for _ in range(task.max_episode_steps):
        a = algo.act(obs)
        res = task.step(a)
        algo.observe(obs["state"], a, res.reward, res.observation["state"], res.terminated)
        obs = res.observation
        ep_ret += res.reward
        steps += 1
        if algo.ready:
            m = algo.update()
        if res.terminated:
            success = True
            break
        if res.truncated:
            break
    hist.append((steps, ep_ret, success, m.get("q_mean", float("nan")), m.get("alpha", float("nan")), m.get("aux_loss", float("nan")), m.get("critic_loss", float("nan"))))

for row in hist[::3]:
    print(f"steps={row[0]:5d} ret={row[1]:7.2f} succ={int(row[2])} q={row[3]:7.3f} alpha={row[4]:.3f} aux={row[5]:.4f} critic={row[6]:.4f}")
