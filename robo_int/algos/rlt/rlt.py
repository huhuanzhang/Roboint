"""RLT（RL Token）：冻结 VLA + RL token 读出 + 轻量 actor-critic 在线 RL。

论文：RL Token: Bootstrapping Online RL with Vision-Language-Action Models
（C. Xu et al., 2026, Physical Intelligence, arXiv:2604.23073）。
**无官方代码**（RLinf 内实现已知有 bug，仅对照不作依赖）。
完整差异表见本目录 README.md，要点：

| 论文（v2 精读校准，2026-09-02） | 本实现（mock 级） |
|---|---|
| transformer 读出 RL token，示教数据预训练后冻结 | MLP 自编码，RL 在线联合训练（见 README 差异 1） |
| π0.6 真实 VLA 表征 + 本体感觉 | BaseVLA.extract_features（mock 下为冻结 MLP 特征） |
| actor 以 VLA 参考 chunk 为条件 + BC 正则，chunked C=10 | 单步动作，action_mode=direct/residual（差异 3，待补） |
| TD3 风格（双 Q min、无熵正则、UTD=5、稀疏奖励） | 标准 SAC（双 Q min + 熵正则 + 自动温度、UTD=2、dense 奖励；差异 4） |

训练流程（在线交互型算法，区别于 SFT 的 update(batch)）::

    algo = RLT(model, action_limit=0.15)
    obs = task.reset()
    for t in range(max_steps):
        a = algo.act(obs)                      # 预热期内部自动随机探索
        res = task.step(a)
        algo.observe(obs, a, res.reward, res.observation, res.terminated)
        if algo.ready:
            algo.update()                      # 内部采样 + UTD 次梯度步
        obs = res.observation
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from robo_int.algos.base import BaseAlgo
from robo_int.algos.common.nets import EnsembleCritic, TanhGaussianActor, mlp
from robo_int.algos.common.replay import SimpleReplayBuffer
from robo_int.models.base import BaseVLA


class RLTokenHead(nn.Module):
    """VLA 特征 -> RL token 的压缩读出头（+ 重建辅助损失）。

    论文用 encoder-decoder transformer；mock 级用 MLP 自编码代替，
    保留两个关键性质：低维 token（actor-critic 便宜）、重建辅助
    （token 不丢任务相关信息）。差异点 1/2，详见 README 表格。
    """

    def __init__(self, feature_dim: int, token_dim: int = 64):
        super().__init__()
        self.encoder = mlp([feature_dim, token_dim])
        self.decoder = mlp([token_dim, feature_dim])

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        return self.encoder(features)

    def reconstruction_loss(self, token: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(self.decoder(token), features)


class RLT(BaseAlgo):
    """RL Token 算法（在线交互型：act / observe / update）。

    Args:
        model: 冻结的 VLA 基模（需实现 extract_features）。
        action_limit: 任务侧安全限幅（动作最终被裁到 ±action_limit）。
        action_mode: direct=actor 直接输出动作；residual=actor 输出加在
            VLA 原动作上的残差。residual 下 critic 输入 concat base 动作
            [token, executed, base]（真机 VLA flow 采样随机，同一 token 下
            base 不同，Q(token, a) 缺 base 系统性模糊）；direct 模式 critic
            输入不变 [token, action]。两者均为本仓库设计选择——论文用的是
            第三种：actor 以 VLA 采样参考 chunk 为条件输入 + BC 正则
            （README 差异 3，actor 侧条件化待补）。
    """

    def __init__(
        self,
        model: BaseVLA,
        *,
        token_dim: int = 64,
        hidden_dim: int = 256,
        action_limit: float = 0.15,
        action_mode: str = "direct",
        gamma: float = 0.99,
        tau: float = 0.005,
        lr: float = 3e-4,
        aux_weight: float = 1.0,
        utd: int = 2,
        batch_size: int = 256,
        replay_capacity: int = 200_000,
        start_steps: int = 300,
        seed: int = 0,
    ):
        super().__init__()
        if action_mode not in ("direct", "residual"):
            raise ValueError(f"未知 action_mode={action_mode}；可选 direct / residual")
        self.model = model
        self.action_dim = model.action_dim
        self.action_limit = action_limit
        self.action_mode = action_mode
        self.gamma, self.tau, self.utd, self.batch_size = gamma, tau, utd, batch_size
        self.aux_weight = aux_weight
        self.start_steps = start_steps
        self.replay_capacity = replay_capacity

        self._torch_rng = torch.Generator().manual_seed(seed)
        self._np_rng = np.random.default_rng(seed)

        # 冻结基模：RLT 的核心设定（论文表 diff #2）
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

        self.token_head = RLTokenHead(model.feature_dim, token_dim)
        # 输出裕量 > 任务限幅，避免 tanh 饱和（见 algos/common/nets.py）
        self.actor = TanhGaussianActor(
            token_dim, self.action_dim, hidden=hidden_dim, action_margin=2.0 * action_limit
        )
        # critic 动作侧输入维度按模式定：residual=[executed, base]（2A），
        # direct=[action]（A）。direct 的构造序列与原实现完全一致，
        # 不静默改变其行为（RNG 消耗/初始化均不变）
        critic_action_dim = self.action_dim * (2 if action_mode == "residual" else 1)
        self.critic = EnsembleCritic(token_dim, critic_action_dim, num_qs=2, hidden=hidden_dim)
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        # 自动温度（目标熵 = -action_dim，SAC 标准取法）
        self.log_alpha = torch.tensor(np.log(0.2), dtype=torch.float32, requires_grad=True)
        self.target_entropy = -float(self.action_dim)

        # 三段式标准 SAC 优化器分工（曾用单一 "main" optimizer 联合 backward，
        # actor 梯度同时流进 critic / token_head，actor 靠搬动 token 表征虚增 Q，
        # critic 追不上表征漂移 → Q 值无锚定指数发散）：
        # - critic: token_head + critic（表征与 Q 回归同向，另有重建辅助锚定）
        # - actor:  仅 actor（Q 对其 detach，动作好才涨 Q，不能靠表征作弊）
        # - alpha:  温度
        self._optimizers = {
            "critic": Adam(
                list(self.token_head.parameters()) + list(self.critic.parameters()),
                lr=lr,
            ),
            "actor": Adam(self.actor.parameters(), lr=lr),
            "alpha": Adam([self.log_alpha], lr=lr),
        }

        self.buffer: SimpleReplayBuffer | None = None
        self._total_updates = 0
        # residual 模式：act() 时采样的 VLA 原动作（未限幅），observe() 缺省取走。
        # 每次进入 act 先置 None，防跨 transition 复用旧状态的 base
        self.last_base: np.ndarray | None = None

    # ------------------------------------------------------------------ 在线交互

    def _features(self, state: np.ndarray) -> torch.Tensor:
        """obs.state -> 冻结 VLA 特征（无梯度）。"""
        with torch.no_grad():
            feats = self.model.extract_features({"state": np.asarray(state, dtype=np.float32)})
        return feats.detach()

    def _token(self, state: np.ndarray) -> torch.Tensor:
        """obs.state -> RL token（无梯度版本，act 用）。"""
        with torch.no_grad():
            token = self.token_head.encode(self._features(state))
        return token

    def _predict_base_batch(self, states: np.ndarray) -> np.ndarray:
        """逐条采样 VLA 原动作，返回 (N, A)（predict_actions 是单观测接口）。

        真机 VLA flow 采样随机：每次调用都是 base 的新样本，target 端跨
        update 次数近似对 base 分布求期望（mock 下确定性，无差别）。
        真机部署时可换批量推理优化。
        """
        outs = [
            self.model.predict_actions({"state": np.asarray(s, dtype=np.float32)})[0]
            for s in np.asarray(states)
        ]
        return np.stack(outs).astype(np.float32)

    @torch.no_grad()
    def act(self, observation: dict, deterministic: bool = False) -> np.ndarray:
        state = observation["state"]
        self.last_base = None
        if self.action_mode == "residual":
            # base 是「状态侧上下文」而非动作：预热探索步也如实记录，
            # 否则 critic 在 residual 模式缺 base 输入（Q 系统性模糊）
            self.last_base = np.asarray(
                self.model.predict_actions(observation)[0], dtype=np.float32
            )
        if len(self.buffer or []) < self.start_steps and not deterministic:
            # 预热期：均匀随机探索（SAC 标准做法）
            return (
                self._np_rng.uniform(-self.action_limit, self.action_limit, self.action_dim)
                .astype(np.float32)
            )
        token = self._token(state)
        action, _, mean = self.actor(token)
        raw = (mean if deterministic else action).reshape(-1)  # 去掉 batch 维
        if self.action_mode == "residual":
            raw = torch.as_tensor(self.last_base) + raw
        return np.clip(raw.detach().numpy(), -self.action_limit, self.action_limit).astype(np.float32)

    def observe(self, state, action, reward, next_state, done, base_action=None) -> None:
        """单条 transition 入回放缓冲（在线交互循环每步调用）。

        base_action：residual 模式该步 act() 采样到的 VLA 原动作（未限幅）。
        标准循环 act -> step -> observe 顺序下缺省自动取 ``self.last_base``；
        direct 模式存零占位（回放 schema 统一，critic 不使用）。
        """
        if self.buffer is None:
            dim = int(np.asarray(state).shape[-1])
            self.buffer = SimpleReplayBuffer(
                self.replay_capacity,
                dim,
                self.action_dim,
                seed=int(self._np_rng.integers(1 << 31)),
            )
        if base_action is None:
            base_action = self.last_base
        if base_action is None:  # direct 模式 / residual 未经过 act 的手动灌数据
            base_action = np.zeros(self.action_dim, dtype=np.float32)
        self.buffer.add(state, action, reward, next_state, done, base_action=base_action)

    @property
    def ready(self) -> bool:
        return len(self.buffer or []) >= max(self.start_steps, self.batch_size)

    # ------------------------------------------------------------------ 更新

    def update(self, batch: dict | None = None) -> dict:
        """一次 SAC 梯度步。batch=None 时内部从回放采样（在线模式）。

        在线模式每步独立采样，跑 utd 次；外部传 batch（离线批模式）固定
        **只做 1 步**——同一批重复 utd 次梯度只是同方向放大步长，无采样
        多样性可言。
        """
        if not self.ready:
            return {"skipped": 1}
        metrics: dict = {}
        n_steps = self.utd if batch is None else 1
        for _ in range(n_steps):
            b = batch if batch is not None else self.buffer.sample(self.batch_size)
            step_metrics = self._sac_step(b)
            for k, v in step_metrics.items():
                metrics[k] = v  # 只留最后一次（监控够用）
            self._total_updates += 1
        return metrics

    def _sac_step(self, batch: dict) -> dict:
        state = torch.as_tensor(batch["observation.state"])
        next_state = torch.as_tensor(batch["next_observation.state"])
        action = torch.as_tensor(batch["action"])
        reward = torch.as_tensor(batch["reward"]).float()
        done = torch.as_tensor(batch["done"]).float()
        # residual 上下文；direct 零占位。外部批未带 base_action 时按零处理
        base = torch.as_tensor(
            batch.get("base_action", np.zeros_like(batch["action"], dtype=np.float32))
        ).float()
        residual = self.action_mode == "residual"

        # ---- 特征/token（VLA 冻结，只训练 token head 及之后的组件）
        features = self._features(state.numpy())
        token = self.token_head.encode(features)

        # ---- 1) critic + token_head：Q 回归 + 重建辅助（表征学习的锚）
        with torch.no_grad():
            next_token = self.token_head.encode(self._features(next_state.numpy()))
            next_a, next_logp, _ = self.actor(next_token)
            if residual:
                # s' 的 base 重新采一次：部署时 executed = clip(VLA(s') + 残差)，
                # target 必须在真实 base 分布上评估（fresh sample，no_grad）
                next_base = torch.as_tensor(self._predict_base_batch(next_state.numpy()))
                next_exec = torch.clamp(next_base + next_a, -self.action_limit, self.action_limit)
                q_next_in = torch.cat([next_exec, next_base], dim=-1)
            else:
                q_next_in = next_a
            q_next = torch.stack(self.critic_target(next_token, q_next_in)).min(dim=0).values
            target_q = reward + self.gamma * (1.0 - done) * (q_next - self.alpha * next_logp)
        # critic 动作侧输入：residual=[executed, base]（Markov 化），direct=[action]
        q_in = torch.cat([action, base], dim=-1) if residual else action
        q_losses = [
            nn.functional.mse_loss(q, target_q) for q in self.critic(token, q_in)
        ]
        aux_loss = self.token_head.reconstruction_loss(token, features)
        critic_loss = torch.stack(q_losses).mean() + self.aux_weight * aux_loss

        self._optimizers["critic"].zero_grad()
        critic_loss.backward()
        self._optimizers["critic"].step()

        # ---- 2) actor：token 必须 detach——actor 只能通过动作提升 Q，
        #      不能通过搬动 token 表征「作弊」（联合更新曾致 Q 无锚定发散）
        token_d = token.detach()
        pi_action, logp, _ = self.actor(token_d)
        if residual:
            # 与部署一致：executed = clip(base + 残差)。限幅区外梯度为零
            #（部署时同样被裁剪），residual 的探索需自行留在限幅区内
            executed_pi = torch.clamp(base + pi_action, -self.action_limit, self.action_limit)
            q_pi_in = torch.cat([executed_pi, base], dim=-1)
        else:
            q_pi_in = pi_action
        q_pi = torch.stack(self.critic(token_d, q_pi_in)).min(dim=0).values
        # alpha 在 actor 损失里 detach：否则 log_alpha 从 actor/alpha 两路收梯度
        actor_loss = (self.alpha.detach() * logp - q_pi).mean()
        self._optimizers["actor"].zero_grad()
        actor_loss.backward()
        self._optimizers["actor"].step()

        # ---- 3) alpha：温度自适应（目标熵 = -action_dim）
        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self._optimizers["alpha"].zero_grad()
        alpha_loss.backward()
        self._optimizers["alpha"].step()

        # ---- 目标网络软更新
        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
                pt.mul_(1.0 - self.tau).add_(self.tau * p)

        return {
            # loss 仅作监控聚合值（actor/critic 量纲不同，直接相加无优化意义）；
            # 分析训练曲线看分项 actor_loss / critic_loss / aux_loss
            "loss": float((actor_loss + critic_loss).detach()),
            "actor_loss": float(actor_loss.detach()),
            "critic_loss": float(critic_loss.detach()),
            "aux_loss": float(aux_loss.detach()),
            "q_mean": float(q_pi.mean().detach()),
            "alpha": float(self.alpha.detach()),
        }

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    # ------------------------------------------------------------------ 存取

    def save_state(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "token_head": self.token_head.state_dict(),
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "log_alpha": self.log_alpha.detach(),
                "total_updates": self._total_updates,
            },
            path,
        )

    def load_state(self, path: str | Path) -> None:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        self.token_head.load_state_dict(payload["token_head"])
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        self.critic_target.load_state_dict(payload["critic_target"])
        with torch.no_grad():
            self.log_alpha.copy_(payload["log_alpha"])
        self._total_updates = payload["total_updates"]
