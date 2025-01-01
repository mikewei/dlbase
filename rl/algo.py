from typing import Callable, Dict, Tuple, NamedTuple, Literal, Optional
from itertools import chain
from gymnasium.spaces import Box, Discrete
import torch
from torch import nn, optim, Tensor
from torch.utils import data
from torch.nn import functional as F
from torch.nn.utils import clip_grad
from dlbase.utils.py import save_params
from .gyms import GymAlgo, GymMemory, Env, VectorEnv

class DActor(nn.Module):
    # states -> actions
    def forward(self, s: Tensor) -> Tensor:
        ...

class PActor(nn.Module):
    # states -> (actions_mean, actions_std)
    def forward(self, s: Tensor) -> Tuple[Tensor, Tensor]:
        ...

class QCritic(nn.Module):
    # (states, actions) -> Q-values
    def forward(self, s: Tensor, a: Tensor) -> Tensor:
        ...

class VCritic(nn.Module):
    # states -> V-values
    def forward(self, s: Tensor) -> Tensor:
        ...

class DefaultDActor(DActor):
    def __init__(self, state_dim, action_dim, hidden_dim=128, hidden_multiplier=2):
        super().__init__()
        self.layer_1 = nn.Linear(state_dim, hidden_dim * hidden_multiplier)
        self.layer_2 = nn.Linear(hidden_dim * hidden_multiplier, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, action_dim)
    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        return F.tanh(self.layer_3(x))
class DefaultQCritic(QCritic):
    def __init__(self, state_dim, action_dim, hidden_dim=128, hidden_multiplier=2):
        super().__init__()
        self.layer_1 = nn.Linear(state_dim + action_dim, hidden_dim * hidden_multiplier)
        self.layer_2 = nn.Linear(hidden_dim * hidden_multiplier, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, 1)
    def forward(self, x, a):
        xa = torch.cat([x, a], 1)
        xa = F.relu(self.layer_1(xa))
        xa = F.relu(self.layer_2(xa))
        return self.layer_3(xa)

class TD3(GymAlgo):
    class Params(NamedTuple):
        device: str = 'cuda'
        batch_size: int = 128
        lr: float = 1e-4
        gamma: float = 0.99  # gamma通常取[0.95, 0.99]
        tau: float = 0.005   # tau通常取[0.005, 0.01]
        actor_update_freq: int = 2
        explore_std_fn: Callable[[int], float] = lambda t: 0.1
        noise_std: float = 0.1
        noise_clip: float = 0.5
        grad_clip: float = 100
    class TrainStats(NamedTuple):
        critic_loss: float

    def __init__(self, factory: Callable[[], Tuple[DActor, QCritic, QCritic]], **params):
        self.hp = self.Params(**params)
        self.update_count = 0
        self.steps_count = 0
        self.actor, self.critic_1, self.critic_2 = (m.to(self.hp.device) for m in factory())
        self.target_actor, self.target_critic_1, self.target_critic_2 = (m.to(self.hp.device) for m in factory())
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.hp.lr, amsgrad=True)
        self.critic_optim = optim.Adam(chain(self.critic_1.parameters(), self.critic_2.parameters()), lr=self.hp.lr, amsgrad=True)
    def register(self, env: Env | VectorEnv) -> Optional[Dict]:
        if not isinstance(env.action_space, Box):
            raise ValueError("TD3 only supports continuous action spaces")
        if isinstance(env, VectorEnv):
            self.action_min = torch.tensor(env.action_space.low, device=self.hp.device)[0]
            self.action_max = torch.tensor(env.action_space.high, device=self.hp.device)[0]
        else:
            self.action_min = torch.tensor(env.action_space.low, device=self.hp.device)
            self.action_max = torch.tensor(env.action_space.high, device=self.hp.device)
        return None
    def get_device(self):
        return self.hp.device
    def select_actions(self, states: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        return self._select_actions(states, self.actor, self.hp.explore_std_fn(self.steps_count)), {}
    def _select_actions(self, states, actor, noise_std=0.0, noise_clip=None):
        with torch.no_grad():
            actor.eval()
            actions = actor(states)
            noise = torch.randn_like(actions) * noise_std
            if noise_clip is not None:
                noise = noise.clamp(-noise_clip, noise_clip)
            actions = actions + noise
            actions = actions.clamp(self.action_min, self.action_max)
        return actions
    def update(self, memory: GymMemory, dones: Tensor) -> Optional[TrainStats]:
        if memory.size() < self.hp.batch_size:
            return None
        batch, _ = memory.sample(self.hp.batch_size)
        non_final_mask = ~batch.terms
        non_final_next_states = batch.next_states[non_final_mask]
        # update critic
        self.critic_1.train()
        self.critic_2.train()
        qvalues_1 = self.critic_1(batch.states, batch.actions)
        qvalues_2 = self.critic_2(batch.states, batch.actions)
        next_qvalues = torch.zeros_like(qvalues_1)
        with torch.no_grad():
            self.target_critic_1.eval()
            self.target_critic_2.eval()
            non_final_next_actions = self._select_actions(non_final_next_states, self.target_actor, self.hp.noise_std, self.hp.noise_clip)
            next_qvalues[non_final_mask] = torch.min(self.target_critic_1(non_final_next_states, non_final_next_actions),
                                                     self.target_critic_2(non_final_next_states, non_final_next_actions))
            target_qvalues = batch.rewards + self.hp.gamma * next_qvalues
        critic_loss = F.smooth_l1_loss(qvalues_1, target_qvalues) + F.smooth_l1_loss(qvalues_2, target_qvalues)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        clip_grad.clip_grad_norm_(self.critic_1.parameters(), self.hp.grad_clip)
        clip_grad.clip_grad_norm_(self.critic_2.parameters(), self.hp.grad_clip)
        self.critic_optim.step()
        # soft update target critic
        # θ′ ← τ θ + (1 −τ )θ′
        critic_1_state_dict = self.critic_1.state_dict()
        critic_2_state_dict = self.critic_2.state_dict()
        target_critic_1_state_dict = self.target_critic_1.state_dict()
        target_critic_2_state_dict = self.target_critic_2.state_dict()
        for key in critic_1_state_dict:
            target_critic_1_state_dict[key] = critic_1_state_dict[key]*self.hp.tau + target_critic_1_state_dict[key]*(1-self.hp.tau)
            target_critic_2_state_dict[key] = critic_2_state_dict[key]*self.hp.tau + target_critic_2_state_dict[key]*(1-self.hp.tau)
        self.target_critic_1.load_state_dict(target_critic_1_state_dict)
        self.target_critic_2.load_state_dict(target_critic_2_state_dict)
        # update actor
        if self.update_count % self.hp.actor_update_freq == 0:
            self.actor.train()
            self.critic_1.eval()
            actor_loss = -self.critic_1(batch.states, self.actor(batch.states)).mean()
            self.critic_optim.zero_grad()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            clip_grad.clip_grad_norm_(self.actor.parameters(), self.hp.grad_clip)
            self.actor_optim.step()
            # soft update target actor
            target_actor_state_dict = self.target_actor.state_dict()
            actor_state_dict = self.actor.state_dict()
            for key in actor_state_dict:
                target_actor_state_dict[key] = actor_state_dict[key]*self.hp.tau + target_actor_state_dict[key]*(1-self.hp.tau)
            self.target_actor.load_state_dict(target_actor_state_dict)
        self.update_count += 1
        self.steps_count += len(dones)
        return self.TrainStats(critic_loss.item() / 2)

class PPO(GymAlgo):
    class Params(NamedTuple):
        device: str = 'cuda'
        train_per_steps: int = 6400
        train_epoch_num: int = 6
        batch_size: int = 128
        lr: float = 3e-4
        gamma: float = 0.99  # gamma通常取[0.95, 0.99]
        lambd: float = 0.95
        eps_clip: float = 0.2
        c1: float = 4.0
        c2: float = 0.02
        action_regu_fn: Callable[[Tensor], Tensor] = None
    class TrainStats(NamedTuple):
        a_loss: float
        c_loss: float
        advantage: float
        std: float = 0.0

    def __init__(self, actor: PActor, critic: VCritic, continuous_action: bool, **params):
        self.hp = self.Params(**params)
        self.continuous_action = continuous_action
        self.actor = actor.to(self.hp.device)
        self.critic = critic.to(self.hp.device)
        self.optim_actor = optim.Adam(self.actor.parameters(), lr=self.hp.lr, amsgrad=True)
        self.optim_critic = optim.Adam(self.critic.parameters(), lr=self.hp.lr, amsgrad=True)
        self.collect_steps = 0
    def register(self, env: Env | VectorEnv) -> Optional[Dict]:
        if self.continuous_action:
            if not isinstance(env.action_space, Box):
                raise ValueError("The environment must have continuous action spaces")
            if isinstance(env, VectorEnv):
                self.action_min = torch.tensor(env.action_space.low, device=self.hp.device)[0]
                self.action_max = torch.tensor(env.action_space.high, device=self.hp.device)[0]
            else:
                self.action_min = torch.tensor(env.action_space.low, device=self.hp.device)
                self.action_max = torch.tensor(env.action_space.high, device=self.hp.device)
        else:
            if not isinstance(env.action_space, Discrete):
                raise ValueError("The environment must have discrete action spaces")
        return None
    def get_device(self):
        return self.hp.device
    def select_actions(self, states: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        with torch.no_grad():
            if self.continuous_action:
                mean, std = self.actor(states)
                action = torch.clamp(mean + torch.randn_like(std) * std, -1.0, 1.0)
                return action, {'old_log_prob': self._log_prob_normal(mean, std, action)}
            else:
                probs = self.actor(states)
                action = torch.argmax(probs, dim=-1)
                return action, {'old_log_prob': torch.log(probs[:, action])}
    def update(self, memory: GymMemory, dones: Tensor) -> Optional[TrainStats]:
        self.collect_steps += 1
        if self.collect_steps < self.hp.train_per_steps or not dones.any():
            return None
        else:
            self.collect_steps = 0
        states, action_ctxs = memory.all()
        with torch.no_grad():
            values = self.critic(states.states)
            next_values = self.critic(states.next_states)
        # calc advantage
        advantages = self.calc_traj_advantage(values, states.rewards, next_values, states.terms, states.truncs, memory.unit_size)
        # train batch
        dataset = data.TensorDataset(states.states, states.actions, action_ctxs['old_log_prob'], values, advantages)
        loader = data.DataLoader(dataset, batch_size=self.hp.batch_size, shuffle=True)
        total_loss = actor_loss = critic_loss = torch.tensor(0.0)
        for _ in range(self.hp.train_epoch_num):
            for batch in loader:
                total_loss, (actor_loss, critic_loss, _) = self.total_loss(*batch)
                self.optim_actor.zero_grad()
                self.optim_critic.zero_grad()
                total_loss.backward()
                self.optim_actor.step()
                self.optim_critic.step()
        memory.clear()
        # monitor output
        with torch.no_grad():
            total_loss_val, actor_loss_val, critic_loss_val = total_loss.item(), actor_loss.item(), critic_loss.item()
            avg_advantage = advantages.mean().item()
            std = getattr(self.actor, 'std', None)
            if std is not None:
                std_val = std.mean().item()
            else:
                std_val = 0.0
            return self.TrainStats(actor_loss_val, critic_loss_val, avg_advantage, std_val)
    def calc_traj_advantage(self, values, rewards, next_values, terms, truncs, unit_size):
        # note that next_values should not be ignored for truncated trajectory
        values = values.reshape(-1, unit_size)
        rewards = rewards.reshape(-1, unit_size)
        next_values = next_values.reshape(-1, unit_size)
        terms = terms.reshape(-1, unit_size)
        truncs = truncs.reshape(-1, unit_size)
        deltas = rewards + self.hp.gamma * next_values * ~terms - values
        advantages = torch.empty_like(deltas)
        advantages[-1] = deltas[-1]
        for t in range(len(deltas)-2, -1, -1):
            advantages[t] = deltas[t] + self.hp.gamma * self.hp.lambd * advantages[t+1] * ~(terms[t]|truncs[t])
        return advantages.reshape(-1)
    def total_loss(self, states, actions, old_log_probs, old_values, advantages):
        actor_loss = self.actor_loss(states, actions, old_log_probs, advantages)
        critic_loss = self.hp.c1 * self.critic_loss(states, old_values, advantages)
        entropy_loss = self.hp.c2 * self.entropy_loss()
        total = actor_loss + critic_loss + entropy_loss
        return total, (actor_loss, critic_loss, entropy_loss)
    def actor_loss(self, states, actions, old_log_probs, advantages):
        mean, std = self.actor(states)
        action_regu = self.hp.action_regu_fn(mean) if self.hp.action_regu_fn is not None else 0.0
        new_log_probs = self._log_prob_normal(mean, std, actions)
        ratio = (new_log_probs - old_log_probs).exp()
        return -torch.min(ratio * advantages, torch.clamp(ratio, 1-self.hp.eps_clip, 1+self.hp.eps_clip) * advantages).mean() + action_regu
    def critic_loss(self, states, old_values, advantages):
        new_values = self.critic(states)
        return F.mse_loss(new_values, old_values + advantages)
    def entropy_loss(self):
        std = getattr(self.actor, 'std', None)
        if self.continuous_action and std is not None:
            return -std.log().sum()
        else:
            return 0
    def _log_prob_normal(self, mean, std, sample):
        """ ignore constant term -0.5*ln(2*pi) """
        return -((sample - mean) / std).pow(2).sum(-1) / 2 - std.log().sum(-1)