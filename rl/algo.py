from typing import Callable, Dict, Tuple, NamedTuple, Literal, Optional
from itertools import chain
from gymnasium.spaces import Box
import torch
from torch import nn, optim, Tensor
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
