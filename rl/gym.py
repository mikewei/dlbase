from typing import Literal, NamedTuple, Tuple, Dict, Optional
from abc import ABC, abstractmethod
from collections import deque
import gymnasium as gym
from gymnasium import Env
from gymnasium.vector import VectorEnv
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, Tensor
from dlbase.utils.nb import TableDisplay, FigureDisplay
from dlbase.utils.tch import cat_namedtuples, cat_dicts
from dlbase.viz.board import ProgressBoard

class StatesMemory(NamedTuple):
    states: Tensor
    actions: Tensor
    rewards: Tensor
    next_states: Tensor
    terms: Tensor
    truncs: Tensor

class GymMemory:
    def __init__(self, maxlen: int, unit_size: int):
        self.states_memory = deque(maxlen=maxlen)
        self.action_ctxs_memory = deque(maxlen=maxlen)
        self.unit_size = unit_size
    def append(self, states_memory: StatesMemory, action_ctxs: Dict[str, Tensor]):
        self.states_memory.append(states_memory)
        self.action_ctxs_memory.append(action_ctxs)
    def size(self):
        return len(self.states_memory) * self.unit_size
    def __len__(self):
        return self.size()
    def __getitem__(self, index):
        return StatesMemory(*(t[index % self.unit_size] for t in self.states_memory[index // self.unit_size])), \
               dict((k, v[index % self.unit_size]) for k, v in self.action_ctxs_memory[index // self.unit_size].items())
    def sample(self, batch_size: int) -> Tuple[StatesMemory, Dict[str, Tensor]]:
        indices = np.random.choice(len(self.states_memory), batch_size // self.unit_size, replace=False)
        return cat_namedtuples(*(self.states_memory[i] for i in indices)), \
               cat_dicts(*(self.action_ctxs_memory[i] for i in indices))
    def clear(self):
        self.states_memory.clear()
        self.action_ctxs_memory.clear()

class GymAlgo(ABC):
    def register_env(self, env: Env | VectorEnv):
        pass
    def get_device(self):
        return 'cpu'
    def need_training(self):
        return True
    @abstractmethod
    def select_actions(self, states: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        ...
    @abstractmethod
    def step_check(self, dones: Tensor) -> Literal['update', 'quit', 'goon']:
        ...
    def update(self, memory: GymMemory) -> Optional[NamedTuple]:
        return None

class RandomAlgo(GymAlgo):
    def register_env(self, env: Env | VectorEnv):
        self.env = env
        self.is_vec = isinstance(env, VectorEnv)
        self.num_envs = env.num_envs if isinstance(env, VectorEnv) else 1
        self.dones = torch.zeros(self.num_envs, dtype=torch.bool)
    def need_training(self):
        return False
    def select_actions(self, states):
        if self.is_vec:
            actions = torch.tensor(self.env.action_space.sample())
        else:
            actions = torch.tensor(self.env.action_space.sample()).unsqueeze(0)
        return actions, {}
    def step_check(self, dones: Tensor) -> Literal['update', 'quit', 'goon']:
        self.dones |= dones
        return 'quit' if self.dones.all() else 'goon'

class GymDisplay(FigureDisplay):
    def __init__(self, env: Env | VectorEnv):
        assert env.render_mode == 'rgb_array'
        super().__init__(auto_close_fig=None, hint='<GymDisplay>')
        self.env = env
        if isinstance(env, Env):
            self.show = self._show_env  # type: ignore
        elif isinstance(env, VectorEnv):
            self.show = self._show_envs  # type: ignore
        else:
            raise ValueError(f'Unsupported env type: {type(env)}')
    def _show_env(self):
        env = self.env
        img = env.render()
        if img is None:
            return
        fig = plt.figure()
        plt.imshow(img)
        plt.axis("off")
        super().show(fig)
        plt.close(fig)
    def _show_envs(self):
        envs = self.env
        imgs = envs.render()
        if imgs is None:
            return
        fig = plt.figure(figsize=(12, 8))
        for i, img in enumerate(imgs):
            fig.add_subplot((len(imgs)+1)//2, 2, i+1)
            plt.imshow(img)
            plt.axis("off")
            plt.title(f'env {i}')
        super().show(fig)
        plt.close(fig)

class GymRunner:
    class Params(NamedTuple):
        max_episodes: int = 5000
        table_output_last_n: int = 10
        show_env_freq: int = 0
        memory_size: int = 100000
    def __init__(self, env: Env | VectorEnv, algo: GymAlgo = RandomAlgo(), **params):
        self.hp = self.Params(**params)
        self.env = env
        self.is_vec = isinstance(env, VectorEnv)
        self.num_envs = env.num_envs if isinstance(env, VectorEnv) else 1
        self.algo = algo
        self.algo.register_env(env)
        self.need_training = self.algo.need_training()
        if self.need_training:
            self.memory = GymMemory(self.hp.memory_size, self.num_envs)
            self.last_train_stats = None
            self.board = ProgressBoard(xlabel='episodes', xlim=(0, self.hp.max_episodes))
            self.table = TableDisplay(last_n=self.hp.table_output_last_n)
        if env.render_mode == 'rgb_array' and self.hp.show_env_freq > 0:
            self.gym_dispaly = GymDisplay(env)
        else:
            self.gym_dispaly = None
    @property
    def device(self):
        return self.algo.get_device() if self.algo else 'cpu'
    def run(self):
        state, _ = self.env.reset()
        if self.is_vec:
            states = torch.tensor(state, dtype=torch.float32, device=self.device)
        else:
            states = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        total_steps = 0
        loop_steps = 0
        total_episodes_started = self.num_envs
        total_episodes_done = 0
        show_env_counter = 0
        episode_rewards = torch.zeros(self.num_envs, dtype=torch.float32)
        episode_steps = torch.zeros(self.num_envs, dtype=torch.int32)
        while True:
            if self.gym_dispaly and show_env_counter % self.hp.show_env_freq == 0:
                self.gym_dispaly.show()
            actions, action_ctxs = self.algo.select_actions(states)
            if self.is_vec:
                next_states, rewards, terms, truncs, _ = self.env.step(actions.cpu().numpy())
                next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
                rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
                terms = torch.tensor(terms, dtype=torch.bool, device=self.device)
                truncs = torch.tensor(truncs, dtype=torch.bool, device=self.device)
            else:
                next_state, reward, terminated, truncated, _ = self.env.step(actions[0].cpu().numpy())
                next_states = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
                rewards = torch.tensor(reward, dtype=torch.float32, device=self.device).unsqueeze(0)
                terms = torch.tensor(terminated, dtype=torch.bool, device=self.device).unsqueeze(0)
                truncs = torch.tensor(truncated, dtype=torch.bool, device=self.device).unsqueeze(0)
            total_steps += self.num_envs
            loop_steps += 1
            episode_steps += 1
            episode_rewards += rewards
            episode_dones = terms | truncs
            num_episodes_done = episode_dones.sum()
            if num_episodes_done > 0:  # some episodes done
                total_episodes_started += num_episodes_done
                total_episodes_done += num_episodes_done
                show_env_counter += 1
            check_result = self.algo.step_check(episode_dones)
            if check_result == 'quit':
                break
            if self.need_training:
                self.memory.append(StatesMemory(states, actions, rewards, next_states, terms, truncs), action_ctxs)
                if check_result == 'update':
                    train_stats = self.algo.update(self.memory)
                    if train_stats is not None:
                        self.last_train_stats = train_stats
            if episode_dones.all():
                state, _ = self.env.reset()
                if self.is_vec:
                    states = torch.tensor(state, dtype=torch.float32, device=self.device)
                else:
                    states = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            else:
                states = next_states
