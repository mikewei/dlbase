from typing import Literal, NamedTuple, Tuple, Dict, Optional, Callable
from abc import ABC, abstractmethod
from collections import deque, namedtuple
from contextlib import ExitStack
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
    def __init__(self, max_steps: int, unit_size: int, sample_device: Optional[str] = None):
        maxlen = max_steps // unit_size + 1
        self.states_memory: deque[StatesMemory] = deque(maxlen=maxlen)
        self.action_ctxs_memory: deque[Dict[str, Tensor]] = deque(maxlen=maxlen)
        self.unit_size = unit_size
        self.sample_device = sample_device
    def append(self, states_memory: StatesMemory, action_ctxs: Dict[str, Tensor]):
        self.states_memory.append(states_memory)
        self.action_ctxs_memory.append(action_ctxs)
    def size(self):
        return len(self.states_memory) * self.unit_size
    def __len__(self):
        return self.size()
    def __getitem__(self, index):
        return StatesMemory(*(t[index % self.unit_size].to(self.sample_device) for t in self.states_memory[index // self.unit_size])), \
               dict((k, v[index % self.unit_size].to(self.sample_device)) for k, v in self.action_ctxs_memory[index // self.unit_size].items())
    def sample(self, batch_size: int) -> Tuple[StatesMemory, Dict[str, Tensor]]:
        indices = np.random.choice(len(self.states_memory), batch_size // self.unit_size, replace=False)
        states = cat_namedtuples(*(self.states_memory[i] for i in indices))
        action_ctxs = cat_dicts(*(self.action_ctxs_memory[i] for i in indices))
        if self.sample_device is not None:
            states = StatesMemory(*(t.to(self.sample_device) for t in states))
            action_ctxs = dict(*((k, v.to(self.sample_device)) for k, v in action_ctxs.items()))
        return states, action_ctxs
    def clear(self):
        self.states_memory.clear()
        self.action_ctxs_memory.clear()

class GymAlgo(ABC):
    def register(self, env: Env | VectorEnv) -> Optional[Dict]:
        return None
    def get_device(self):
        return 'cpu'
    @abstractmethod
    def select_actions(self, states: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        ...
    def update(self, memory: GymMemory, dones: Tensor) -> Optional[NamedTuple]:
        return None

class RandomAlgo(GymAlgo):
    def __init__(self, show_log=True):
        super().__init__()
        self.show_log = show_log
    def register(self, env: Env | VectorEnv) -> Optional[Dict]:
        self.env = env
        self.is_vec = isinstance(env, VectorEnv)
        self.done_once = torch.zeros(env.num_envs if isinstance(env, VectorEnv) else 1, dtype=torch.bool)
        def stop_on_all_dones(dones: Tensor):
            self.done_once |= dones
            return self.done_once.all()
        return {'stop_on_dones': stop_on_all_dones}
    def select_actions(self, states):
        if self.is_vec:
            actions = torch.tensor(self.env.action_space.sample())
        else:
            actions = torch.tensor(self.env.action_space.sample()).unsqueeze(0)
        if self.show_log:
            print(f'select actions {actions} on states ({states.shape})')
        return actions, {}
    def update(self, memory: GymMemory, dones: Tensor) -> Optional[NamedTuple]:
        if self.show_log:
            print(f'get rewards {memory.states_memory[-1].rewards} and dones {dones}')
        return None

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

def reset_env(env: Env | VectorEnv, device: torch.device | str) -> Tensor:
    state, _ = env.reset()
    if isinstance(env, VectorEnv):
        states = torch.tensor(state, dtype=torch.float32, device=device)
    else:
        states = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    return states

def step_env(env: Env | VectorEnv, actions: Tensor, device: torch.device | str) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    if isinstance(env, VectorEnv):
        next_state, reward, term, trunc, _ = env.step(actions.cpu().numpy())
    else:
        next_state, reward, term, trunc, _ = env.step(actions[0].cpu().numpy())
    next_states = torch.tensor(next_state, device=device)
    rewards = torch.tensor(reward, dtype=torch.float32, device=device)
    terms = torch.tensor(term, dtype=torch.bool, device=device)
    truncs = torch.tensor(trunc, dtype=torch.bool, device=device)
    return next_states, rewards, terms, truncs

def actor_run(env: Env | VectorEnv, actor: Callable[[Tensor], Tensor]):
    env_display = GymDisplay(env)
    ret_display = TableDisplay(1)
    num_envs = env.num_envs if isinstance(env, VectorEnv) else 1
    ReturnLog = namedtuple('ReturnLog', ['steps'] + [f'env_{i}_rewards' for i in range(num_envs)])
    if isinstance(actor, nn.Module) and (p := next(iter(actor.parameters()), None)) is not None:
        device = p.device
    else:
        device = torch.device('cpu')
    done_once = torch.zeros(num_envs, dtype=torch.bool, device=device)
    returns = torch.zeros(num_envs, dtype=torch.float32, device=device)
    states = reset_env(env, device)
    steps = 0
    while not done_once.all():
        with torch.inference_mode():
            actor_ret = actor(states)
            actions = actor_ret[0] if isinstance(actor_ret, Tuple) else actor_ret
        states, rewards, terms, truncs = step_env(env, actions, device)
        returns += rewards * ~done_once
        done_once |= terms | truncs
        steps += 1
        env_display.show()
        ret_display.log(ReturnLog(steps, *(r.item() for r in returns)))

class GymRunner:
    class Params(NamedTuple):
        max_episodes: int = 5000
        table_output_last_n: int = 10
        show_env_freq: int = 0
        memory_size: int = 100000
        stop_on_rewards: Callable[[float], bool] = lambda rewards: False
        stop_on_dones: Callable[[Tensor], bool] = lambda dones: False
        reset_on_dones: Callable[[Tensor], bool] = lambda dones: bool(dones.all().item())
        with_old_memory: Optional[GymMemory] = None
    class BasicStats(NamedTuple):
        t: int
        i_episode: int
        steps: int
        rewards: float
        term: bool
        trunc: bool

    def __init__(self, env: Env | VectorEnv, algo: GymAlgo = RandomAlgo(), **params):
        self.hp = self.Params(**params)
        self.env = env
        self.is_vec = isinstance(env, VectorEnv)
        self.num_envs = env.num_envs if isinstance(env, VectorEnv) else 1
        self.algo = algo
        algo_params = self.algo.register(env)
        if algo_params is not None:
            self.hp = self.hp._replace(**algo_params)
        self.device = self.algo.get_device()
        self.memory_device = 'cpu'
        self.memory = GymMemory(self.hp.memory_size, self.num_envs, self.device) if self.hp.with_old_memory is None else self.hp.with_old_memory
    def run(self, clear_memory: bool = True):
        with ExitStack() as stack:
            if clear_memory:
                self.memory.clear()
            self.do_run(stack)
    def do_run(self, stack: ExitStack):
        if self.env.render_mode == 'rgb_array' and self.hp.show_env_freq > 0:
            gym_dispaly = stack.enter_context(GymDisplay(self.env))
        else:
            gym_dispaly = None
        board = stack.enter_context(ProgressBoard(xlabel='episodes', xlim=(0, self.hp.max_episodes)))
        table = stack.enter_context(TableDisplay(last_n=self.hp.table_output_last_n))
        last_train_stats = None
        # start environment
        state, _ = self.env.reset()
        if self.is_vec:
            states = torch.tensor(state, device=self.memory_device)
        else:
            states = torch.tensor(state, device=self.memory_device).unsqueeze(0)
        total_steps = 0
        loop_steps = 0
        total_episodes_started = self.num_envs
        total_episodes_done = 0
        show_env_counter = 0
        episode_rewards = torch.zeros(self.num_envs, dtype=torch.float32, device=self.memory_device)
        episode_steps = torch.zeros(self.num_envs, dtype=torch.int32, device=self.memory_device)
        while True:
            if gym_dispaly and show_env_counter % self.hp.show_env_freq == 0:
                gym_dispaly.show()
            actions, action_ctxs = self.algo.select_actions(states.to(self.device))
            actions = actions.to(self.memory_device)
            if len(action_ctxs) > 0:
                action_ctxs = dict(*(v.to(self.memory_device) for v in action_ctxs.values()))
            if self.is_vec:
                next_states, rewards, terms, truncs, _ = self.env.step(actions.numpy())
                next_states = torch.tensor(next_states, device=self.memory_device)
                rewards = torch.tensor(rewards, dtype=torch.float32, device=self.memory_device)
                terms = torch.tensor(terms, dtype=torch.bool, device=self.memory_device)
                truncs = torch.tensor(truncs, dtype=torch.bool, device=self.memory_device)
            else:
                next_state, reward , terminated, truncated, _ = self.env.step(actions[0].numpy())
                next_states = torch.tensor(next_state, device=self.memory_device).unsqueeze(0)
                rewards = torch.tensor(reward, dtype=torch.float32, device=self.memory_device).unsqueeze(0)
                terms = torch.tensor(terminated, dtype=torch.bool, device=self.memory_device).unsqueeze(0)
                truncs = torch.tensor(truncated, dtype=torch.bool, device=self.memory_device).unsqueeze(0)
            total_steps += self.num_envs
            loop_steps += 1
            episode_steps += 1
            episode_rewards += rewards
            episode_dones = terms | truncs
            num_episodes_done = int(episode_dones.sum().item())
            if num_episodes_done > 0:  # some episodes done
                for i, j in enumerate(episode_dones.nonzero().squeeze(-1).tolist()):
                    i_episode = total_episodes_done + i
                    n_steps = int(episode_steps[j].item())
                    f_rewards = float(episode_rewards[j].item())
                    b_term = bool(terms[j].item())
                    b_trunc = bool(truncs[j].item())
                    # monitor display
                    board.draw(x=i_episode, y=f_rewards, label='rewards')
                    basic_stats = self.BasicStats(total_steps, i_episode, n_steps, f_rewards, b_term, b_trunc)
                    if last_train_stats is None:
                        table.log(basic_stats)
                    else:
                        table.log(basic_stats, last_train_stats)
                    # check stop condition
                    if self.hp.stop_on_rewards(f_rewards) or (i_episode + 1) >= self.hp.max_episodes:
                        return
                if self.hp.stop_on_dones(episode_dones):
                    return
                total_episodes_started += num_episodes_done
                total_episodes_done += num_episodes_done
                show_env_counter += 1
                episode_steps[episode_dones] = 0
                episode_rewards[episode_dones] = 0
            self.memory.append(StatesMemory(states, actions, rewards, next_states, terms, truncs), action_ctxs)
            train_stats = self.algo.update(self.memory, episode_dones)
            if train_stats is not None:
                last_train_stats = train_stats
            if episode_dones.all():
                state, _ = self.env.reset()
                if self.is_vec:
                    states = torch.tensor(state, device=self.memory_device)
                else:
                    states = torch.tensor(state, device=self.memory_device).unsqueeze(0)
            else:
                states = next_states
