import typer
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import random

import gymnasium as gym

import matplotlib
import matplotlib.pyplot as plt

from collections import namedtuple, deque

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    def __init__(self, capacity: int) -> None:
        self.memory = deque([], maxlen=capacity)

    def push(self, *args) -> None:
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> list:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_observations: int, n_actions: int) -> None:
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class Agent:
    def __init__(
        self,
        n_observations: int,
        n_actions: int,
        alpha: float,
        gamma: float,
        tau: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        buffer_size: int,
        device: torch.device,
    ) -> None:
        self.policy = DQN(n_observations, n_actions).to(device)
        self.target = DQN(n_observations, n_actions).to(device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()

        self.gamma = gamma
        self.tau = tau
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.optimizer = optim.Adam(self.policy.parameters(), lr=alpha)
        self.memory = ReplayMemory(buffer_size)
        self.device = device

    def epsilon_greedy_policy(self, env: gym.Env, state: torch.tensor) -> torch.tensor:
        theta = np.random.uniform(0, 1)
        if theta < self.epsilon:
            action = torch.tensor(
                [[env.action_space.sample()]], device=self.device, dtype=torch.long
            )
        else:
            action = self.greedy_policy(state)
        return action

    def greedy_policy(self, state: torch.tensor) -> torch.tensor:
        with torch.no_grad():
            action = self.policy(state).max(1).indices.view(1, 1)
        return action

    def store_memory(
        self,
        state: torch.tensor,
        action: torch.tensor,
        next_state: torch.tensor,
        reward: torch.tensor,
    ) -> None:
        self.memory.push(state, action, next_state, reward)

    def update(self, batch_size: int = 128) -> None:
        if len(self.memory) < batch_size:
            return
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = (
                self.target(non_final_next_states).max(1).values
            )
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        criterion = nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy.parameters(), 100)
        self.optimizer.step()

        # soft update
        target_state_dict = self.target.state_dict()
        policy_state_dict = self.policy.state_dict()
        for key in policy_state_dict:
            target_state_dict[key] = policy_state_dict[
                key
            ] * self.tau + target_state_dict[key] * (1 - self.tau)
        self.target.load_state_dict(target_state_dict)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)


def train(
    env: gym.Env,
    agent: Agent,
    n_episodes: int,
    device: torch.device,
    n_ma: int = 10,
    seed: int = 0,
) -> tuple[Agent, np.array]:
    agent.policy.train()
    episode_durations = np.zeros((n_episodes,))
    pbar = tqdm(
        range(n_episodes),
        total=n_episodes,
        desc="Starting DQN Training",
        leave=True,
    )
    for episode in pbar:
        state, _ = env.reset(seed=seed + episode)
        env.action_space.seed(seed + episode)
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        done = False
        while not done:
            action = agent.epsilon_greedy_policy(env, state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(
                    observation, dtype=torch.float32, device=device
                ).unsqueeze(0)
            agent.store_memory(state, action, next_state, reward)
            state = next_state
            agent.update()
            episode_durations[episode] += 1
        agent.decay_epsilon()
        if episode > n_ma:
            avg = np.mean(episode_durations[episode - n_ma : episode])
            pbar.set_description(f"Average Episode Duration: {avg:.3f}")
    return agent, episode_durations


def evaluate(
    env: gym.Env, agent: Agent, n_episodes: int, device: torch.device, seed: int = 31415
) -> np.array:
    agent.policy.eval()
    episode_durations = np.zeros((n_episodes,))
    for episode in tqdm(range(n_episodes)):
        state, _ = env.reset(seed=seed + episode)
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        done = False
        while not done:
            action = agent.greedy_policy(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            state = torch.tensor(
                observation, dtype=torch.float32, device=device
            ).unsqueeze(0)
            episode_durations[episode] += 1
    return episode_durations


def show_render(
    env: gym.Env, agent: Agent, n_episodes: int, device: torch.device
) -> None:
    agent.policy.eval()
    for episode in tqdm(range(n_episodes)):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        done = False
        while not done:
            env.render()
            action = agent.greedy_policy(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            state = torch.tensor(
                observation, dtype=torch.float32, device=device
            ).unsqueeze(0)
    env.close()


def main(
    seed: int = 0,
    n_train: int = 1000,
    n_test: int = 100,
    batch_size: int = 128,
    alpha: float = 5e-4,
    gamma: float = 0.99,
    tau: float = 0.005,
    initial_epsilon: float = 0.9,
    epsilon_decay: float = 0.95,
    final_epsilon: float = 0.05,
    buffer_size: int = 10000,
    plot: bool = False,
    render: bool = False,
) -> None:
    # set random seeds
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # select device
    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # set up environment
    env = gym.make("CartPole-v1")

    # get n_actions + n_observations
    n_actions = env.action_space.n
    state, _ = env.reset()
    n_observations = len(state)

    agent = Agent(
        n_observations,
        n_actions,
        alpha,
        gamma,
        tau,
        initial_epsilon,
        epsilon_decay,
        final_epsilon,
        buffer_size,
        device,
    )

    # train
    agent, train_durations = train(env, agent, n_train, device, seed=seed)

    # evaluate
    test_durations = evaluate(env, agent, n_test, device, seed=seed)
    print(f"Average Episode Length = {test_durations.mean():.3f}")

    if render:
        env = gym.make("CartPole-v1", render_mode="human")
        show_render(env, agent, 5, device)


if __name__ == "__main__":
    typer.run(main)
