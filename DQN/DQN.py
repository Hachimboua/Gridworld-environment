import numpy as np
import matplotlib.pyplot as plt
import random
import time
from matplotlib.colors import ListedColormap
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import animation
from collections import deque

# ==========================================================
# Environment
# ==========================================================
class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.max_steps = 2 * self.size * self.size
        self.action_space = 4
        self.reset()
        self.obstacles = self._generate_obstacles()
        self.goals = self._generate_goals()

    def _generate_obstacles(self):
        n_obs = max(1, self.size // 2)
        obs = set()
        while len(obs) < n_obs:
            cell = (np.random.randint(0, self.size), np.random.randint(0, self.size))
            if cell != (0, 0) and cell != (self.size - 1, self.size - 1):
                obs.add(cell)
        return obs

    def _generate_goals(self):
        return {(self.size - 1, self.size - 1)}

    def reset(self):
        self.agent_pos = [0, 0]
        self.steps = 0
        return tuple(self.agent_pos)

    def step(self, action):
        r, c = self.agent_pos
        if action == 0 and r > 0: r -= 1
        elif action == 1 and c < self.size - 1: c += 1
        elif action == 2 and r < self.size - 1: r += 1
        elif action == 3 and c > 0: c -= 1
        self.agent_pos = [r, c]
        self.steps += 1

        if (r, c) in self.goals:
            reward, done = 10.0, True
        elif (r, c) in self.obstacles:
            reward, done = -5.0, False
        else:
            reward, done = -1.0, False

        if self.steps >= self.max_steps:
            done = True
        return (r, c), reward, done

    def render_frame(self):
        grid = np.zeros((self.size, self.size))
        for (r, c) in self.obstacles:
            grid[r, c] = 3
        for (r, c) in self.goals:
            grid[r, c] = 2
        ar, ac = self.agent_pos
        grid[ar, ac] = 1
        return grid


# ==========================================================
# DQN Core
# ==========================================================
class QNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=128, action_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )

    def forward(self, x):
        return self.net(x)


def state_to_onehot(state, env):
    d = env.size * env.size
    vec = np.zeros(d, dtype=np.float32)
    idx = state[0] * env.size + state[1]
    vec[idx] = 1.0
    return torch.from_numpy(vec).unsqueeze(0)


# ==========================================================
# Replay Buffer
# ==========================================================
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# ==========================================================
# DQN Algorithm
# ==========================================================
def dqn_train(env, episodes=1000, gamma=0.95, alpha=1e-3,
              batch_size=64, buffer_size=10000,
              eps_start=1.0, eps_end=0.05, eps_decay=500,
              target_update=20, render_episodes=None, gif_prefix=None, device='cpu'):

    state_dim = env.size * env.size
    action_dim = env.action_space

    policy_net = QNetwork(state_dim, hidden_dim=128, action_dim=action_dim).to(device)
    target_net = QNetwork(state_dim, hidden_dim=128, action_dim=action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=alpha)
    loss_fn = nn.MSELoss()
    buffer = ReplayBuffer(capacity=buffer_size)

    if render_episodes is None:
        render_episodes = []

    rewards = np.zeros(episodes)
    losses = np.zeros(episodes)
    successes = np.zeros(episodes, dtype=int)
    gif_frames = {}

    for ep in range(episodes):
        epsilon = eps_end + (eps_start - eps_end) * np.exp(-ep / eps_decay)
        state = env.reset()
        done = False
        total_reward = 0.0
        ep_loss = 0.0
        step_count = 0
        frames = []
        render = ep in render_episodes or ep == episodes - 1
        if render:
            print(f"🎬 Recording episode {ep+1}/{episodes} (ε={epsilon:.3f})")

        while not done:
            s_tensor = state_to_onehot(state, env).to(device)
            if random.random() < epsilon:
                action = random.randint(0, action_dim - 1)
            else:
                with torch.no_grad():
                    q_vals = policy_net(s_tensor)
                    action = int(torch.argmax(q_vals).item())

            next_state, reward, done = env.step(action)
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            step_count += 1

            if render:
                frames.append(env.render_frame())

            # Sample batch and update network
            if len(buffer) >= batch_size:
                states, actions, rewards_b, next_states, dones = buffer.sample(batch_size)

                states_t = torch.cat([state_to_onehot(s, env) for s in states]).to(device)
                next_states_t = torch.cat([state_to_onehot(s, env) for s in next_states]).to(device)
                actions_t = torch.tensor(actions, dtype=torch.long, device=device)
                rewards_t = torch.tensor(rewards_b, dtype=torch.float32, device=device)
                dones_t = torch.tensor(dones, dtype=torch.float32, device=device)

                q_pred = policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    q_next = target_net(next_states_t).max(1)[0]
                    q_target = rewards_t + gamma * q_next * (1 - dones_t)

                loss = loss_fn(q_pred, q_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ep_loss += loss.item()

        rewards[ep] = total_reward
        losses[ep] = ep_loss / max(1, step_count)
        successes[ep] = 1 if tuple(env.agent_pos) in env.goals else 0

        if render and gif_prefix:
            gif_frames[ep] = frames

        if ep % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

    # 🎞️ Save GIFs
    if gif_prefix and len(gif_frames) > 0:
        cmap = ListedColormap(["white", "green", "red", "black"])
        for ep, frames in gif_frames.items():
            fig, ax = plt.subplots()
            def update(i):
                ax.clear()
                ax.imshow(frames[i], cmap=cmap, origin="upper", vmin=0, vmax=3)
                ax.set_title(f"Episode {ep+1}, Step {i+1}", fontsize=10)
                ax.axis("off")
            ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=200)
            gif_name = f"{gif_prefix}_ep{ep+1}.gif"
            ani.save(gif_name, writer="pillow")
            plt.close(fig)
            print(f"Saved {gif_name}")

    return policy_net, rewards, losses, successes


# ==========================================================
# Main
# ==========================================================
if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    env = GridWorld(size=6)
    EPISODES = 600
    RENDER_EPS = [0, EPISODES // 2, EPISODES - 1]

    qnet, rewards, losses, successes = dqn_train(
        env, episodes=EPISODES, alpha=1e-3, gamma=0.95,
        eps_start=1.0, eps_end=0.05, eps_decay=200,
        batch_size=64, target_update=20,
        render_episodes=RENDER_EPS, gif_prefix="dqn_gridworld", device='cpu'
    )

    plt.figure(figsize=(9, 4))
    plt.subplot(1, 2, 1)
    plt.plot(rewards, color='teal', alpha=0.8)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('DQN: Reward per Episode')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(losses, color='purple', alpha=0.9)
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('DQN: Training Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    final_success_rate = np.mean(successes[-100:]) if EPISODES >= 100 else np.mean(successes)
    print(f"Final success rate (last 100 episodes): {final_success_rate:.3f}")

    state_vals = np.zeros((env.size, env.size))
    best_actions = np.zeros((env.size, env.size), dtype=int)
    for r in range(env.size):
        for c in range(env.size):
            s = (r, c)
            s_t = state_to_onehot(s, env)
            with torch.no_grad():
                q = qnet(s_t).numpy()[0]
            state_vals[r, c] = np.max(q)
            best_actions[r, c] = int(np.argmax(q))

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(state_vals, cmap='coolwarm', origin='upper')
    plt.title('State Value (max Q)')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(best_actions, cmap='tab10', origin='upper')
    plt.title('Best Action (0:up,1:right,2:down,3:left)')
    plt.colorbar()
    plt.tight_layout()
    plt.show()
