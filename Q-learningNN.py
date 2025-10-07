import numpy as np
import matplotlib.pyplot as plt
import random
import time
from matplotlib.colors import ListedColormap
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import animation


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
# Neural Q-Network
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
# Q-Learning with Neural Network + GIFs
# ==========================================================
def q_learning_nn(env, episodes=1000, alpha=1e-3, gamma=0.95,
                  eps_start=1.0, eps_end=0.05, eps_decay_episodes=None,
                  render_episodes=None, gif_prefix=None, device='cpu'):

    state_dim = env.size * env.size
    action_dim = env.action_space

    qnet = QNetwork(state_dim, hidden_dim=128, action_dim=action_dim).to(device)
    optimizer = optim.Adam(qnet.parameters(), lr=alpha)
    loss_fn = nn.MSELoss()

    if eps_decay_episodes is None:
        eps_decay_episodes = episodes
    if render_episodes is None:
        render_episodes = []

    rewards = np.zeros(episodes)
    losses = np.zeros(episodes)
    successes = np.zeros(episodes, dtype=int)
    gif_frames = {}  # episode -> list of frames

    for ep in range(episodes):
        epsilon = max(eps_end, eps_start - (eps_start - eps_end) * (ep / eps_decay_episodes))
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
            with torch.no_grad():
                q_vals = qnet(s_tensor)
            action = random.randint(0, action_dim - 1) if random.random() < epsilon else int(torch.argmax(q_vals).item())

            next_state, reward, done = env.step(action)
            total_reward += reward

            next_s_tensor = state_to_onehot(next_state, env).to(device)
            with torch.no_grad():
                q_next = qnet(next_s_tensor)
                max_q_next = torch.max(q_next).item() if not done else 0.0
                target_val = reward + gamma * max_q_next

            q_pred = qnet(s_tensor)[0, action]
            target_tensor = torch.tensor(target_val, dtype=torch.float32, device=device)
            loss = loss_fn(q_pred, target_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ep_loss += loss.item()
            step_count += 1
            state = next_state

            if render:
                frames.append(env.render_frame())

        rewards[ep] = total_reward
        losses[ep] = ep_loss / max(1, step_count)
        successes[ep] = 1 if tuple(env.agent_pos) in env.goals else 0

        if render and gif_prefix:
            gif_frames[ep] = frames

    # 🎞️ Create GIFs for selected episodes
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

    return qnet, rewards, losses, successes


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

    qnet, rewards, losses, successes = q_learning_nn(
        env, episodes=EPISODES, alpha=1e-3, gamma=0.95,
        eps_start=1.0, eps_end=0.05, eps_decay_episodes=EPISODES,
        render_episodes=RENDER_EPS, gif_prefix="nn_gridworld", device='cpu'
    )

    plt.figure(figsize=(9, 4))
    plt.subplot(1, 2, 1)
    plt.plot(rewards, color='teal', alpha=0.8)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('NN Q-Learning: Reward per Episode')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(losses, color='purple', alpha=0.9)
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('NN Q-Learning: Training Loss')
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
