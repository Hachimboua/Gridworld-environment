import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.colors import ListedColormap
from matplotlib import animation

P_MAIN = 0.8
P_SIDE = 0.1


class GridWorld:
    def __init__(self, size=5, goals=[(4, 4)], obstacles=None,
                 max_steps=50, dynamic_elements=False,
                 obstacle_penalty=-5.0, step_penalty=-1.0, goal_reward=1.0):
        self.size = size
        self.goals = goals
        self.obstacle_penalty = obstacle_penalty
        self.step_penalty = step_penalty
        self.goal_reward = goal_reward
        if obstacles is None:
            obstacles = []
        self.obstacles = obstacles
        self.max_steps = max_steps
        self.dynamic_elements = dynamic_elements
        self.state_space = [(r, c) for r in range(size) for c in range(size)]
        self.action_space = 4
        self.reset()

    def reset(self):
        self.agent_pos = [0, 0]
        self.steps = 0
        return self.get_state()

    def get_state(self):
        return tuple(self.agent_pos)

    def _get_next_pos(self, x, y, action):
        if action == 0 and x > 0: x -= 1
        elif action == 1 and x < self.size - 1: x += 1
        elif action == 2 and y > 0: y -= 1
        elif action == 3 and y < self.size - 1: y += 1
        return [x, y]

    def _move_dynamic_elements(self):
        if self.dynamic_elements:
            self.goals = [(
                random.randint(0, self.size - 1),
                random.randint(0, self.size - 1)
            ) for _ in self.goals]
            self.obstacles = [(
                random.randint(0, self.size - 1),
                random.randint(0, self.size - 1)
            ) for _ in self.obstacles]

    def step(self, action):
        x, y = self.agent_pos
        move_choices = [action, 2, 3] if action in [0, 1] else [action, 0, 1]
        effective_action = np.random.choice(move_choices, p=[P_MAIN, P_SIDE, P_SIDE])
        new_pos = self._get_next_pos(x, y, effective_action)

        if tuple(new_pos) in self.obstacles:
            reward = self.obstacle_penalty
        else:
            self.agent_pos = new_pos
            reward = self.step_penalty

        self.steps += 1
        self._move_dynamic_elements()

        if tuple(self.agent_pos) in self.goals:
            reward = self.goal_reward
            done = True
        else:
            done = self.steps >= self.max_steps

        return self.get_state(), reward, done

    def render_frame(self):
        grid = np.zeros((self.size, self.size))
        for ox, oy in self.obstacles:
            grid[ox, oy] = 3
        for gx, gy in self.goals:
            grid[gx, gy] = 2
        frame = np.copy(grid)
        frame[self.agent_pos[0], self.agent_pos[1]] = 1
        return frame


def q_learning_run(env, episodes=500, alpha=0.1, gamma=0.95,
                   eps_start=1.0, eps_end=0.05, eps_decay_episodes=None,
                   gif_name=None):
    if eps_decay_episodes is None:
        eps_decay_episodes = episodes

    state_count = env.size * env.size
    Q = np.zeros((state_count, env.action_space))
    rewards = np.zeros(episodes)
    successes = np.zeros(episodes, dtype=int)
    gif_frames = []

    for ep in range(episodes):
        epsilon = max(eps_end, eps_start - (eps_start - eps_end) * (ep / eps_decay_episodes))
        state = env.reset()
        done, total_reward = False, 0
        episode_frames = []

        while not done:
            s_index = state[0] * env.size + state[1]
            action = random.randint(0, env.action_space - 1) if random.random() < epsilon else int(np.argmax(Q[s_index, :]))
            next_state, reward, done = env.step(action)
            ns_index = next_state[0] * env.size + next_state[1]

            Q[s_index, action] = (1 - alpha) * Q[s_index, action] + alpha * (reward + gamma * np.max(Q[ns_index, :]))
            state = next_state
            total_reward += reward
            episode_frames.append(env.render_frame())

        rewards[ep] = total_reward
        successes[ep] = 1 if tuple(env.agent_pos) in env.goals else 0

        # Save GIF for the first, middle, and last episode
        if gif_name and ep in [0, episodes // 2, episodes - 1]:
            gif_frames.append((ep, episode_frames))

    # Create GIFs
    if gif_name:
        for ep, frames in gif_frames:
            fig, ax = plt.subplots()
            cmap = ListedColormap(["white", "green", "red", "black"])

            def update(i):
                ax.clear()
                ax.imshow(frames[i], cmap=cmap, origin="upper", vmin=0, vmax=3)
                ax.set_title(f"Episode {ep}, Step {i + 1}")
                ax.axis("off")

            ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=200)
            ani.save(f"{gif_name}_ep{ep}.gif", writer="pillow")
            plt.close(fig)

    return Q, rewards, successes


def moving_average(x, w):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode='valid')


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)

    SIZES = list(range(4, 8))
    EPISODES = 500
    ALPHA = 0.1
    GAMMA = 0.95
    EPS_START = 1.0
    EPS_END = 0.05
    MAX_STEPS = 50

    all_rewards, all_successes = {}, {}

    for size in SIZES:
        goal = (size - 1, size - 1)
        rng = random.Random(size)
        obstacles = []
        while len(obstacles) < max(1, size):
            ox, oy = rng.randint(0, size - 1), rng.randint(0, size - 1)
            if (ox, oy) not in [(0, 0), goal, *obstacles]:
                obstacles.append((ox, oy))

        env = GridWorld(size=size, goals=[goal], obstacles=obstacles,
                        max_steps=MAX_STEPS, dynamic_elements=False,
                        obstacle_penalty=-5.0, step_penalty=-1.0, goal_reward=10.0)

        print(f"\n=== Grid {size}x{size} | Goal: {goal} | Obstacles: {obstacles} ===")
        Q, rewards, successes = q_learning_run(env, episodes=EPISODES, alpha=ALPHA, gamma=GAMMA,
                                               eps_start=EPS_START, eps_end=EPS_END,
                                               eps_decay_episodes=EPISODES,
                                               gif_name=f"gridworld_{size}x{size}")
        all_rewards[size] = rewards
        all_successes[size] = successes

    # Plot rewards
    plt.figure(figsize=(12, 6))
    window = 50
    for size in SIZES:
        sm = moving_average(all_rewards[size], window)
        plt.plot(np.arange(len(sm)) + window // 2, sm, label=f"{size}x{size}")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward (smoothed)")
    plt.title("Q-Learning Convergence")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Plot success rates
    plt.figure(figsize=(10, 4))
    final_success_rates = [np.mean(all_successes[s][-50:]) for s in SIZES]
    plt.bar([str(s) for s in SIZES], final_success_rates)
    plt.xlabel("Grid size")
    plt.ylabel("Success rate (last 50 episodes)")
    plt.title("Final Success Rate vs Grid Size")
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    plt.show()
