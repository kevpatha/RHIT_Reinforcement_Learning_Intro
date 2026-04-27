import numpy as np
import matplotlib.pyplot as plt
import os

# Fix for libiomp5md error (multiple copies of OpenMP)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pickle
import random
import itertools
import ctypes
import torch
import torch.nn as nn
import torch.optim as optim

# Import environments
from rosie_bonfire_env import RosieGridEnv
from rosie_continuous_env import RosieContinuousEnv
from rosie_momentum_env import RosieMomentumEnv
from rosie_momentum_random_holes_env import RosieMomentumRandomHolesEnv

# Import models
from dqn_model import DQN, CNNDQN, ReplayBuffer

# --- Sleep Prevention Logic ---
# Windows constants for SetThreadExecutionState
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002 # Keep display on too if desired

def prevent_sleep():
    """Prevents the computer from going to sleep during training."""
    if os.name == 'nt': # Windows
        print("Preventing system sleep...")
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)

def allow_sleep():
    """Allows the computer to go to sleep again."""
    if os.name == 'nt': # Windows
        print("Allowing system sleep...")
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)

# --- Utility Functions ---
if not os.path.exists('saved_results'):
    os.makedirs('saved_results')

def save_result(obj, filename):
    with open(os.path.join('saved_results', filename), 'wb') as f:
        pickle.dump(obj, f)

def load_result(filename):
    path = os.path.join('saved_results', filename)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

# --- 1. Tabular Q-Learning (Grid) ---
def train_q_learning_grid(episodes=5000):
    env = RosieGridEnv(grid_size=10)
    Q_table = np.zeros((env.grid_size, env.grid_size, len(env.action_space)))
    
    max_steps = 500
    gamma = 0.9
    epsilon = 0.1
    alpha = 0.1
    rewards = np.zeros(episodes)
    
    q_snapshots = []
    snapshot_episodes = []
    snapshot_interval = 1000
    
    for e in range(episodes):
        if e % snapshot_interval == 0:
            q_snapshots.append(Q_table.copy())
            snapshot_episodes.append(e)

        ep_reward = 0
        env.reset()
        state = env.agent_pos
        for j in range(max_steps):
            if epsilon < np.random.rand():
                action = Q_table[state[0], state[1]].argmax()
            else:
                action = np.random.randint(0, len(env.action_space))
            
            state_, rew, done = env.step(action)
            ep_reward += rew
        
            Q_table[state[0], state[1]][action] += alpha * (rew + gamma * np.max(Q_table[state_[0], state_[1]]) - Q_table[state[0], state[1]][action])
            state = state_
            if done:
                break
        rewards[e] = ep_reward
    
    q_snapshots.append(Q_table.copy())
    snapshot_episodes.append(episodes)
    return Q_table, rewards, (q_snapshots, snapshot_episodes)

# --- 2. Continuous Discretized Q-Learning ---
def discretize_state(state, n_bins_pos, n_bins_vel, grid_size=10.0, max_vel=5.0):
    x, y, vx, vy = state
    ix = int(np.clip(x / grid_size * n_bins_pos, 0, n_bins_pos - 1))
    iy = int(np.clip(y / grid_size * n_bins_pos, 0, n_bins_pos - 1))
    ivx = int(np.clip((vx + max_vel) / (2 * max_vel) * n_bins_vel, 0, n_bins_vel - 1))
    ivy = int(np.clip((vy + max_vel) / (2 * max_vel) * n_bins_vel, 0, n_bins_vel - 1))
    return (ix, iy, ivx, ivy)

def train_q_learning_continuous(n_bins_pos, n_bins_vel, episodes=20000):
    env = RosieContinuousEnv()
    q_table = np.zeros((n_bins_pos, n_bins_pos, n_bins_vel, n_bins_vel, len(env.action_space)))
    alpha, gamma, epsilon = 0.1, 0.95, 0.1
    max_steps = 500
    rewards_history = []
    
    for e in range(episodes):
        state = env.reset()
        d_state = discretize_state(state, n_bins_pos, n_bins_vel)
        total_reward = 0
        for _ in range(max_steps):
            if np.random.rand() < epsilon:
                action = np.random.randint(0, len(env.action_space))
            else:
                action = np.argmax(q_table[d_state])
            next_state, reward, done = env.step(action)
            d_next_state = discretize_state(next_state, n_bins_pos, n_bins_vel)
            best_next_action = np.argmax(q_table[d_next_state])
            q_table[d_state][action] += alpha * (reward + gamma * q_table[d_next_state][best_next_action] - q_table[d_state][action])
            d_state = d_next_state
            total_reward += reward
            if done: break
        rewards_history.append(total_reward)
        if e % 1000 == 0:
            train_perf = np.convolve(rewards_history, np.ones(100)/100, mode='valid')
            print(f"  Episode {e}: {train_perf[-1]:.2f}")
    return q_table, rewards_history

# --- 3. Linear Function Approximation (Sarsa) ---
def get_features(state, n_tiles, n_tilings, grid_size=10.0, max_vel=5.0):
    x, y, vx, vy = state
    nx, ny = x / grid_size, y / grid_size
    nvx, nvy = (vx + max_vel) / (2 * max_vel), (vy + max_vel) / (2 * max_vel)
    features = []
    for i in range(n_tilings):
        offset = i / (n_tiles * n_tilings)
        ix = int(np.clip((nx + offset) * n_tiles, 0, n_tiles - 1))
        iy = int(np.clip((ny + offset) * n_tiles, 0, n_tiles - 1))
        ivx = int(np.clip((nvx + offset) * n_tiles, 0, n_tiles - 1))
        ivy = int(np.clip((nvy + offset) * n_tiles, 0, n_tiles - 1))
        flat_idx = i * (n_tiles**4) + (ix + iy * n_tiles + ivx * (n_tiles**2) + ivy * (n_tiles**3))
        features.append(flat_idx)
    return np.array(features)

def get_polynomial_features(state, order=4, grid_size=10.0, max_vel=5.0):
    x, y, vx, vy = state
    s = np.array([x/grid_size, y/grid_size, (vx+max_vel)/(2*max_vel), (vy+max_vel)/(2*max_vel)])
    features = [1.0]
    for o in range(1, order + 1):
        for combo in itertools.combinations_with_replacement(range(4), o):
            features.append(np.prod(s[list(combo)]))
    return np.array(features)

def get_fourier_features(state, order=5, grid_size=10.0, max_vel=5.0):
    x, y, vx, vy = state
    s = np.array([x/grid_size, y/grid_size, (vx+max_vel)/(2*max_vel), (vy+max_vel)/(2*max_vel)])
    if not hasattr(get_fourier_features, "coefficients"):
        get_fourier_features.coefficients = np.array([c for c in itertools.product(range(order + 1), repeat=4) if sum(c) <= order + 1])
    return np.cos(np.pi * np.dot(get_fourier_features.coefficients, s))

def train_linear_sarsa(feature_fn, feature_size, is_sparse=True, episodes=20000, lam=0.9, alpha_scale=0.1, epsilon_start=1.0, epsilon_min=0.01):
    env = RosieContinuousEnv()
    n_actions = len(env.action_space)
    w = np.zeros((n_actions, feature_size))
    epsilon = epsilon_start
    epsilon_decay = (epsilon_min / epsilon_start)**(1/episodes)
    rewards_history = []
    for e in range(episodes):
        state = env.reset()
        feat = feature_fn(state)
        alpha = alpha_scale * (0.1 + 0.9 * (1 - e/episodes))
        if is_sparse:
            q_values = [np.sum(w[a, feat]) for a in range(n_actions)]
        else:
            q_values = [np.dot(w[a], feat) for a in range(n_actions)]
        action = np.random.randint(n_actions) if np.random.rand() < epsilon else np.argmax(q_values)
        q_old = q_values[action]
        z = np.zeros((n_actions, feature_size))
        total_reward = 0
        for _ in range(500):
            next_state, reward, done = env.step(action)
            next_feat = feature_fn(next_state)
            total_reward += reward
            if is_sparse:
                next_q_values = [np.sum(w[a, next_feat]) for a in range(n_actions)]
            else:
                next_q_values = [np.dot(w[a], next_feat) for a in range(n_actions)]
            next_action = np.random.randint(n_actions) if np.random.rand() < epsilon else np.argmax(next_q_values)
            q, q_next = q_values[action], next_q_values[next_action]
            delta = reward + 0.95 * (0 if done else q_next) - q
            if is_sparse:
                x_feat = np.zeros(feature_size); x_feat[feat] = 1
                dot_z_x = np.sum(z[action, feat])
            else:
                x_feat = feat; dot_z_x = np.dot(z[action], x_feat)
            z[action] = 0.95 * lam * z[action] + (1 - alpha * 0.95 * lam * dot_z_x) * x_feat
            w += alpha * (delta + q - q_old) * z
            if is_sparse: w[action, feat] -= alpha * (q - q_old)
            else: w[action] -= alpha * (q - q_old) * x_feat
            if done: break
            state, feat, action, q_values, q_old = next_state, next_feat, next_action, next_q_values, q_next
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards_history.append(total_reward)
        if e % 1000 == 0:
            perf = np.convolve(rewards_history, np.ones(100)/100, mode='valid')
            if len(perf) > 0: print(f"  Episode {e}: {perf[-1]:.2f}")
    return w, rewards_history

# --- 6b. Linear Tile Coding SARSA (Random Holes, 7D State) ---
def get_random_holes_tile_features(state, n_tiles=3, n_tilings=8, grid_size=10.0, max_vel=5.0, max_water=25.0, max_hole_dist=10.0):
    x, y, vx, vy, water, dx_hole, dy_hole = state
    s = np.array([
        x / grid_size,
        y / grid_size,
        (vx + max_vel) / (2 * max_vel),
        (vy + max_vel) / (2 * max_vel),
        water / max_water,
        (dx_hole + max_hole_dist) / (2 * max_hole_dist),
        (dy_hole + max_hole_dist) / (2 * max_hole_dist),
    ])
    features = []
    for i in range(n_tilings):
        offset = i / (n_tiles * n_tilings)
        indices = [int(np.clip((s[d] + offset) * n_tiles, 0, n_tiles - 1)) for d in range(7)]
        flat = i * (n_tiles ** 7) + sum(indices[d] * (n_tiles ** d) for d in range(7))
        features.append(flat)
    return np.array(features)

def train_linear_sarsa_random_holes(n_tiles=5, n_tilings=8, episodes=20000, alpha_scale=0.0125):
    env = RosieMomentumRandomHolesEnv()
    n_actions = len(env.action_space)
    feature_size = n_tilings * (n_tiles ** 7)
    w = np.zeros((n_actions, feature_size))
    gamma, lam, epsilon, epsilon_min = 0.95, 0.9, 1.0, 0.05
    epsilon_decay = (epsilon_min / epsilon) ** (1 / episodes)
    rewards_history = []
    for e in range(episodes):
        state = env.reset()
        feat = get_random_holes_tile_features(state, n_tiles, n_tilings)
        alpha = alpha_scale * (0.1 + 0.9 * (1 - e / episodes))
        q_values = [np.sum(w[a, feat]) for a in range(n_actions)]
        action = np.random.randint(n_actions) if np.random.rand() < epsilon else np.argmax(q_values)
        z = np.zeros((n_actions, feature_size))
        q_old, total_reward, done = q_values[action], 0, False
        for _ in range(1000):
            next_state, reward, done = env.step(action)
            next_feat = get_random_holes_tile_features(next_state, n_tiles, n_tilings)
            total_reward += reward
            next_q = [np.sum(w[a, next_feat]) for a in range(n_actions)]
            next_action = np.random.randint(n_actions) if np.random.rand() < epsilon else np.argmax(next_q)
            delta = reward + gamma * (0 if done else next_q[next_action]) - q_values[action]
            x_feat = np.zeros(feature_size); x_feat[feat] = 1
            dot_z_x = np.sum(z[action, feat])
            z[action] = gamma * lam * z[action] + (1 - alpha * gamma * lam * dot_z_x) * x_feat
            w += alpha * (delta + q_values[action] - q_old) * z
            w[action, feat] -= alpha * (q_values[action] - q_old)
            state, feat, action, q_values, q_old = next_state, next_feat, next_action, next_q, next_q[next_action]
            if done: break
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards_history.append(total_reward)
        if e % 500 == 0:
            print(f"  Episode {e}, Avg Reward: {np.mean(rewards_history[-100:]):.2f}")
    return w, rewards_history

# --- 4. Deep Q-Learning (MLP) ---
def train_dqn(episodes=5000, batch_size=64, gamma=0.99, lr=1e-3, target_update=10):
    env = RosieMomentumRandomHolesEnv()
    state_dim, action_dim = 7, len(env.action_space)
    policy_net = DQN(state_dim, action_dim).float()
    target_net = DQN(state_dim, action_dim).float()
    target_net.load_state_dict(policy_net.state_dict()); target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    memory = ReplayBuffer(10000)
    epsilon, epsilon_end, epsilon_decay = 1.0, 0.05, 0.995
    rewards_history = []
    for e in range(episodes):
        state, total_reward, done = env.reset(), 0, False
        while not done:
            if random.random() < epsilon: action = random.randint(0, action_dim - 1)
            else:
                with torch.no_grad(): action = policy_net(torch.FloatTensor(state).unsqueeze(0)).argmax().item()
            next_state, reward, done = env.step(action)
            memory.push(state, action, reward, next_state, done)
            state, total_reward = next_state, total_reward + reward
            if len(memory) > batch_size:
                s, a, r, ns, d = memory.sample(batch_size)
                cur_q = policy_net(torch.FloatTensor(s)).gather(1, torch.LongTensor(a).unsqueeze(1))
                with torch.no_grad():
                    next_q = target_net(torch.FloatTensor(ns)).max(1)[0].unsqueeze(1)
                    target_q = torch.FloatTensor(r).unsqueeze(1) + (1 - torch.FloatTensor(d).unsqueeze(1)) * gamma * next_q
                loss = torch.nn.functional.mse_loss(cur_q, target_q)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        rewards_history.append(total_reward)
        if e % target_update == 0: target_net.load_state_dict(policy_net.state_dict())
        if e % 500 == 0: print(f"  Episode {e}, Avg Reward: {np.mean(rewards_history[-10:]):.2f}")
    return policy_net, rewards_history

# --- 5. Momentum Environment Q-Learning ---
def discretize_momentum_state(state, n_bins_pos=20, n_bins_vel=7, n_bins_water=5, grid_size=10.0, max_vel=5.0, initial_water=25.0):
    x, y, vx, vy, water = state
    ix = int(np.clip(x / grid_size * n_bins_pos, 0, n_bins_pos - 1))
    iy = int(np.clip(y / grid_size * n_bins_pos, 0, n_bins_pos - 1))
    ivx = int(np.clip((vx + max_vel) / (2 * max_vel) * n_bins_vel, 0, n_bins_vel - 1))
    ivy = int(np.clip((vy + max_vel) / (2 * max_vel) * n_bins_vel, 0, n_bins_vel - 1))
    iw = int(np.clip(water / initial_water * n_bins_water, 0, n_bins_water - 1))
    return (ix, iy, ivx, ivy, iw)

def train_momentum_q_learning(episodes=20000):
    env = RosieMomentumEnv()
    q_table = np.zeros((20, 20, 7, 7, 5, len(env.action_space)))
    alpha, gamma, epsilon = 0.1, 0.95, 0.1
    rewards_history = []
    for e in range(episodes):
        state = env.reset()
        d_state = discretize_momentum_state(state)
        total_reward = 0
        for _ in range(500):
            action = np.random.randint(len(env.action_space)) if np.random.rand() < epsilon else np.argmax(q_table[d_state])
            next_state, reward, done = env.step(action)
            d_next_state = discretize_momentum_state(next_state)
            q_table[d_state][action] += alpha * (reward + gamma * np.max(q_table[d_next_state]) - q_table[d_state][action])
            d_state, total_reward = d_next_state, total_reward + reward
            if done: break
        rewards_history.append(total_reward)
        if e % 1000 == 0: print(f"  Episode {e}: Avg Reward: {np.mean(rewards_history[-100:]):.2f}")
    return q_table, rewards_history

# --- 6. Momentum Random Holes (Tabular & CNN) ---
def discretize_random_holes_state(state, n_bins_pos=20, n_bins_vel=7, n_bins_water=5, n_bins_hole=10, grid_size=10.0, max_vel=5.0, max_water=25.0):
    x, y, vx, vy, water, dx_hole, dy_hole = state
    ix, iy = int(np.clip(x/grid_size*n_bins_pos, 0, n_bins_pos-1)), int(np.clip(y/grid_size*n_bins_pos, 0, n_bins_pos-1))
    ivx, ivy = int(np.clip((vx+max_vel)/(2*max_vel)*n_bins_vel, 0, n_bins_vel-1)), int(np.clip((vy+max_vel)/(2*max_vel)*n_bins_vel, 0, n_bins_vel-1))
    iw = int(np.clip(water/max_water*n_bins_water, 0, n_bins_water-1))
    idxh, idyh = int(np.clip((dx_hole+grid_size)/(2*grid_size)*n_bins_hole, 0, n_bins_hole-1)), int(np.clip((dy_hole+grid_size)/(2*grid_size)*n_bins_hole, 0, n_bins_hole-1))
    return (ix, iy, ivx, ivy, iw, idxh, idyh)

def train_momentum_random_holes(episodes=20000):
    env = RosieMomentumRandomHolesEnv()
    Q = np.zeros((20, 20, 7, 7, 5, 10, 10, len(env.action_space)))
    alpha, gamma, epsilon = 0.1, 0.99, 0.1
    history = []
    for e in range(episodes):
        state, done, total_reward = env.reset(), False, 0
        d_state = discretize_random_holes_state(state)
        while not done:
            action = np.random.randint(len(env.action_space)) if np.random.rand() < epsilon else np.argmax(Q[d_state])
            next_state, reward, done = env.step(action)
            d_next_state = discretize_random_holes_state(next_state)
            Q[d_state][action] += alpha * (reward + gamma * np.max(Q[d_next_state]) - Q[d_state][action])
            d_state, total_reward = d_next_state, total_reward + reward
        history.append(total_reward)
        if e % 500 == 0: print(f"  Episode {e}, Avg Reward: {np.mean(history[-500:]):.2f}")
    return Q, history

def train_cnn_dqn(episodes=5000, batch_size=32, gamma=0.99, lr=1e-4, target_update=50, n_stack=4):
    env = RosieMomentumRandomHolesEnv()
    action_dim = len(env.action_space)
    policy_net, target_net = CNNDQN(action_dim, n_channels=n_stack).float(), CNNDQN(action_dim, n_channels=n_stack).float()
    target_net.load_state_dict(policy_net.state_dict()); target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    memory = ReplayBuffer(50000)
    epsilon, epsilon_end, epsilon_decay = 1.0, 0.05, 0.999
    history = []
    from collections import deque
    for e in range(episodes):
        env.reset(); frame, done, total_reward = env.get_image_observation(), False, 0
        stack = deque([frame] * n_stack, maxlen=n_stack)
        state_stack = np.array(stack)
        while not done:
            if random.random() < epsilon: action = random.randint(0, action_dim - 1)
            else:
                policy_net.eval()
                with torch.no_grad(): action = policy_net(torch.FloatTensor(state_stack).unsqueeze(0)).argmax().item()
                policy_net.train()
            _, reward, done = env.step(action); next_frame = env.get_image_observation()
            next_stack = stack.copy()
            next_stack.append(next_frame)
            next_state_stack = np.array(next_stack)
            memory.push(state_stack, action, reward, next_state_stack, done)
            state_stack, stack, total_reward = next_state_stack, next_stack, total_reward + reward
            if len(memory) > batch_size:
                s, a, r, ns, d = memory.sample(batch_size)
                cur_q = policy_net(torch.FloatTensor(s)).gather(1, torch.LongTensor(a).unsqueeze(1))
                with torch.no_grad():
                    next_q = target_net(torch.FloatTensor(ns)).max(1)[0].unsqueeze(1)
                    target_q = torch.FloatTensor(r).unsqueeze(1) + (1 - torch.FloatTensor(d).unsqueeze(1)) * gamma * next_q
                loss = torch.nn.functional.mse_loss(cur_q, target_q)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        history.append(total_reward)
        if e % target_update == 0: target_net.load_state_dict(policy_net.state_dict())
        if e % 50 == 0: print(f"  Episode {e}, Avg Reward: {np.mean(history[-5:]):.2f}")
    return policy_net, history

# --- MAIN EXECUTION ---
def main():
    prevent_sleep()
    try:
        print("--- 1. Tabular Grid Q-Learning ---")
        Q_grid = load_result('q_table_grid.pkl')
        if Q_grid is None:
            Q_grid, rew_grid, snaps = train_q_learning_grid()
            save_result(Q_grid, 'q_table_grid.pkl')
            save_result(rew_grid, 'rewards_grid.pkl')
            save_result(snaps, 'snapshots_grid.pkl')
        else: print("Loaded saved Grid Q-Learning results.")

        print("\n--- 2. Continuous Discretized Q-Learning ---")
        q_tables = load_result('continuous_q_tables.pkl')
        if q_tables is None:
            q_tables, results = {}, {}
            for res in [(5, 3), (10, 5), (20, 7)]:
                print(f"Training resolution {res[0]}x{res[0]}...")
                q, h = train_q_learning_continuous(res[0], res[1])
                q_tables[f"{res[0]}x{res[0]}"], results[f"{res[0]}x{res[0]}"] = q, h
            save_result(q_tables, 'continuous_q_tables.pkl')
            save_result(results, 'continuous_q_results.pkl')
        else: print("Loaded saved Continuous Q-Learning results.")

        print("\n--- 4. Linear Sarsa(lambda) ---")
        linear_res = load_result('linear_sarsa_results.pkl')
        if linear_res is None:
            print("Training Tile Coding...")
            w_tile, h_tile = train_linear_sarsa(lambda s: get_features(s, 8, 8), 8 * (8**4), alpha_scale=0.2/8)
            print("Training Polynomial...")
            w_poly, h_poly = train_linear_sarsa(lambda s: get_polynomial_features(s, 4), len(get_polynomial_features(np.zeros(4), 4)), is_sparse=False, alpha_scale=0.005)
            print("Training Fourier...")
            w_four, h_four = train_linear_sarsa(lambda s: get_fourier_features(s, 5), len(get_fourier_features(np.zeros(4), 5)), is_sparse=False, alpha_scale=0.001)
            linear_res = {'tile': (w_tile, h_tile), 'poly': (w_poly, h_poly), 'fourier': (w_four, h_four)}
            save_result(linear_res, 'linear_sarsa_results.pkl')
        else: print("Loaded saved Linear Sarsa results.")

        print("\n--- 5. Momentum Q-Learning ---")
        if load_result('momentum_q_results.pkl') is None:
            q_m, h_m = train_momentum_q_learning()
            save_result((q_m, h_m), 'momentum_q_results.pkl')
        else: print("Loaded saved Momentum Q-Learning results.")

        print("\n--- 6. Momentum Random Holes ---")
        if load_result('random_holes_q_results.pkl') is None:
            q_rh, h_rh = train_momentum_random_holes()
            save_result((q_rh, h_rh), 'random_holes_q_results.pkl')
        else: print("Loaded saved Tabular Random Holes results.")

        print("\n--- 6b. Linear Tile Coding SARSA (Random Holes) ---")
        linear_rh_result = load_result('linear_sarsa_random_holes_results.pkl')
        if linear_rh_result is None:
            w_rh_linear, h_rh_linear = train_linear_sarsa_random_holes()
            save_result((w_rh_linear, h_rh_linear), 'linear_sarsa_random_holes_results.pkl')
        else:
            print("Loaded saved Linear Sarsa (Random Holes) results.")

        print("\n--- 7. Deep Q-Learning (MLP-DQN) ---")
        if load_result('dqn_random_holes_history.pkl') is None:
            dqn_policy, dqn_history = train_dqn()
            save_result(dqn_history, 'dqn_random_holes_history.pkl')
            torch.save(dqn_policy.state_dict(), os.path.join('saved_results', 'dqn_random_holes_weights.pth'))
        else: print("Loaded saved MLP-DQN results.")

        print("\n--- 8. CNN Deep Q-Learning (Visual) ---")
        if load_result('cnn_dqn_random_holes_history.pkl') is None:
            cnn_policy, cnn_history = train_cnn_dqn()
            save_result(cnn_history, 'cnn_dqn_random_holes_history.pkl')
            torch.save(cnn_policy.state_dict(), os.path.join('saved_results', 'cnn_dqn_random_holes_weights.pth'))
        else: print("Loaded saved CNN-DQN results.")

        print("\nAll training complete!")
    finally:
        allow_sleep()

if __name__ == "__main__":
    main()
