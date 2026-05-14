"""
Drone Delivery Q-Learning - Progetto Unificato
==============================================
Questo script include l'ambiente, l'agente Q-Learning, il ciclo di training,
la visualizzazione dei grafici e una simulazione in tempo reale finale.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# ============================================================================
# 1. COSTANTI
# ============================================================================
EMPTY    = 0   # cella libera
OBSTACLE = 1   # edificio / zona vietata
DEPOT    = 2   # magazzino (punto di partenza)
TARGET   = 3   # destinazione consegna
CHARGER  = 4   # stazione di ricarica
DRONE    = 5   # marker del drone (usato per il rendering)

ARROWS = {0: "↑", 1: "↓", 2: "←", 3: "→"}

# ============================================================================
# 2. AMBIENTE (DroneEnv)
# ============================================================================
class DroneEnv:
    MOVES = {
        0: (-1,  0),  # Su
        1: ( 1,  0),  # Giù
        2: ( 0, -1),  # Sinistra
        3: ( 0,  1),  # Destra
    }

    def __init__(self, grid_size: int = 8, max_battery: int = 50, seed: int = 42):
        self.grid_size   = grid_size
        self.max_battery = max_battery
        self.rng         = np.random.default_rng(seed)

        self.grid = self._build_grid()
        self.depot_pos  = tuple(zip(*np.where(self.grid == DEPOT)))[0]
        self.target_pos = tuple(zip(*np.where(self.grid == TARGET)))[0]

        self.drone_pos = None
        self.battery   = None
        self.done      = False

        self._fig = None
        self._ax  = None

    def _build_grid(self) -> np.ndarray:
        g = np.zeros((self.grid_size, self.grid_size), dtype=int)
        obstacles = [
            (1, 2), (1, 3), (2, 5), (3, 1), (3, 2),
            (4, 4), (4, 5), (5, 2), (5, 3), (6, 5), (6, 6),
        ]
        for r, c in obstacles:
            if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                g[r, c] = OBSTACLE

        g[4, 1] = CHARGER
        g[0, 0] = DEPOT
        g[self.grid_size - 1, self.grid_size - 1] = TARGET
        return g

    def reset(self):
        self.drone_pos = list(self.depot_pos)
        self.battery   = self.max_battery
        self.done      = False
        return self._get_state()

    def step(self, action: int):
        assert not self.done, "Episodio terminato, chiama reset()."

        dr, dc = self.MOVES[action]
        new_r  = self.drone_pos[0] + dr
        new_c  = self.drone_pos[1] + dc

        reward = -1
        info   = {}

        if not (0 <= new_r < self.grid_size and 0 <= new_c < self.grid_size):
            reward -= 5
            info["event"] = "fuori_griglia"
        else:
            cell = self.grid[new_r, new_c]
            if cell == OBSTACLE:
                reward -= 50
                info["event"] = "collisione"
            else:
                self.drone_pos = [new_r, new_c]
                if cell == TARGET:
                    reward += 100
                    self.done = True
                    info["event"] = "consegna"
                elif cell == CHARGER:
                    recharge       = min(20, self.max_battery - self.battery)
                    self.battery  += recharge
                    info["event"]  = f"ricarica +{recharge}"

        self.battery -= 1
        if self.battery <= 0:
            reward    -= 100
            self.done  = True
            info["event"] = info.get("event", "") + " | batteria esaurita"

        return self._get_state(), reward, self.done, info

    def _get_state(self) -> tuple:
        bat_level = min(4, self.battery * 5 // (self.max_battery + 1))
        return (self.drone_pos[0], self.drone_pos[1], bat_level)

    @property
    def state_space_size(self) -> tuple:
        return (self.grid_size, self.grid_size, 5)

    @property
    def n_actions(self) -> int:
        return 4

    def render(self, episode: int = 0, step: int = 0, pause: float = 0.2):
        display_grid = self.grid.copy().astype(float)
        display_grid[self.drone_pos[0], self.drone_pos[1]] = DRONE

        cmap = ListedColormap(["white", "#555555", "#2196F3", "#4CAF50", "#FF9800", "#E91E63"])

        if self._fig is None:
            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(6, 6))
            self._fig.canvas.manager.set_window_title("Live Simulation")

        self._ax.clear()
        self._ax.imshow(display_grid, cmap=cmap, vmin=0, vmax=5)

        for x in range(self.grid_size + 1):
            self._ax.axhline(x - 0.5, color="gray", linewidth=0.5)
            self._ax.axvline(x - 0.5, color="gray", linewidth=0.5)

        self._ax.set_title(f"Episodio {episode} | Step {step} | Batteria {self.battery}/{self.max_battery}", fontsize=11)
        patches = [
            mpatches.Patch(color="#555555", label="Ostacolo"),
            mpatches.Patch(color="#2196F3", label="Depot"),
            mpatches.Patch(color="#4CAF50", label="Target"),
            mpatches.Patch(color="#FF9800", label="Ricarica"),
            mpatches.Patch(color="#E91E63", label="Drone"),
        ]
        self._ax.legend(handles=patches, loc="upper right", fontsize=8)
        self._ax.set_xticks([])
        self._ax.set_yticks([])

        plt.tight_layout()
        plt.pause(pause)

    def close(self):
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None

# ============================================================================
# 3. AGENTE (QLearningAgent)
# ============================================================================
class QLearningAgent:
    def __init__(self, state_space_size, n_actions, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995, seed=42):
        self.n_actions     = n_actions
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng           = np.random.default_rng(seed)
        self.q_table       = np.zeros((*state_space_size, n_actions))
        self.epsilon_history = []

    def choose_action(self, state: tuple) -> int:
        if self.rng.random() < self.epsilon:
            return self.rng.integers(0, self.n_actions)
        return int(np.argmax(self.q_table[state]))

    def greedy_action(self, state: tuple) -> int:
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state, done):
        current_q = self.q_table[state][action]
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state][action] = current_q + self.alpha * (target - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.epsilon_history.append(self.epsilon)

# ============================================================================
# 4. TRAINING & PLOTTING FUNCTIONS
# ============================================================================
def train(n_episodes=2000, grid_size=8, max_battery=50):
    env   = DroneEnv(grid_size=grid_size, max_battery=max_battery)
    agent = QLearningAgent(env.state_space_size, env.n_actions)
    
    all_rewards  = []
    all_steps    = []
    success_rate = []
    recent_wins  = 0
    eval_every   = 100

    print("=" * 50)
    print("  Avvio Addestramento Q-Learning...")
    print("=" * 50)

    for ep in range(1, n_episodes + 1):
        state = env.reset()
        total_reward = 0
        won = False

        for step in range(200):
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                if info.get("event") == "consegna": won = True
                break

        agent.decay_epsilon()
        all_rewards.append(total_reward)
        all_steps.append(step + 1)
        if won: recent_wins += 1

        if ep % eval_every == 0:
            rate = recent_wins / eval_every * 100
            success_rate.append(rate)
            recent_wins = 0
            print(f"  Ep {ep:5d}/{n_episodes} | Successi: {rate:5.1f}% | ε: {agent.epsilon:.3f}")

    print("\nTraining completato!")
    return agent, env, all_rewards, success_rate, eval_every

def show_training_results(rewards, success_rate, agent, n_episodes, eval_every):
    window = 100
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.canvas.manager.set_window_title("Risultati Training")
    fig.suptitle("Drone Delivery – Risultati Training Q-Learning", fontsize=13)

    axes[0].plot(rewards, alpha=0.3, color="#90CAF9", label="Reward")
    axes[0].plot(range(window - 1, n_episodes), smoothed, color="#1565C0", linewidth=1.8, label="Media mobile")
    axes[0].set_title("Reward per episodio")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    x_eval = [(i + 1) * eval_every for i in range(len(success_rate))]
    axes[1].plot(x_eval, success_rate, marker="o", color="#388E3C", linewidth=1.8)
    axes[1].set_title("Tasso di successo")
    axes[1].set_ylim(0, 105)
    axes[1].grid(alpha=0.3)

    axes[2].plot(agent.epsilon_history, color="#E53935", linewidth=1.8)
    axes[2].set_title("Decadimento esplorazione (ε)")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    fig.show()

def show_policy(agent, grid_size=8, battery_level=4):
    env = DroneEnv(grid_size=grid_size)
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.canvas.manager.set_window_title("Policy Appresa")
    
    cmap = ListedColormap(["#FAFAFA", "#424242", "#2196F3", "#4CAF50", "#FF9800"])
    ax.imshow(env.grid, cmap=cmap, vmin=0, vmax=4)

    for r in range(grid_size):
        for c in range(grid_size):
            if env.grid[r, c] == OBSTACLE: continue
            action = agent.greedy_action((r, c, battery_level))
            color = "white" if env.grid[r, c] in (DEPOT, TARGET, CHARGER) else "#212121"
            ax.text(c, r, ARROWS[action], ha="center", va="center", fontsize=14, color=color, fontweight="bold")

    for x in range(grid_size + 1):
        ax.axhline(x - 0.5, color="gray", linewidth=0.5)
        ax.axvline(x - 0.5, color="gray", linewidth=0.5)

    ax.set_title(f"Policy greedy (livello batteria={battery_level})")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    fig.show()

def show_q_heatmap(agent, grid_size=8, battery_level=4):
    values = np.max(agent.q_table[:, :, battery_level, :], axis=2)
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.canvas.manager.set_window_title("Heatmap Q-Values")
    
    im = ax.imshow(values, cmap="RdYlGn")
    plt.colorbar(im, ax=ax, label="max Q(s,a)")

    for r in range(grid_size):
        for c in range(grid_size):
            ax.text(c, r, f"{values[r,c]:.0f}", ha="center", va="center", fontsize=8, color="#111")

    ax.set_title(f"Valori Q massimi (batteria={battery_level})")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    fig.show()

# ============================================================================
# 5. SIMULAZIONE IN TEMPO REALE
# ============================================================================
def run_realtime_simulation(agent, env):
    print("\n" + "=" * 50)
    print("  Avvio Simulazione in Tempo Reale...")
    print("  Guarda la finestra del plot per vedere il drone.")
    print("=" * 50)

    state = env.reset()
    total_reward = 0

    for step in range(100):
        # Renderizza a schermo l'ambiente (con una pausa per vedere il frame)
        env.render(episode="Test", step=step, pause=0.3)
        
        # Scelta azione fully greedy
        action = agent.greedy_action(state)
        state, reward, done, info = env.step(action)
        total_reward += reward

        if done:
            # Mostra l'ultimo frame
            env.render(episode="Test", step=step+1, pause=1.0)
            print(f"Simulazione terminata: {info.get('event', '-')} | Reward: {total_reward}")
            break

# ============================================================================
# MAIN LOOP
# ============================================================================
if __name__ == "__main__":
    # Abilita la modalità interattiva di matplotlib per aprire più finestre 
    # senza bloccare l'esecuzione fino alla fine.
    plt.ion()

    # 1. Avvia Addestramento
    N_EPISODES = 1500  # Abbassato leggermente per velocizzare il test, ma sufficiente per convergere
    agent, env, rewards, success_rate, eval_every = train(n_episodes=N_EPISODES)

    # 2. Genera tutti i grafici statici a schermo
    show_training_results(rewards, success_rate, agent, N_EPISODES, eval_every)
    show_policy(agent)
    show_q_heatmap(agent)

    # 3. Avvia la simulazione visiva
    run_realtime_simulation(agent, env)

    # 4. Mantieni le finestre aperte al termine
    print("\nTutte le operazioni concluse. Chiudi le finestre dei grafici per uscire dallo script.")
    plt.ioff()
    plt.show()