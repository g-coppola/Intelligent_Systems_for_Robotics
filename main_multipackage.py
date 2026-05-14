"""
Drone Delivery Q-Learning - MISSIONE COMPLETA
=============================================
Il drone deve raccogliere 3 pacchi sparsi per la griglia e
RITORNARE AL DEPOT (0,0) prima di esaurire la batteria.
Include grafici avanzati per l'analisi della Q-Function.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ============================================================================
# 1. COSTANTI E PARAMETRI
# ============================================================================
EMPTY    = 0
OBSTACLE = 1
DEPOT    = 2
TARGET   = 3
CHARGER  = 4
DRONE    = 5

ARROWS = {0: "↑", 1: "↓", 2: "←", 3: "→"}

# ============================================================================
# 2. AMBIENTE (Multi-Pacco + Ritorno)
# ============================================================================
class DroneEnvMultiReturn:
    MOVES = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    def __init__(self, grid_size=8, max_battery=100, seed=42):
        self.grid_size   = grid_size
        self.max_battery = max_battery
        self.rng         = np.random.default_rng(seed)

        self.grid = self._build_grid()
        self.depot_pos = tuple(zip(*np.where(self.grid == DEPOT)))[0]
        
        # Le posizioni fisse dei 3 pacchi
        self.target_positions = [(1, 6), (6, 1), (7, 7)]

        self.drone_pos = None
        self.battery   = None
        self.collected = [False, False, False]
        self.done      = False

        self._fig = None
        self._ax  = None

    def _build_grid(self) -> np.ndarray:
        g = np.zeros((self.grid_size, self.grid_size), dtype=int)
        obstacles = [(1, 2), (1, 3), (2, 5), (3, 1), (3, 2), (4, 4), (4, 5), (5, 2), (5, 3), (6, 5), (6, 6)]
        for r, c in obstacles: g[r, c] = OBSTACLE

        g[4, 1] = CHARGER
        g[0, 0] = DEPOT
        return g

    def reset(self):
        self.drone_pos = list(self.depot_pos)
        self.battery   = self.max_battery
        self.collected = [False, False, False]
        self.done      = False
        return self._get_state()

    def step(self, action: int):
        assert not self.done, "Episodio terminato, chiama reset()."

        new_r = self.drone_pos[0] + self.MOVES[action][0]
        new_c = self.drone_pos[1] + self.MOVES[action][1]

        reward = -1
        info   = {}

        if not (0 <= new_r < self.grid_size and 0 <= new_c < self.grid_size):
            reward -= 5
        else:
            cell = self.grid[new_r, new_c]
            if cell == OBSTACLE:
                reward -= 50
            else:
                self.drone_pos = [new_r, new_c]
                
                # 1. Controllo Ricarica
                if cell == CHARGER:
                    recharge = min(20, self.max_battery - self.battery)
                    self.battery += recharge
                
                # 2. Controllo Ritorno alla Base (VITTORIA FINALE)
                if cell == DEPOT and all(self.collected):
                    reward += 300
                    self.done = True
                    info["event"] = "MISSIONE COMPIUTA: Rientro alla base!"
                    
                # 3. Controllo Raccolta Pacchi
                for i, t_pos in enumerate(self.target_positions):
                    if (new_r, new_c) == t_pos and not self.collected[i]:
                        self.collected[i] = True
                        reward += 50
                        info["event"] = f"Pacco {i+1} raccolto!"
                        if all(self.collected):
                            info["event"] += " -> ORA TORNA AL DEPOT!"

        self.battery -= 1
        if self.battery <= 0 and not self.done:
            reward -= 100
            self.done = True
            info["event"] = info.get("event", "") + " Batteria esaurita"

        return self._get_state(), reward, self.done, info

    def _get_state(self) -> tuple:
        bat_level = min(4, self.battery * 5 // (self.max_battery + 1))
        # Lo stato ora include i boolean dei pacchi (0 o 1)
        return (self.drone_pos[0], self.drone_pos[1], bat_level, 
                int(self.collected[0]), int(self.collected[1]), int(self.collected[2]))

    @property
    def state_space_size(self) -> tuple:
        return (self.grid_size, self.grid_size, 5, 2, 2, 2)

    @property
    def n_actions(self) -> int: return 4

    def render(self, episode="Test", step=0, pause=0.2):
        display_grid = self.grid.copy().astype(float)
        
        # Disegna solo i pacchi NON ancora raccolti
        for i, (r, c) in enumerate(self.target_positions):
            if not self.collected[i]:
                display_grid[r, c] = TARGET
                
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

        p_status = sum(self.collected)
        self._ax.set_title(f"Step {step} | Batt {self.battery}/{self.max_battery} | Pacchi {p_status}/3", fontsize=11)
        self._ax.set_xticks([]); self._ax.set_yticks([])
        plt.tight_layout()
        plt.pause(pause)

# ============================================================================
# 3. AGENTE Q-LEARNING
# ============================================================================
class QLearningAgent:
    def __init__(self, state_space_size, n_actions, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.999, seed=42):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay 
        self.rng = np.random.default_rng(seed)
        self.q_table = np.zeros((*state_space_size, n_actions))
        self.epsilon_history = []

    def choose_action(self, state: tuple) -> int:
        if self.rng.random() < self.epsilon: 
            return self.rng.integers(0, self.n_actions)
        return int(np.argmax(self.q_table[state]))

    def greedy_action(self, state: tuple) -> int:
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state, done):
        current_q = self.q_table[state][action]
        target = reward if done else reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state][action] = current_q + self.alpha * (target - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.epsilon_history.append(self.epsilon)

# ============================================================================
# 4. TRAINING E SIMULAZIONE
# ============================================================================
def train(n_episodes=10000):
    env = DroneEnvMultiReturn()
    agent = QLearningAgent(env.state_space_size, env.n_actions, epsilon_decay=0.9995) 
    
    all_rewards, success_rate = [], []
    q_s0_history = []  # Salva i valori Q per lo stato iniziale
    
    recent_wins = 0
    eval_every = 500

    print("=" * 50)
    print("  Avvio Addestramento: 3 Pacchi + Rientro...")
    print(f"  Stati totali: {np.prod(env.state_space_size)}")
    print("=" * 50)

    # Memorizziamo lo stato iniziale esatto
    s0 = env.reset()

    for ep in range(1, n_episodes + 1):
        state = env.reset()
        total_reward = 0
        won = False

        for step in range(400):
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                if "MISSIONE COMPIUTA" in info.get("event", ""):
                    won = True
                break

        agent.decay_epsilon()
        all_rewards.append(total_reward)
        if won: recent_wins += 1

        if ep % eval_every == 0:
            rate = recent_wins / eval_every * 100
            success_rate.append(rate)
            recent_wins = 0
            
            # Salviamo una copia dei 4 valori Q per lo stato iniziale
            q_s0_history.append(agent.q_table[s0].copy())
            
            print(f"  Ep {ep:5d}/{n_episodes} | Successi: {rate:5.1f}% | ε: {agent.epsilon:.3f}")

    print("\nAddestramento completato!")
    return agent, env, all_rewards, success_rate, q_s0_history, eval_every

def run_realtime_simulation(agent, env):
    print("\nSimulazione finale in corso (Guarda la finestra del plot!)...")
    state = env.reset()
    for step in range(150):
        env.render(step=step, pause=0.2)
        action = agent.greedy_action(state)
        state, reward, done, info = env.step(action)
        if "event" in info:
            print(f"  [Step {step}] {info['event']}")
        if done:
            env.render(step=step+1, pause=2.0)
            break

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    plt.ion()  # Modalità interattiva per visualizzare i grafici senza bloccare
    N_EPISODES = 10000 
    
    agent, env, rewards, success_rate, q_s0_history, eval_every = train(n_episodes=N_EPISODES)
    
    # -- Disegno dei 3 grafici fianco a fianco --
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.canvas.manager.set_window_title("Risultati Addestramento Q-Learning")
    
    # 1. Grafico Reward
    axes[0].plot(np.convolve(rewards, np.ones(100)/100, mode="valid"), color="#1565C0")
    axes[0].set_title("Training Reward (Media Mobile 100 ep)")
    axes[0].set_xlabel("Episodi")
    axes[0].grid(alpha=0.3)
    
    # 2. Grafico Tasso di Successo
    x_eval = [(i+1)*eval_every for i in range(len(success_rate))]
    axes[1].plot(x_eval, success_rate, color="#388E3C", marker="o", linewidth=2)
    axes[1].set_title(f"Tasso di Successo (ogni {eval_every} ep)")
    axes[1].set_xlabel("Episodi")
    axes[1].set_ylim(-5, 105)
    axes[1].grid(alpha=0.3)

    # 3. Grafico Q(s0, a)
    q_s0_history = np.array(q_s0_history)
    actions_labels = ["Su (0)", "Giù (1)", "Sinistra (2)", "Destra (3)"]
    colors = ["#E53935", "#1E88E5", "#FFB300", "#43A047"]
    
    for a in range(4):
        axes[2].plot(x_eval, q_s0_history[:, a], label=actions_labels[a], color=colors[a], linewidth=2)
    
    axes[2].set_title("Evoluzione Q(s0, a) - Stato Iniziale")
    axes[2].set_xlabel("Episodi")
    axes[2].set_ylabel("Valore Q")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # -- Avvia la simulazione visuale finale --
    run_realtime_simulation(agent, env)
    
    # Disattiva la modalità interattiva e mantiene le finestre aperte
    print("\nTutte le operazioni concluse. Chiudi le finestre dei grafici per terminare il programma.")
    plt.ioff()
    plt.show()