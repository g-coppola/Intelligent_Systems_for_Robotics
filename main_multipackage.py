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
        for r, c in obstacles: 
            g[r, c] = OBSTACLE

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
                
                if cell == CHARGER:
                    recharge = min(20, self.max_battery - self.battery)
                    self.battery += recharge
                
                if cell == DEPOT and all(self.collected):
                    reward += 300
                    self.done = True
                    info["event"] = "MISSIONE COMPIUTA: Rientro alla base!"
                    
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
        return (self.drone_pos[0], self.drone_pos[1], bat_level, 
                int(self.collected[0]), int(self.collected[1]), int(self.collected[2]))

    @property
    def state_space_size(self) -> tuple:
        return (self.grid_size, self.grid_size, 5, 2, 2, 2)

    @property
    def n_actions(self) -> int: return 4

    def render(self, episode="Test", step=0, pause=0.2):
        display_grid = self.grid.copy().astype(float)
        
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
        self._ax.set_xticks([])
        self._ax.set_yticks([])
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

    def choose_action(self, state: tuple) -> int:
        if self.rng.random() < self.epsilon: 
            return self.rng.integers(0, self.n_actions)
        return int(np.argmax(self.q_table[state]))

    def greedy_action(self, state: tuple) -> int:
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state, done):
        current_q = self.q_table[state][action]
        target = reward if done else reward + self.gamma * np.max(self.q_table[next_state])
        
        # Calcoliamo l'errore TD (Temporal Difference)
        td_error = target - current_q
        self.q_table[state][action] = current_q + self.alpha * td_error
        
        return td_error  # Restituiamo l'errore per poterlo plottare

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# ============================================================================
# 4. TRAINING E SIMULAZIONE
# ============================================================================
def train(n_episodes=10000):
    env = DroneEnvMultiReturn()
    agent = QLearningAgent(env.state_space_size, env.n_actions, epsilon_decay=0.9995) 
    
    # Metriche da salvare
    all_rewards = []
    success_rate = []
    q_s0_history = []
    episode_lengths = []    # NUOVO: Passi per episodio
    td_errors_history = []  # NUOVO: Errore TD medio per episodio
    
    # NUOVO: Mappa delle frequenze di visita
    visitation_map = np.zeros((env.grid_size, env.grid_size))
    
    recent_wins = 0
    eval_every = 500

    print("=" * 50)
    print("  Avvio Addestramento Avanzato: 3 Pacchi + Rientro...")
    print("=" * 50)

    s0 = env.reset()

    for ep in range(1, n_episodes + 1):
        state = env.reset()
        total_reward = 0
        won = False
        ep_td_errors = [] # Raccoglie gli errori TD di questo singolo episodio

        for step in range(400):
            # Registriamo la visita in questa cella
            visitation_map[state[0], state[1]] += 1
            
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            
            # Aggiorniamo Q e salviamo l'errore TD (usiamo il valore assoluto per capire la grandezza dell'errore)
            td_error = agent.update(state, action, reward, next_state, done)
            ep_td_errors.append(abs(td_error))
            
            state = next_state
            total_reward += reward

            if done:
                if "MISSIONE COMPIUTA" in info.get("event", ""):
                    won = True
                break

        agent.decay_epsilon()
        
        # Salvataggio Metriche
        all_rewards.append(total_reward)
        episode_lengths.append(step + 1)
        td_errors_history.append(np.mean(ep_td_errors))
        if won: 
            recent_wins += 1

        if ep % eval_every == 0:
            rate = recent_wins / eval_every * 100
            success_rate.append(rate)
            recent_wins = 0
            q_s0_history.append(agent.q_table[s0].copy())
            print(f"  Ep {ep:5d}/{n_episodes} | Successi: {rate:5.1f}% | ε: {agent.epsilon:.3f}")

    print("\nAddestramento completato!")
    return agent, env, all_rewards, success_rate, q_s0_history, eval_every, episode_lengths, td_errors_history, visitation_map

def run_realtime_simulation(agent, env):
    print("\nSimulazione finale in corso...")
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
    plt.ion()
    N_EPISODES = 10000 
    
    agent, env, rewards, success_rate, q_s0_history, eval_every, ep_lengths, td_errors, visitation_map = train(n_episodes=N_EPISODES)
    
    # ====================================================
    # FIGURA 1: Apprendimento di Base (Reward, Successi, Q)
    # ====================================================
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 4.5))
    fig1.canvas.manager.set_window_title("1. Apprendimento Base")
    
    axes1[0].plot(np.convolve(rewards, np.ones(100)/100, mode="valid"), color="#1565C0")
    axes1[0].set_title("Training Reward (Media Mobile)")
    axes1[0].set_xlabel("Episodio")
    axes1[0].set_ylabel("Reward Totale")
    axes1[0].grid(alpha=0.3)
    
    x_eval = [(i+1)*eval_every for i in range(len(success_rate))]
    axes1[1].plot(x_eval, success_rate, color="#388E3C", marker="o", linewidth=2)
    axes1[1].set_title("Tasso di Successo (%)")
    axes1[1].set_xlabel("Episodio")
    axes1[1].set_ylabel("Percentuale Successi (%)")
    axes1[1].set_ylim(-5, 105)
    axes1[1].grid(alpha=0.3)

    q_s0_history = np.array(q_s0_history)
    actions_labels = ["Su", "Giù", "Sinistra", "Destra"]
    colors = ["#E53935", "#1E88E5", "#FFB300", "#43A047"]
    for a in range(4):
        axes1[2].plot(x_eval, q_s0_history[:, a], label=actions_labels[a], color=colors[a], linewidth=2)
    axes1[2].set_title("Evoluzione Q(s0, a) - Stato Iniziale")
    axes1[2].set_xlabel("Episodio")
    axes1[2].set_ylabel("Valore Q(s, a)")
    axes1[2].legend()
    axes1[2].grid(alpha=0.3)
    fig1.tight_layout()

    # ====================================================
    # FIGURA 2: Efficienza ed Errore di Convergenza
    # ====================================================
    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4.5))
    fig2.canvas.manager.set_window_title("2. Efficienza e Convergenza")
    
    axes2[0].plot(np.convolve(ep_lengths, np.ones(100)/100, mode="valid"), color="purple")
    axes2[0].set_title("Lunghezza Episodio (Steps) - Media Mobile")
    axes2[0].set_xlabel("Episodio")
    axes2[0].set_ylabel("Numero di Step (Azioni)")
    axes2[0].grid(alpha=0.3)
    
    axes2[1].plot(np.convolve(td_errors, np.ones(100)/100, mode="valid"), color="#FF8F00")
    axes2[1].set_title("Errore TD Assoluto Medio - Media Mobile")
    axes2[1].set_xlabel("Episodio")
    axes2[1].set_ylabel("Errore TD (Differenza Temporale)")
    axes2[1].grid(alpha=0.3)
    fig2.tight_layout()

    # ====================================================
    # FIGURA 3: Heatmap Valore degli Stati V(s)
    # ====================================================
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    fig3.canvas.manager.set_window_title("3. Mappa del Valore V(s)")
    
    battery_idx = 4
    V_s = np.max(agent.q_table[:, :, battery_idx, 0, 0, 0, :], axis=2).astype(float)
    V_s[env.grid == OBSTACLE] = np.nan  
    
    im3 = ax3.imshow(V_s, cmap="magma")
    plt.colorbar(im3, ax=ax3, label="V(s) = Valore atteso")
    ax3.set_title("Heatmap Valore Stati V(s)\n(Batteria max, 0 pacchi raccolti)")
    ax3.set_xticks(range(env.grid_size))
    ax3.set_yticks(range(env.grid_size))
    ax3.set_xlabel("Colonna della Griglia (X)")
    ax3.set_ylabel("Riga della Griglia (Y)")
    fig3.tight_layout()

    # ====================================================
    # FIGURA 4: Heatmap Frequenze di Visita
    # ====================================================
    fig4, ax4 = plt.subplots(figsize=(6, 5))
    fig4.canvas.manager.set_window_title("4. Esplorazione Spaziale")
    
    visit_map_float = visitation_map.astype(float)
    visit_map_float[env.grid == OBSTACLE] = np.nan
    
    im4 = ax4.imshow(np.log1p(visit_map_float), cmap="viridis")
    plt.colorbar(im4, ax=ax4, label="Log(Visite)")
    ax4.set_title("Heatmap Frequenze di Visita\n(Dove è andato il drone?)")
    ax4.set_xticks(range(env.grid_size))
    ax4.set_yticks(range(env.grid_size))
    ax4.set_xlabel("Colonna della Griglia (X)")
    ax4.set_ylabel("Riga della Griglia (Y)")
    fig4.tight_layout()

    # Mostra tutti i grafici
    plt.show()

    # ====================================================
    # 5. SIMULAZIONE VISIVA FINALE
    # ====================================================
    run_realtime_simulation(agent, env)
    
    print("\nTutte le operazioni concluse. Chiudi le finestre dei grafici per terminare il programma.")
    plt.ioff()
    plt.show()