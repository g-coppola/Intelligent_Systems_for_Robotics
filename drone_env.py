"""
drone_env.py
============
Ambiente di simulazione per Drone Delivery su griglia 2D.

Stato  : (riga, colonna, batteria_discreta)
Azioni : 0=Su, 1=Giu, 2=Sinistra, 3=Destra
Reward :
  +100  consegna completata
  -1    ogni step (costo di movimento)
  -50   collisione con ostacolo
  -100  batteria esaurita (episodio terminato)
  +20   ricarica batteria (se presente stazione)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap


# ─── Costanti celle griglia ───────────────────────────────────────────────────
EMPTY    = 0   # cella libera
OBSTACLE = 1   # edificio / zona vietata
DEPOT    = 2   # magazzino (punto di partenza)
TARGET   = 3   # destinazione consegna
CHARGER  = 4   # stazione di ricarica (estensione)


class DroneEnv:
    """
    Ambiente Drone Delivery compatibile con l'interfaccia Gym (reset / step).

    Parametri
    ----------
    grid_size   : dimensione della griglia quadrata (default 8x8)
    max_battery : passi massimi prima che la batteria si esaurisca
    seed        : seme per riproducibilità
    """

    # Mappa direzione → (delta_riga, delta_col)
    MOVES = {
        0: (-1,  0),  # Su
        1: ( 1,  0),  # Giù
        2: ( 0, -1),  # Sinistra
        3: ( 0,  1),  # Destra
    }
    ACTION_NAMES = {0: "Su", 1: "Giù", 2: "Sinistra", 3: "Destra"}

    def __init__(self, grid_size: int = 8, max_battery: int = 50, seed: int = 42):
        self.grid_size   = grid_size
        self.max_battery = max_battery
        self.rng         = np.random.default_rng(seed)

        # Costruisce la mappa fissa una sola volta
        self.grid = self._build_grid()

        # Trova posizioni fisse di depot e target dalla mappa
        self.depot_pos  = tuple(zip(*np.where(self.grid == DEPOT)))[0]
        self.target_pos = tuple(zip(*np.where(self.grid == TARGET)))[0]

        # Stato corrente (inizializzato da reset)
        self.drone_pos = None
        self.battery   = None
        self.done      = False

        # Per il rendering
        self._fig = None
        self._ax  = None

    # ── Costruzione mappa ────────────────────────────────────────────────────
    def _build_grid(self) -> np.ndarray:
        """Crea una griglia con ostacoli fissi, depot e target."""
        g = np.zeros((self.grid_size, self.grid_size), dtype=int)

        # Ostacoli (edifici) – posizioni fisse per riproducibilità
        obstacles = [
            (1, 2), (1, 3),
            (2, 5),
            (3, 1), (3, 2),
            (4, 4), (4, 5),
            (5, 2), (5, 3),
            (6, 5), (6, 6),
        ]
        for r, c in obstacles:
            if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                g[r, c] = OBSTACLE

        # Stazione di ricarica (estensione opzionale)
        g[4, 1] = CHARGER

        # Depot (angolo in alto a sinistra) e Target (angolo in basso a destra)
        g[0, 0] = DEPOT
        g[self.grid_size - 1, self.grid_size - 1] = TARGET

        return g

    # ── Interfaccia Gym-like ─────────────────────────────────────────────────
    def reset(self):
        """Riporta il drone al depot con batteria piena."""
        self.drone_pos = list(self.depot_pos)
        self.battery   = self.max_battery
        self.done      = False
        return self._get_state()

    def step(self, action: int):
        """
        Esegue l'azione e restituisce (next_state, reward, done, info).

        Parametri
        ----------
        action : int in {0,1,2,3}
        """
        assert not self.done, "Episodio terminato, chiama reset()."

        dr, dc = self.MOVES[action]
        new_r  = self.drone_pos[0] + dr
        new_c  = self.drone_pos[1] + dc

        reward = -1          # costo base per ogni step
        info   = {}

        # ── Controllo bordi ──
        if not (0 <= new_r < self.grid_size and 0 <= new_c < self.grid_size):
            reward -= 5      # penalità extra per tentativo fuori griglia
            info["event"] = "fuori_griglia"
        else:
            cell = self.grid[new_r, new_c]

            if cell == OBSTACLE:
                reward -= 50
                info["event"] = "collisione"
                # Il drone rimane fermo
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

        # ── Consumo batteria ──
        self.battery -= 1
        if self.battery <= 0:
            reward    -= 100
            self.done  = True
            info["event"] = info.get("event", "") + " | batteria esaurita"

        return self._get_state(), reward, self.done, info

    # ── Stato ────────────────────────────────────────────────────────────────
    def _get_state(self) -> tuple:
        """
        Stato = (riga, colonna, livello_batteria)
        Il livello batteria è discretizzato in 5 fasce (0-4)
        per mantenere la Q-table di dimensioni ragionevoli.
        """
        bat_level = min(4, self.battery * 5 // (self.max_battery + 1))
        return (self.drone_pos[0], self.drone_pos[1], bat_level)

    @property
    def state_space_size(self) -> tuple:
        return (self.grid_size, self.grid_size, 5)

    @property
    def n_actions(self) -> int:
        return 4

    # ── Rendering ────────────────────────────────────────────────────────────
    def render(self, episode: int = 0, step: int = 0, pause: float = 0.05):
        """Visualizza la griglia corrente con matplotlib."""
        cmap = ListedColormap(["white", "#555555", "#2196F3", "#4CAF50", "#FF9800"])

        display_grid = self.grid.copy().astype(float)
        display_grid[self.drone_pos[0], self.drone_pos[1]] = 5  # drone marker

        cmap = ListedColormap(["white", "#555555", "#2196F3", "#4CAF50", "#FF9800", "#E91E63"])

        if self._fig is None:
            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(6, 6))

        self._ax.clear()
        self._ax.imshow(display_grid, cmap=cmap, vmin=0, vmax=5)

        # Griglia
        for x in range(self.grid_size + 1):
            self._ax.axhline(x - 0.5, color="gray", linewidth=0.5)
            self._ax.axvline(x - 0.5, color="gray", linewidth=0.5)

        # Etichette
        self._ax.set_title(
            f"Episodio {episode} | Step {step} | Batteria {self.battery}/{self.max_battery}",
            fontsize=11
        )
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
