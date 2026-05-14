"""
visualize.py
============
Visualizzazione della policy appresa e del percorso del drone.

Uso
---
  python visualize.py           # richiede q_table.npy già salvata
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from drone_env import DroneEnv, OBSTACLE, DEPOT, TARGET, CHARGER


# Frecce per le 4 azioni
ARROWS = {0: "↑", 1: "↓", 2: "←", 3: "→"}


def plot_policy(agent_or_qtable, grid_size: int = 8, battery_level: int = 4):
    """
    Disegna la policy greedy sulla griglia:
    ogni cella mostra la freccia dell'azione migliore.
    """
    if isinstance(agent_or_qtable, np.ndarray):
        q = agent_or_qtable
    else:
        q = agent_or_qtable.q_table

    env = DroneEnv(grid_size=grid_size)

    fig, ax = plt.subplots(figsize=(7, 7))

    cmap = ListedColormap(["#FAFAFA", "#424242", "#2196F3", "#4CAF50", "#FF9800"])
    ax.imshow(env.grid, cmap=cmap, vmin=0, vmax=4)

    for r in range(grid_size):
        for c in range(grid_size):
            cell = env.grid[r, c]
            if cell == OBSTACLE:
                continue

            state  = (r, c, battery_level)
            action = int(np.argmax(q[state]))
            arrow  = ARROWS[action]

            color = "white" if cell in (DEPOT, TARGET, CHARGER) else "#212121"
            ax.text(c, r, arrow, ha="center", va="center", fontsize=14,
                    color=color, fontweight="bold")

    # Griglia
    for x in range(grid_size + 1):
        ax.axhline(x - 0.5, color="gray", linewidth=0.5)
        ax.axvline(x - 0.5, color="gray", linewidth=0.5)

    ax.set_title(f"Policy appresa (livello batteria={battery_level})", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])

    patches = [
        mpatches.Patch(color="#424242", label="Ostacolo"),
        mpatches.Patch(color="#2196F3", label="Depot"),
        mpatches.Patch(color="#4CAF50", label="Target"),
        mpatches.Patch(color="#FF9800", label="Ricarica"),
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig("policy_map.png", dpi=150)
    plt.show()
    print("Policy salvata in policy_map.png")


def plot_episode_path(agent_or_qtable, grid_size: int = 8):
    """
    Simula un episodio greedy e disegna il percorso del drone.
    """
    if isinstance(agent_or_qtable, np.ndarray):
        q = agent_or_qtable
        greedy = lambda s: int(np.argmax(q[s]))
    else:
        greedy = agent_or_qtable.greedy_action

    env   = DroneEnv(grid_size=grid_size)
    state = env.reset()
    path  = [tuple(env.drone_pos)]

    for _ in range(200):
        action = greedy(state)
        state, _, done, info = env.step(action)
        path.append(tuple(env.drone_pos))
        if done:
            break

    # Disegno
    fig, ax = plt.subplots(figsize=(7, 7))
    cmap    = ListedColormap(["#FAFAFA", "#424242", "#2196F3", "#4CAF50", "#FF9800"])
    ax.imshow(env.grid, cmap=cmap, vmin=0, vmax=4)

    # Percorso
    rows = [p[0] for p in path]
    cols = [p[1] for p in path]
    ax.plot(cols, rows, color="#E91E63", linewidth=2, zorder=5)
    ax.scatter(cols[0],  rows[0],  color="#2196F3", s=120, zorder=6, label="Start")
    ax.scatter(cols[-1], rows[-1], color="#E91E63",  s=120, zorder=6, marker="*",
               label=f"Fine ({info.get('event', '?')})")

    # Numera i punti di waypoint
    for i, (r, c) in enumerate(path):
        if i % 3 == 0:
            ax.text(c, r, str(i), ha="center", va="center",
                    fontsize=7, color="#333")

    for x in range(grid_size + 1):
        ax.axhline(x - 0.5, color="gray", linewidth=0.5)
        ax.axvline(x - 0.5, color="gray", linewidth=0.5)

    ax.set_title(f"Percorso episodio greedy ({len(path)-1} step)", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("episode_path.png", dpi=150)
    plt.show()
    print("Percorso salvato in episode_path.png")


def plot_q_values_heatmap(agent_or_qtable, grid_size: int = 8, battery_level: int = 4):
    """
    Heatmap del valore massimo di Q per ogni cella
    (= quanto è promettente trovarsi in quella cella).
    """
    if isinstance(agent_or_qtable, np.ndarray):
        q = agent_or_qtable
    else:
        q = agent_or_qtable.q_table

    env    = DroneEnv(grid_size=grid_size)
    values = np.max(q[:, :, battery_level, :], axis=2)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(values, cmap="RdYlGn")
    plt.colorbar(im, ax=ax, label="max Q(s,a)")

    for r in range(grid_size):
        for c in range(grid_size):
            ax.text(c, r, f"{values[r,c]:.0f}",
                    ha="center", va="center", fontsize=7, color="#111")

    ax.set_title(f"Heatmap valori Q massimi (batteria={battery_level})", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig("q_heatmap.png", dpi=150)
    plt.show()
    print("Heatmap salvata in q_heatmap.png")


if __name__ == "__main__":
    # Carica Q-table salvata dal training
    try:
        q_table = np.load("q_table.npy")
        print(f"Q-table caricata: {q_table.shape}")
    except FileNotFoundError:
        print("q_table.npy non trovata. Esegui prima train.py")
        exit(1)

    plot_policy(q_table)
    plot_episode_path(q_table)
    plot_q_values_heatmap(q_table)
