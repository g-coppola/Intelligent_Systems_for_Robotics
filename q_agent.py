"""
q_agent.py
==========
Agente Q-Learning con politica ε-greedy e aggiornamento
basato sull'equazione di Bellman.

Equazione di aggiornamento (TD(0)):
  Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s',a') − Q(s,a)]

Parametri chiave
----------------
alpha   : learning rate      (quanto aggiorniamo ad ogni step)
gamma   : discount factor    (quanto valutiamo i reward futuri)
epsilon : esplorazione       (prob. di scegliere azione random)
"""

import numpy as np


class QLearningAgent:
    """
    Agente Q-Learning tabulare.

    Parametri
    ----------
    state_space_size : tuple (n_righe, n_colonne, n_livelli_batteria)
    n_actions        : numero di azioni disponibili
    alpha            : learning rate (default 0.1)
    gamma            : fattore di sconto (default 0.99)
    epsilon          : esplorazione iniziale (default 1.0)
    epsilon_min      : esplorazione minima (default 0.05)
    epsilon_decay    : fattore di decay dopo ogni episodio (default 0.995)
    """

    def __init__(
        self,
        state_space_size: tuple,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int = 42,
    ):
        self.n_actions     = n_actions
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng           = np.random.default_rng(seed)

        # Q-table inizializzata a zero
        # Dimensioni: (n_righe, n_colonne, n_livelli_batteria, n_azioni)
        self.q_table = np.zeros((*state_space_size, n_actions))

        # Statistiche per analisi
        self.training_rewards = []
        self.epsilon_history  = []

    # ── Selezione azione ────────────────────────────────────────────────────
    def choose_action(self, state: tuple) -> int:
        """
        Politica ε-greedy:
          - con probabilità ε  → azione casuale (esplorazione)
          - con probabilità 1-ε → azione con Q massimo (sfruttamento)
        """
        if self.rng.random() < self.epsilon:
            return self.rng.integers(0, self.n_actions)  # esplora
        return int(np.argmax(self.q_table[state]))       # sfrutta

    def greedy_action(self, state: tuple) -> int:
        """Restituisce sempre l'azione greedy (per test, senza esplorazione)."""
        return int(np.argmax(self.q_table[state]))

    # ── Aggiornamento Q-table ───────────────────────────────────────────────
    def update(
        self,
        state: tuple,
        action: int,
        reward: float,
        next_state: tuple,
        done: bool,
    ):
        """
        Aggiorna Q(state, action) usando l'equazione di Bellman.

        Se l'episodio è terminato (done=True), non c'è stato futuro:
          target = r
        Altrimenti:
          target = r + γ · max_a' Q(s', a')
        """
        current_q = self.q_table[state][action]

        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])

        # Aggiornamento incrementale
        self.q_table[state][action] = current_q + self.alpha * (target - current_q)

    # ── Decay di ε ──────────────────────────────────────────────────────────
    def decay_epsilon(self):
        """Riduce ε alla fine di ogni episodio (fino a epsilon_min)."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ── Utilità ─────────────────────────────────────────────────────────────
    def get_policy_grid(self, grid_size: int, battery_level: int = 4) -> np.ndarray:
        """
        Restituisce una griglia con la direzione greedy per ogni cella,
        dato un livello di batteria fisso (utile per visualizzazione).
        Valori: 0=Su 1=Giù 2=Sinistra 3=Destra
        """
        policy = np.zeros((grid_size, grid_size), dtype=int)
        for r in range(grid_size):
            for c in range(grid_size):
                state = (r, c, battery_level)
                policy[r, c] = self.greedy_action(state)
        return policy

    def save(self, path: str):
        """Salva la Q-table su file .npy"""
        np.save(path, self.q_table)
        print(f"Q-table salvata in {path}")

    def load(self, path: str):
        """Carica una Q-table da file .npy"""
        self.q_table = np.load(path)
        print(f"Q-table caricata da {path}")
        self.epsilon = self.epsilon_min  # non esplorare dopo il caricamento
