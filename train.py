"""
train.py
========
Loop di training principale.

Uso
---
  python train.py

Output
------
  - Grafici di convergenza (reward, successi, epsilon)
  - q_table.npy  (Q-table addestrata)
  - training_results.png
"""

import numpy as np
import matplotlib.pyplot as plt
from drone_env import DroneEnv
from q_agent  import QLearningAgent


# ─── Iperparametri ────────────────────────────────────────────────────────────
N_EPISODES   = 2000    # numero totale di episodi di training
MAX_STEPS    = 200     # step massimi per episodio (evita loop infiniti)
GRID_SIZE    = 8       # dimensione griglia
MAX_BATTERY  = 50      # capacità batteria

# Iperparametri Q-Learning
ALPHA         = 0.1    # learning rate
GAMMA         = 0.99   # discount factor
EPSILON       = 1.0    # esplorazione iniziale (100%)
EPSILON_MIN   = 0.05   # esplorazione minima (5%)
EPSILON_DECAY = 0.995  # decay per episodio

# Valuta ogni N episodi
EVAL_EVERY = 100


def train():
    # ── Inizializzazione ────────────────────────────────────────────────────
    env   = DroneEnv(grid_size=GRID_SIZE, max_battery=MAX_BATTERY)
    agent = QLearningAgent(
        state_space_size = env.state_space_size,
        n_actions        = env.n_actions,
        alpha            = ALPHA,
        gamma            = GAMMA,
        epsilon          = EPSILON,
        epsilon_min      = EPSILON_MIN,
        epsilon_decay    = EPSILON_DECAY,
    )

    # ── Statistiche ─────────────────────────────────────────────────────────
    all_rewards  = []
    all_steps    = []
    success_rate = []   # percentuale successi ogni EVAL_EVERY episodi
    recent_wins  = 0

    print("=" * 50)
    print("  Drone Delivery – Training Q-Learning")
    print("=" * 50)
    print(f"  Episodi: {N_EPISODES} | Griglia: {GRID_SIZE}x{GRID_SIZE}")
    print(f"  α={ALPHA} | γ={GAMMA} | ε={EPSILON}→{EPSILON_MIN}")
    print("=" * 50)

    # ── Loop principale ──────────────────────────────────────────────────────
    for ep in range(1, N_EPISODES + 1):
        state       = env.reset()
        total_reward = 0
        won          = False

        for step in range(MAX_STEPS):
            action                     = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)

            agent.update(state, action, reward, next_state, done)

            state        = next_state
            total_reward += reward

            if done:
                if info.get("event") == "consegna":
                    won = True
                break

        agent.decay_epsilon()

        all_rewards.append(total_reward)
        all_steps.append(step + 1)
        if won:
            recent_wins += 1

        # ── Log periodico ───────────────────────────────────────────────────
        if ep % EVAL_EVERY == 0:
            avg_reward = np.mean(all_rewards[-EVAL_EVERY:])
            avg_steps  = np.mean(all_steps[-EVAL_EVERY:])
            rate       = recent_wins / EVAL_EVERY * 100
            success_rate.append(rate)
            recent_wins = 0

            print(
                f"  Ep {ep:5d}/{N_EPISODES} | "
                f"Reward medio: {avg_reward:7.1f} | "
                f"Steps medi: {avg_steps:5.1f} | "
                f"Successi: {rate:5.1f}% | "
                f"ε: {agent.epsilon:.3f}"
            )

    # ── Salvataggio Q-table ─────────────────────────────────────────────────
    agent.save("q_table.npy")

    # ── Grafici ─────────────────────────────────────────────────────────────
    plot_results(all_rewards, success_rate, agent.epsilon_history, N_EPISODES, EVAL_EVERY)

    print("\nTraining completato!")
    return agent, env


def plot_results(rewards, success_rate, epsilon_history, n_episodes, eval_every):
    """Genera e salva i grafici di convergenza."""
    window = 100
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Drone Delivery – Risultati Training Q-Learning", fontsize=13)

    # 1) Reward per episodio + media mobile
    axes[0].plot(rewards, alpha=0.3, color="#90CAF9", label="Reward")
    axes[0].plot(
        range(window - 1, n_episodes),
        smoothed,
        color="#1565C0",
        linewidth=1.8,
        label=f"Media mobile ({window} ep.)"
    )
    axes[0].set_title("Reward per episodio")
    axes[0].set_xlabel("Episodio")
    axes[0].set_ylabel("Reward totale")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # 2) Tasso di successo
    x_eval = [(i + 1) * eval_every for i in range(len(success_rate))]
    axes[1].plot(x_eval, success_rate, marker="o", color="#388E3C", linewidth=1.8, markersize=4)
    axes[1].set_title(f"Tasso di successo (ogni {eval_every} ep.)")
    axes[1].set_xlabel("Episodio")
    axes[1].set_ylabel("Successi (%)")
    axes[1].set_ylim(0, 105)
    axes[1].grid(alpha=0.3)

    # 3) Decadimento ε
    eps_curve = [
        max(0.05, 1.0 * (0.995 ** ep))
        for ep in range(n_episodes)
    ]
    axes[2].plot(eps_curve, color="#E53935", linewidth=1.8)
    axes[2].set_title("Decadimento ε (esplorazione)")
    axes[2].set_xlabel("Episodio")
    axes[2].set_ylabel("ε")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_results.png", dpi=150)
    plt.show()
    print("Grafici salvati in training_results.png")


def test_agent(agent, env, n_test: int = 10, render: bool = False):
    """Testa l'agente senza esplorazione per n_test episodi."""
    wins    = 0
    rewards = []

    print("\n── Test agente addestrato ──")
    for ep in range(1, n_test + 1):
        state        = env.reset()
        total_reward = 0

        for step in range(200):
            action = agent.greedy_action(state)
            state, reward, done, info = env.step(action)
            total_reward += reward

            if render and ep == 1:           # renderizza solo il primo episodio
                env.render(episode=ep, step=step)

            if done:
                if info.get("event") == "consegna":
                    wins += 1
                break

        rewards.append(total_reward)
        print(f"  Ep {ep}: reward={total_reward:.0f} | {info.get('event', '-')}")

    print(f"\nSuccessi: {wins}/{n_test} | Reward medio: {np.mean(rewards):.1f}")
    env.close()


if __name__ == "__main__":
    agent, env = train()
    test_agent(agent, env, n_test=10, render=False)
