# Drone Delivery con Q-Learning
## Progetto per corso di Intelligenza Artificiale e Robotica

---

## Struttura del progetto

```
drone_delivery_rl/
├── drone_env.py    # Ambiente simulato (griglia 2D)
├── q_agent.py      # Agente Q-Learning
├── train.py        # Loop di training + grafici
├── visualize.py    # Visualizzazione policy e percorso
└── README.md       # Questo file
```

---

## Installazione dipendenze

```bash
pip install numpy matplotlib
```

---

## Come eseguire

### 1. Training
```bash
python train.py
```
Output generato:
- `q_table.npy` — Q-table addestrata
- `training_results.png` — grafici reward, successi, epsilon

### 2. Visualizzazione
```bash
python visualize.py
```
Output generato:
- `policy_map.png` — frecce della policy greedy su griglia
- `episode_path.png` — percorso in un episodio di test
- `q_heatmap.png` — heatmap dei valori Q per cella

---

## Descrizione del problema

### Ambiente
- Griglia **8×8** che rappresenta una città in miniatura
- **Depot** (0,0): punto di partenza del drone
- **Target** (7,7): destinazione di consegna
- **Ostacoli**: edifici che il drone non può attraversare
- **Stazione di ricarica** (4,1): ricarica +20 batteria

### MDP (Markov Decision Process)
| Componente | Descrizione |
|---|---|
| **Stato S** | (riga, colonna, livello_batteria) |
| **Azioni A** | Su, Giù, Sinistra, Destra |
| **Reward R** | +100 consegna / -1 step / -50 ostacolo / -100 no batteria |
| **Policy π** | ε-greedy |

### Algoritmo: Q-Learning (off-policy TD)

```
Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s',a') − Q(s,a)]
```

| Parametro | Valore | Ruolo |
|---|---|---|
| α (alpha) | 0.1 | Velocità di aggiornamento |
| γ (gamma) | 0.99 | Peso reward futuri |
| ε iniziale | 1.0 | Esplorazione 100% |
| ε finale | 0.05 | Esplorazione 5% |
| decay ε | 0.995/ep. | Riduzione graduale |

---

## Connessione al programma del corso

| Modulo | Argomento | Dove nel codice |
|---|---|---|
| Modulo 2 | ML / apprendimento per rinforzo | `q_agent.py` |
| **Modulo 4** | **MDP, Q-Learning, TD-learning** | **tutto il progetto** |
| Modulo 4 | Equazione di Bellman | `q_agent.py → update()` |
| Modulo 4 | Politica ε-greedy | `q_agent.py → choose_action()` |
| Modulo 4 | Convergenza e metriche | `train.py → plot_results()` |

---

## Estensioni possibili (per un voto più alto)

1. **Consegne multiple** — array di target, reward cumulativo
2. **DQN** — sostituire la Q-table con una rete neurale (PyTorch)
3. **Multi-agente** — più droni cooperanti (Modulo 3)
4. **Meteo dinamico** — alcune celle hanno costo variabile
5. **Ambiente stocastico** — il drone può "scivolare" con prob. p

---

## Riferimenti teorici

- Sutton & Barto, *Reinforcement Learning: An Introduction* (cap. 6)
- Mnih et al., *Playing Atari with Deep Reinforcement Learning* (DQN)
