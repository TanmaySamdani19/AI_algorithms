# Artificial Intelligence Assignment 3

**Author:** Tanmay Samdani  
**Date:** 06/04/2025

## Overview

This project implements two AI strategies for playing two classic games: Tic Tac Toe and Connect Four. The strategies are:

- **Minimax Algorithm** (with and without alpha-beta pruning)
- **Tabular Q-Learning**

Additionally, a default opponent is implemented that selects winning moves if available and blocks the opponent’s winning moves. The goal is to compare the performance of these agents under various scenarios:
1. Each algorithm versus the default opponent.
2. Head-to-head matchups between the algorithms.
3. Overall performance comparison using multiple metrics such as win/draw/loss rates, average move and game times, and node evaluations.

## Directory Structure

```
.
├── agents
│   ├── default_opponent.py      # Default opponent implementation
│   ├── minimax.py               # Minimax algorithm (with/without alpha-beta pruning)
│   └── q_learning.py            # Tabular Q-Learning implementation
├── games
│   ├── tic_tac_toe.py           # Tic Tac Toe game logic
│   └── connect_four.py          # Connect Four game logic
├── main.py                      # Main script to run interactive demos and experiments
├── README.md                    # This file
└── [Other files]                # Experiment log files, generated plot images, etc.
```

## Requirements

- Python 3.x
- NumPy
- Matplotlib

Install the required packages via pip:

```bash
pip install numpy matplotlib
```

## Running the Project

### Interactive Demo

The `main.py` script can be run interactively to play a single game. By default, it runs a Tic Tac Toe game with the Minimax agent versus the default opponent.

To run the interactive demo, execute:

```bash
python main.py
```

This will:
- Initialize the game (Tic Tac Toe by default).
- Set up agents based on specified types (e.g., Minimax vs. Default).
- Print the initial board state.
- Process moves interactively, displaying the board after each move along with move time and (if applicable) node evaluation counts.
- Announce the final result.

### Running Experiments

The `main.py` script also supports running a series of experiments. This mode tests various matchups (e.g., Minimax with alpha-beta pruning vs. Default, Q-Learning vs. Minimax, etc.) and logs detailed performance metrics.

#### To Run Experiments:
1. Ensure the parameters in `main.py` are set as desired. By default:
   - `num_games = 100` (games per matchup).
   - `train_episodes_ttt = 1000` for Tic Tac Toe.
   - `train_episodes_cf = 1000` for Connect Four.
2. Run the script:

```bash
python main.py
```

The script will:
- Execute multiple matchups for both games.
- Log outcomes, move times, and node evaluations in a timestamped log file (e.g., `experiment_log_YYYYMMDD_HHMMSS.log`).
- Save plots (in PNG format) for outcome distributions, win rates, node counts, and timing analyses.

### Testing Specific Agent Configurations

To test a specific agent configuration, modify the call to `play_game()` in `main.py`:

- **Test Minimax Agent vs. Default Opponent in Connect Four:**
  ```python
  play_game('connect_four', 'minimax', 'default', train_episodes=1000)
  ```
- **Test Q-Learning Agent vs. Default Opponent in Tic Tac Toe:**
  ```python
  play_game('tic_tac_toe', 'q_learning', 'default', train_episodes=1000)
  ```
- **Test Head-to-Head Matchup (e.g., Q-Learning vs. Minimax):**
  Modify the `matchups` list in the `run_all_experiments()` function accordingly.

After editing the desired configuration, run:

```bash
python main.py
```

## Hyperparameters and Configuration

### Minimax Agent
- **Tic Tac Toe:**
  - Maximum Depth: 10
  - Alpha-Beta Pruning: Tested with (`True`) and without (`False`)
- **Connect Four:**
  - Maximum Depth: 4
  - Evaluation Strategy: Advanced (heuristic evaluation)
  - Alpha-Beta Pruning: Tested with and without

### Q-Learning Agent (for both games)
- Learning Rate: 0.1
- Discount Factor: 0.9
- Initial Exploration Rate: 0.1 (decays to 0.01 with decay rate 0.995)
- Training Episodes: 1000 (Tic Tac Toe), 1000 (Connect Four)

## Logs and Outputs

- **Log Files:**  
  Logs are automatically generated in the working directory with filenames like `experiment_log_YYYYMMDD_HHMMSS.log`. These logs contain detailed information about each game, including node evaluations, move times, and outcomes.
  
- **Plot Images:**  
  The experiments generate several PNG plots, such as:
  - Outcome distributions (e.g., `tic_tac_toe_results.png`, `connect_four_results.png`)
  - Win rates (e.g., `tic_tac_toe_win_rates.png`, `connect_four_win_rates.png`)
  - Node evaluation counts (e.g., `tic_tac_toe_node_counts.png`, `connect_four_node_counts.png`)
  - Timing analysis (e.g., `tic_tac_toe_times.png`, `connect_four_times.png`)

## Experimentation Overview

The experimental analysis is detailed in the accompanying report and covers:
- Comparisons of each algorithm against the default opponent (Questions 4a and 4b).
- Head-to-head matchups for Tic Tac Toe and Connect Four, including scenarios where agent roles are swapped (Questions 4c and 4d).
- An overall comparison that discusses efficiency, move order effects, and scalability (Question 4e).

## License and Acknowledgements

If you use or adapt code from external sources, please ensure you include the appropriate credits and license information here.

---

This README.md file provides comprehensive instructions for running and testing the project, along with details on the project structure, configuration, and experimental outputs.
```

---

This complete README.md should provide all necessary information for someone to run your main script, test specific agents, and understand the project structure and experimental setup. Feel free to adjust any text or values to perfectly match your project details.