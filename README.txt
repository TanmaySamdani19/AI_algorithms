# AI Game Algorithms Assignment

## Setup Instructions
1. Ensure Python 3.8+ is installed
2. Install required packages:
   ```
   pip install numpy matplotlib
   ```

## Running the Project
- To run main game experiments:
  ```
  python main.py
  ```
- To generate performance analysis graphs:
  ```
  python performance_analysis.py
  ```

## Project Structure
- `games/`: Game implementations (Tic Tac Toe, Connect Four)
- `algorithms/`: Minimax and Q-Learning implementations
- `opponents/`: Default opponent strategy
- `main.py`: Main game runner and experiment conductor
- `performance_analysis.py`: Performance visualization script

## Algorithms Implemented
1. Minimax Algorithm
   - With and without alpha-beta pruning
2. Q-Learning Reinforcement Learning

## Game Variations
- Tic Tac Toe
- Connect Four (with reduced complexity)

## Performance Metrics
- Win/Loss/Draw Ratios
- Learning Curve Analysis