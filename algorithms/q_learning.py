import numpy as np
import random
from games.tic_tac_toe import TicTacToe
from games.connect_four import ConnectFour

class QLearning:
    def __init__(self, game, learning_params=None):
        """
        Optimized Q-Learning with adaptive learning parameters.
        
        Args:
            game: Game instance
            learning_params (dict): Configurable learning parameters
        """
        self.game = game
        
        # Default learning parameters with game-specific adjustments
        self.params = {
            'learning_rate': 0.1,
            'discount_factor': 0.9,
            'exploration_rate': 1.0,
            'min_exploration_rate': 0.01,
            'exploration_decay': 0.995 if isinstance(game, TicTacToe) else 0.99,
            'num_episodes': 5000 if isinstance(game, TicTacToe) else 2000
        }
        
        # Override with custom parameters if provided
        if learning_params:
            self.params.update(learning_params)
        
        self.q_table = {}
        self.reset_learning()

    def reset_learning(self):
        """Reset Q-table and learning parameters."""
        self.q_table = {}
        self.exploration_rate = self.params['exploration_rate']

    def get_state(self, board=None):
        """
        Convert game state to a hashable representation.
        
        Args:
            board (np.ndarray, optional): Custom board state
        
        Returns:
            str: Hashable state representation
        """
        board = board if board is not None else self.game.board
        return str(board.flatten())

    def choose_action(self, valid_moves):
        """
        Action selection with exploration-exploitation trade-off.
        
        Args:
            valid_moves (list): List of possible moves
        
        Returns:
            Selected move
        """
        # Exploration
        if random.random() < self.exploration_rate:
            return random.choice(valid_moves)
        
        # Exploitation
        state = self.get_state()
        state_q_values = self.q_table.get(state, {})
        
        # Find best move among valid moves
        best_moves = [
            move for move in valid_moves 
            if state_q_values.get(move, 0) == max(
                (state_q_values.get(m, 0) for m in valid_moves), 
                default=0
            )
        ]
        
        return random.choice(best_moves)

    def calculate_reward(self, winner):
        """
        Sophisticated reward calculation.
        
        Args:
            winner: Game winner
        
        Returns:
            float: Reward value
        """
        if winner == 1:
            return 10  # Winning reward
        elif winner == -1:
            return -10  # Losing penalty
        elif winner == 0:
            return 0    # Draw
        return 0.1      # Small reward for game progression

    def train(self, verbose=False):
        """
        Training with episode tracking and adaptive learning.
        
        Args:
            verbose (bool): Print training progress
        """
        total_episodes = self.params['num_episodes']
        
        for episode in range(total_episodes):
            self.game.reset()
            current_state = self.get_state()
            
            while not self.game.game_over:
                valid_moves = self.game.get_valid_moves()
                action = self.choose_action(valid_moves)
                
                # Simulate move
                if isinstance(self.game, TicTacToe):
                    self.game.make_move(action[0], action[1])
                    winner = self.game.check_winner()
                else:  # Connect Four
                    # For Connect Four, find row and column of the move
                    for row in range(self.game.rows-1, -1, -1):
                        if self.game.board[row][action] == 0:
                            self.game.make_move(action)
                            winner = self.game.check_winner(row, action)
                            break
                
                # Check game state
                next_state = self.get_state()
                reward = self.calculate_reward(winner)
                
                # Q-table update
                self._update_q_table(current_state, action, reward, next_state)
                
                current_state = next_state
                
                # Decay exploration rate
                self.exploration_rate = max(
                    self.params['min_exploration_rate'], 
                    self.exploration_rate * self.params['exploration_decay']
                )
            
            # Optional verbose tracking
            if verbose and (episode + 1) % 500 == 0:
                print(f"Episode {episode + 1}/{total_episodes}")

    def _update_q_table(self, state, action, reward, next_state):
        """
        Q-table update with SARSA-like learning.
        
        Args:
            state (str): Current game state
            action: Chosen action
            reward (float): Reward received
            next_state (str): Next game state
        """
        # Initialize state and action in Q-table if not exists
        if state not in self.q_table:
            self.q_table[state] = {}
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0
        
        # Q-learning update
        current_q = self.q_table[state][action]
        max_next_q = max(
            self.q_table.get(next_state, {}).values(), 
            default=0
        )
        
        new_q = current_q + self.params['learning_rate'] * (
            reward + self.params['discount_factor'] * max_next_q - current_q
        )
        
        self.q_table[state][action] = new_q

    def get_best_move(self):
        """
        Retrieve best move based on current Q-table.
        
        Returns:
            Best move according to Q-values
        """
        valid_moves = self.game.get_valid_moves()
        state = self.get_state()
        
        if state not in self.q_table:
            return random.choice(valid_moves)
        
        state_q_values = self.q_table[state]
        
        # Find moves with maximum Q-value
        max_q_value = max(state_q_values.get(move, 0) for move in valid_moves)
        best_moves = [
            move for move in valid_moves 
            if state_q_values.get(move, 0) == max_q_value
        ]
        
        return random.choice(best_moves)