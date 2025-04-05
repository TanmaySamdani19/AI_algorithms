import numpy as np
from typing import Dict, Tuple, List
from games.tic_tac_toe import TicTacToe
from games.connect_four import ConnectFour

class QLearning:
    """
    Implements tabular Q-learning for Tic Tac Toe and Connect Four.
    """
    def __init__(self, game, player=1, learning_rate=0.1, discount_factor=0.9, 
                 initial_exploration_rate=0.1, min_exploration_rate=0.01, decay_rate=0.995):
        """
        Initialize Q-learning with game instance and parameters.
        
        Args:
            game: Game instance (TicTacToe or ConnectFour)
            player (int): Player number
            learning_rate (float): Learning rate for updates
            discount_factor (float): Discount factor for future rewards
            initial_exploration_rate (float): Initial rate for exploration
            min_exploration_rate (float): Minimum exploration rate
            decay_rate (float): Multiplicative factor to decay exploration rate
        """
        self.game = game
        self.player = player
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.initial_exploration_rate = initial_exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.decay_rate = decay_rate
        self.exploration_rate = initial_exploration_rate
        self.q_table: Dict[Tuple, Dict[int, float]] = {}
        self.state_history = []

    def get_state(self):
        """
        Convert current board to a hashable state.
        Returns:
            tuple: Hashable state representation
        """
        return tuple(map(tuple, self.game.board))

    def get_valid_moves(self):
        """
        Get list of valid moves for the current state.
        Returns:
            List of valid moves
        """
        return self.game.get_valid_moves()

    def get_action(self, state, action):
        """
        Get Q-value for a state-action pair, initialize if not exists.
        Args:
            state: Current game state
            action: Possible action
        
        Returns:
            float: Q-value
        """
        if state not in self.q_table:
            self.q_table[state] = {}
        return self.q_table[state].get(action, 0.0)
    
    def choose_action(self, valid_moves):
        """
        Choose action using epsilon-greedy policy.
        Args:
            valid_moves: List of valid actions (e.g., list of tuples like [(row, col), ...])
            
        Returns:
            tuple: Chosen action (row, col)
        """
        if np.random.random() < self.exploration_rate:
            return valid_moves[np.random.randint(len(valid_moves))]  # Randomly select one of the valid (row, col) tuples
            
        state = self.get_state()
        q_values = [self.get_action(state, move) for move in valid_moves]
        return valid_moves[np.argmax(q_values)]

    def update_q_table(self, state, action, reward, next_state, next_valid_moves):
        """
        Update Q-table using Q-learning update rule.
        Args:
            state: Current state
            action: Action taken
            reward: Immediate reward
            next_state: Next state
            next_valid_moves: Valid moves in next state
        """
        current_q = self.get_action(state, action)
        next_max_q = max([self.get_action(next_state, move) for move in next_valid_moves]
                        if next_valid_moves else [0])
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_max_q - current_q)
        if state not in self.q_table:
            self.q_table[state] = {}
        self.q_table[state][action] = new_q

    def train(self, episodes=1000):
        for episode in range(episodes):
            self.game.reset()
            state = self.get_state()
            done = False

            while not done:
                valid_moves = self.get_valid_moves()
                if not valid_moves:
                    break

                action = self.choose_action(valid_moves)
                if isinstance(self.game, TicTacToe):
                    row, col = action
                    self.game.make_move(row, col)
                else:
                    self.game.make_move(action)

                next_state = self.get_state()
                next_valid_moves = self.get_valid_moves()

                if self.game.game_over:
                    if self.game.winner == self.player:
                        reward = 1
                    elif self.game.winner == -1:
                        reward = -1
                    else:
                        reward = 0
                    done = True
                else:
                    reward = 0

                self.update_q_table(state, action, reward, next_state, next_valid_moves)
                state = next_state

            # Decay exploration rate after each episode
            self.exploration_rate = max(self.min_exploration_rate, 
                                        self.exploration_rate * self.decay_rate)
            
        return

    def play(self):
        """
        Play a game using the learned policy.
        """
        self.game.reset()
        while not self.game.game_over:
            valid_moves = self.get_valid_moves()
            if not valid_moves:
                break

            action = self.choose_action(valid_moves)
            if isinstance(self.game, TicTacToe):
                row, col = action
                self.game.make_move(row, col)
            else:
                self.game.make_move(action)
            self.game.print_board()