import math
from typing import Union
from games.tic_tac_toe import TicTacToe
from games.connect_four import ConnectFour

class Minimax:
    def __init__(self, game, max_depth=5, use_alpha_beta=True, eval_strategy='simple'):
        """
        Optimized Minimax algorithm with configurable depth and evaluation strategy.
        
        Args:
            game: Game instance (TicTacToe or ConnectFour)
            max_depth (int): Maximum search depth
            use_alpha_beta (bool): Enable alpha-beta pruning
            eval_strategy (str): Evaluation strategy ('simple', 'advanced')
        """
        self.game = game
        self.max_depth = max_depth if isinstance(game, TicTacToe) else 3  # Reduced depth for Connect Four
        self.use_alpha_beta = use_alpha_beta
        self.eval_strategy = eval_strategy

    def minimax(self, depth, is_maximizing, alpha=-math.inf, beta=math.inf):
        # Terminal state or depth limit reached
        winner = self._get_winner()
        if winner is not None or depth == 0:
            return self._evaluate_board(winner, depth)

        if is_maximizing:
            max_eval = -math.inf
            for move in self.game.get_valid_moves():
                move_result = self._simulate_move(move, 1)
                if move_result is None:
                    continue
                
                eval_score = self.minimax(depth-1, False, alpha, beta)
                max_eval = max(max_eval, eval_score)

                # Undo move
                self._undo_move(move, move_result)

                # Alpha-beta pruning
                if self.use_alpha_beta:
                    alpha = max(alpha, eval_score)
                    if beta <= alpha:
                        break

            return max_eval
        else:
            min_eval = math.inf
            for move in self.game.get_valid_moves():
                move_result = self._simulate_move(move, -1)
                if move_result is None:
                    continue
                
                eval_score = self.minimax(depth-1, True, alpha, beta)
                min_eval = min(min_eval, eval_score)

                # Undo move
                self._undo_move(move, move_result)

                # Alpha-beta pruning
                if self.use_alpha_beta:
                    beta = min(beta, eval_score)
                    if beta <= alpha:
                        break

            return min_eval

    def _get_winner(self):
        """Get game winner, handling different game types."""
        if isinstance(self.game, TicTacToe):
            return self.game.check_winner()
        else:  # Connect Four
            return self.game.check_winner(
                self.game.rows-1, 
                max(range(self.game.cols), key=lambda col: col 
                    if self.game.board[self.game.rows-1][col] != 0 else -1)
            )

    def _simulate_move(self, move, player):
        """Simulate a move on the game board."""
        if isinstance(self.game, TicTacToe):
            row, col = move
            if self.game.board[row][col] != 0:
                return None
            self.game.board[row][col] = player
            return (row, col)
        else:  # Connect Four
            col = move
            for row in range(self.game.rows-1, -1, -1):
                if self.game.board[row][col] == 0:
                    self.game.board[row][col] = player
                    return (row, col)
            return None

    def _undo_move(self, move, move_result):
        """Undo a simulated move."""
        if move_result is None:
            return
        
        if isinstance(self.game, TicTacToe):
            row, col = move_result
            self.game.board[row][col] = 0
        else:  # Connect Four
            row, col = move_result
            self.game.board[row][col] = 0

    def _evaluate_board(self, winner, depth):
        """
        Advanced board evaluation considering game state and depth.
        
        Args:
            winner: Game winner
            depth: Current search depth
        
        Returns:
            Evaluation score
        """
        if winner == 1:
            return 100 + depth  # Prefer winning faster
        elif winner == -1:
            return -100 - depth  # Prefer opponent losing faster
        elif winner == 0:
            return 0  # Draw
        
        # For ongoing games, more advanced strategies can be implemented
        if self.eval_strategy == 'advanced':
            return self._advanced_evaluation()
        
        return 0.5  # Neutral ongoing game state

    def _advanced_evaluation(self):
        """
        Advanced board evaluation for ongoing games.
        Potentially implement feature-based scoring.
        """
        # Placeholder for advanced evaluation strategy
        return 0

    def get_best_move(self):
        """Find the best move using Minimax."""
        best_score = -math.inf
        best_move = None
        
        for move in self.game.get_valid_moves():
            move_result = self._simulate_move(move, 1)
            if move_result is None:
                continue
            
            move_score = self.minimax(self.max_depth-1, False)
            
            # Undo move
            self._undo_move(move, move_result)
            
            # Update best move
            if move_score > best_score:
                best_score = move_score
                best_move = move
        
        return best_move