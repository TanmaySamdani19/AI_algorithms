import math
from typing import Union
from games.tic_tac_toe import TicTacToe
from games.connect_four import ConnectFour

class Minimax:
    """
    Optimized Minimax algorithm with configurable depth and evaluation strategy,
    supporting both Tic Tac Toe and Connect Four.
    """
    def __init__(self, game, max_depth=5, use_alpha_beta=True, eval_strategy='simple'):
        """
        Initialize Minimax with game instance and parameters.
        
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
        """
        Implement Minimax algorithm with optional alpha-beta pruning.
        
        Args:
            depth (int): Current depth in search tree
            is_maximizing (bool): Whether current player is maximizing
            alpha (float): Best already explored option for maximizer
            beta (float): Best already explored option for minimizer
        
        Returns:
            float: Evaluation score
        """
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

                self._undo_move(move, move_result)

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

                self._undo_move(move, move_result)

                if self.use_alpha_beta:
                    beta = min(beta, eval_score)
                    if beta <= alpha:
                        break

            return min_eval

    def _get_winner(self):
        """
        Get game winner, handling different game types.
        Returns:
            int: Winner (1, -1, 0 for draw, None if ongoing)
        """
        if isinstance(self.game, TicTacToe):
            return self.game.winner if self.game.game_over else None
        else:  # Connect Four
            return self.game.winner if self.game.game_over else None

    def _simulate_move(self, move, player):
        """
        Simulate a move on the game board.
        
        Args:
            move: Move to simulate (depends on game type)
            player (int): Player making the move (1 or -1)
        
        Returns:
            tuple: Position of the move, or None if invalid
        """
        if isinstance(self.game, TicTacToe):
            row, col = move
            if self.game.board[row, col] != 0:
                return None
            self.game.board[row, col] = player
            return (row, col)
        else:  # Connect Four
            col = move
            for row in range(self.game.rows-1, -1, -1):
                if self.game.board[row, col] == 0:
                    self.game.board[row, col] = player
                    return (row, col)
            return None

    def _undo_move(self, move, move_result):
        """
        Undo a simulated move.
        
        Args:
            move: Move to undo
            move_result: Result of the move simulation
        """
        if move_result is None:
            return
        
        if isinstance(self.game, TicTacToe):
            row, col = move_result
            self.game.board[row, col] = 0
        else:  # Connect Four
            row, col = move_result
            self.game.board[row, col] = 0

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
        
        if self.eval_strategy == 'advanced':
            return self._advanced_evaluation()
        
        return 0.5  # Neutral ongoing game state

    def _advanced_evaluation(self):
        """
        Advanced board evaluation for ongoing games.
        Implements heuristic scoring based on potential winning lines.
        """
        score = 0
        if isinstance(self.game, ConnectFour):
            def evaluate_window(window, player):
                count_player = window.count(player)
                count_opponent = window.count(-player)
                count_empty = window.count(0)
                if count_player == 4:
                    return 100  # Win
                elif count_player == 3 and count_empty == 1:
                    return 5
                elif count_player == 2 and count_empty == 2:
                    return 2
                elif count_opponent == 3 and count_empty == 1:
                    return -5
                elif count_opponent == 2 and count_empty == 2:
                    return -2
                return 0

            # Horizontal
            for row in range(self.game.rows):
                for col in range(self.game.cols - 3):
                    window = [self.game.board[row, col + i] for i in range(4)]
                    score += evaluate_window(window, 1)
                    score -= evaluate_window(window, -1)

            # Vertical
            for col in range(self.game.cols):
                for row in range(self.game.rows - 3):
                    window = [self.game.board[row + i, col] for i in range(4)]
                    score += evaluate_window(window, 1)
                    score -= evaluate_window(window, -1)

            # Diagonal /
            for row in range(self.game.rows - 3):
                for col in range(self.game.cols - 3):
                    window = [self.game.board[row + i, col + i] for i in range(4)]
                    score += evaluate_window(window, 1)
                    score -= evaluate_window(window, -1)

            # Diagonal \
            for row in range(3, self.game.rows):
                for col in range(self.game.cols - 3):
                    window = [self.game.board[row - i, col + i] for i in range(4)]
                    score += evaluate_window(window, 1)
                    score -= evaluate_window(window, -1)

        return score

    def get_best_move(self):
        """
        Find the best move using Minimax.
        Returns:
            Move: Best move found, or None if no valid moves
        """
        best_score = -math.inf
        best_move = None
        
        for move in self.game.get_valid_moves():
            move_result = self._simulate_move(move, 1)
            if move_result is None:
                continue
            
            move_score = self.minimax(self.max_depth-1, False)
            
            self._undo_move(move, move_result)
            
            if move_score > best_score:
                best_score = move_score
                best_move = move
        
        return best_move