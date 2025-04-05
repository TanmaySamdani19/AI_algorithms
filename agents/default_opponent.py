from games.tic_tac_toe import TicTacToe
from games.connect_four import ConnectFour
import random
import numpy as np

class DefaultOpponent:
    """
    Implements a default opponent that is better than random, prioritizing winning and blocking moves.
    """
    def __init__(self, game):
        """
        Initialize with game instance.
        
        Args:
            game: Game instance (TicTacToe or ConnectFour)
        """
        self.game = game

    def get_move(self):
        """
        Get the next move, prioritizing winning moves, then blocking opponent, then random.
        Returns:
            Move: Chosen move
        """
        valid_moves = self.game.get_valid_moves()
        if not valid_moves:
            return None

        # Check for winning move
        for move in valid_moves:
            game_copy = self._copy_game()
            if isinstance(self.game, TicTacToe):
                row, col = move
                if game_copy.make_move(row, col):
                    if game_copy.winner == self.game.current_player:
                        return move
            else:  # Connect Four
                if game_copy.make_move(move):
                    game_copy.check_winner(game_copy.rows-1, move)
                    if game_copy.game_over and game_copy.winner == self.game.current_player:
                        return move

        # Check for blocking opponent's winning move
        opponent_player = -self.game.current_player
        for move in valid_moves:
            game_copy = self._copy_game()
            if isinstance(self.game, TicTacToe):
                row, col = move
                if game_copy.make_move(row, col):
                    game_copy.current_player = opponent_player
                    if game_copy.check_winner() == opponent_player:
                        return move
            else:  # Connect Four
                if game_copy.make_move(move):
                    game_copy.current_player = opponent_player
                    game_copy.check_winner(game_copy.rows-1, move)
                    if game_copy.game_over and game_copy.winner == opponent_player:
                        return move

        # If no winning or blocking move, choose randomly
        return random.choice(valid_moves)

    def _copy_game(self):
        """
        Create a deep copy of the current game state.
        Returns:
            Game instance: Copy of the current game
        """
        if isinstance(self.game, TicTacToe):
            new_game = TicTacToe()
            new_game.board = np.copy(self.game.board)
            new_game.current_player = self.game.current_player
            new_game.game_over = self.game.game_over
            new_game.winner = self.game.winner
            return new_game
        else:  # Connect Four
            new_game = ConnectFour(self.game.rows, self.game.cols)
            new_game.board = np.copy(self.game.board)
            new_game.current_player = self.game.current_player
            new_game.game_over = self.game.game_over
            new_game.winner = self.game.winner
            return new_game