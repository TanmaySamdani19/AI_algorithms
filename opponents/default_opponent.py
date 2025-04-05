import random
from typing import Union, List
from games.tic_tac_toe import TicTacToe
from games.connect_four import ConnectFour

class DefaultOpponent:
    def __init__(self, game):
        """
        Initialize a more intelligent opponent that:
        1. Checks for winning moves
        2. Blocks opponent's winning moves
        3. Falls back to random move if no strategic move exists
        """
        self.game = game

    def get_move(self) -> Union[tuple, int]:
        """
        Intelligent move selection strategy:
        1. Check for winning move
        2. Check for blocking move
        3. Fallback to random move
        """
        # Create a copy of the game to simulate moves
        game_copy = self._copy_game()
        
        # Try to find a winning move
        winning_move = self._find_strategic_move(1)
        if winning_move is not None:
            return winning_move
        
        # Try to block opponent's winning move
        blocking_move = self._find_strategic_move(-1)
        if blocking_move is not None:
            return blocking_move
        
        # Fallback to random move
        valid_moves = game_copy.get_valid_moves()
        return random.choice(valid_moves)

    def _copy_game(self):
        """Create a deep copy of the current game state."""
        if hasattr(self.game, 'rows') and hasattr(self.game, 'cols'):
            # For Connect Four
            game_copy = type(self.game)(rows=self.game.rows, cols=self.game.cols)
        else:
            # For Tic Tac Toe
            game_copy = type(self.game)()
        
        game_copy.board = self.game.board.copy()
        game_copy.current_player = self.game.current_player
        return game_copy

    def _find_strategic_move(self, player):
        """
        Find a winning or blocking move for a given player.
        
        Args:
            player (int): Player to check moves for (1 or -1)
        
        Returns:
            move: Winning or blocking move, or None
        """
        game_copy = self._copy_game()
        
        for move in game_copy.get_valid_moves():
            # Simulate move
            if isinstance(game_copy, TicTacToe):
                game_copy.board[move[0]][move[1]] = player
                if game_copy.check_winner() == player:
                    return move
                game_copy.board[move[0]][move[1]] = 0
            else:  # Connect Four
                col = move
                for row in range(game_copy.rows-1, -1, -1):
                    if game_copy.board[row][col] == 0:
                        game_copy.board[row][col] = player
                        winner = game_copy.check_winner(row, col)
                        if winner == player:
                            return col
                        game_copy.board[row][col] = 0
                        break
        
        return None