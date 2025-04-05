import numpy as np

class ConnectFour:
    """
    A class to manage the Connect Four game, implementing game logic with robust error handling
    and documentation for academic distinction level.
    """
    def __init__(self, rows=6, cols=7):
        """
        Initialize the game with a board of specified rows and columns, starting with Player 1.
        Args:
            rows (int): Number of rows in the board. Default is 6.
            cols (int): Number of columns in the board. Default is 7.
        """
        self.rows = rows
        self.cols = cols
        self.board = np.zeros((rows, cols), dtype=int)
        self.current_player = 1
        self.game_over = False
        self.winner = None

    def reset(self):
        """
        Reset the game to initial state.
        """
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1
        self.game_over = False
        self.winner = None

    def get_valid_moves(self):
        """
        Return a list of columns that are not full.
        Returns: List of valid column indices.
        """
        return [col for col in range(self.cols) if self.board[0, col] == 0]

    def make_move(self, col: int) -> bool:
        """
        Attempt to make a move in the specified column for the current player.
        Args:
            col (int): Column index to drop the piece.
        Returns:
            bool: True if move was successful, False otherwise.
        Raises:
            ValueError: If col is out of bounds.
        """
        if self.game_over:
            return False

        if not (0 <= col < self.cols):
            raise ValueError("Column must be between 0 and {}.".format(self.cols - 1))

        if self.board[0, col] != 0:
            return False  # Column is full

        # Find the lowest empty row in the column
        for row in range(self.rows - 1, -1, -1):
            if self.board[row, col] == 0:
                self.board[row, col] = self.current_player
                self.check_winner(row, col)
                if not self.game_over:
                    self.current_player = -self.current_player
                return True
        return False  # Should not reach here

    def check_winner(self, last_row: int, last_col: int):
        """
        Check if the last move resulted in a win or draw.
        Args:
            last_row (int): Row index of the last move.
            last_col (int): Column index of the last move.
        Returns: None, updates internal state.
        """
        player = self.board[last_row, last_col]
        # Check horizontal
        count = 0
        for c in range(self.cols):
            if self.board[last_row, c] == player:
                count += 1
                if count == 4:
                    self.winner = player
                    self.game_over = True
                    return
            else:
                count = 0
        # Check vertical
        count = 0
        for r in range(self.rows):
            if self.board[r, last_col] == player:
                count += 1
                if count == 4:
                    self.winner = player
                    self.game_over = True
                    return
            else:
                count = 0
        # Check diagonal (top-left to bottom-right)
        count = 0
        for r, c in [(last_row + dr, last_col + dr) for dr in range(-3, 4) if 0 <= last_row + dr < self.rows and 0 <= last_col + dr < self.cols]:
            if self.board[r, c] == player:
                count += 1
                if count == 4:
                    self.winner = player
                    self.game_over = True
                    return
            else:
                count = 0
        # Check diagonal (top-right to bottom-left)
        count = 0
        for r, c in [(last_row + dr, last_col - dr) for dr in range(-3, 4) if 0 <= last_row + dr < self.rows and 0 <= last_col - dr < self.cols]:
            if self.board[r, c] == player:
                count += 1
                if count == 4:
                    self.winner = player
                    self.game_over = True
                    return
            else:
                count = 0
        # Check for draw
        if len(self.get_valid_moves()) == 0:
            self.game_over = True
            self.winner = 0  # Draw

    def print_board(self):
        """
        Print the current board state in a readable format.
        """
        symbols = {0: ' ', 1: 'X', -1: 'O'}
        for row in range(self.rows):
            print(' | '.join(symbols[self.board[row, col]] for col in range(self.cols)))
            if row < self.rows - 1:
                print('-' * (4 * self.cols - 1))