import numpy as np

class TicTacToe:
    """
    A class to manage the Tic Tac Toe game, implementing game logic with robust error handling
    and documentation for academic distinction level.
    """
    PLAYER_X = 1
    PLAYER_O = -1
    DRAW = 0

    def __init__(self):
        """
        Initialize the game with a 3x3 board, starting with Player X.
        """
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = self.PLAYER_X
        self.game_over = False
        self.winner = None

    def reset(self):
        """
        Reset the game to initial state.
        """
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = self.PLAYER_X
        self.game_over = False
        self.winner = None

    def get_valid_moves(self):
        """
        Return a list of tuples (row, col) for empty positions on the board.
        Returns: List of valid moves.
        """
        return [(i, j) for i in range(3) for j in range(3) if self.board[i, j] == 0]

    def make_move(self, row: int, col: int) -> bool:
        """
        Attempt to make a move at the specified position for the current player.
        Args:
            row (int): Row index (0-2).
            col (int): Column index (0-2).
        Returns:
            bool: True if move was successful, False otherwise.
        Raises:
            ValueError: If row or col is out of bounds.
        """
        if self.game_over:
            return False
        
        if not (0 <= row < 3 and 0 <= col < 3):
            raise ValueError("Row and column must be between 0 and 2.")
        
        if self.board[row, col] != 0:
            return False

        self.board[row, col] = self.current_player
        self.check_winner()
        if not self.game_over:
            self.current_player *= -1  # Switch player
        return True

    def check_winner(self):
        """
        Check if there is a winner or draw. Updates game_over and winner attributes.
        Returns: None, updates internal state.
        """
        # Check rows
        for i in range(3):
            if abs(np.sum(self.board[i, :])) == 3:
                self.winner = self.board[i, 0]
                self.game_over = True
                return

        # Check columns
        for j in range(3):
            if abs(np.sum(self.board[:, j])) == 3:
                self.winner = self.board[0, j]
                self.game_over = True
                return

        # Check diagonals
        if abs(np.sum(np.diag(self.board))) == 3:
            self.winner = self.board[0, 0]
            self.game_over = True
            return
        if abs(np.sum(np.diag(np.fliplr(self.board)))) == 3:
            self.winner = self.board[0, 2]
            self.game_over = True
            return

        # Check for draw (no empty spaces)
        if len(self.get_valid_moves()) == 0:
            self.game_over = True
            self.winner = self.DRAW

    def print_board(self):
        """
        Print the current board state in a readable format.
        """
        symbols = {0: ' ', self.PLAYER_X: 'X', self.PLAYER_O: 'O'}
        for i in range(3):
            print(' | '.join(symbols[num] for num in self.board[i, :]))
            if i < 2:
                print('-' * 9)