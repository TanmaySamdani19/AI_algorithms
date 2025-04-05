import numpy as np

class ConnectFour:
    def __init__(self, rows=6, cols=7):
        self.rows = rows
        self.cols = cols
        self.board = np.zeros((rows, cols), dtype=int)
        self.current_player = 1
        self.game_over = False
        self.winner = None

    def reset(self):
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1
        self.game_over = False
        self.winner = None

    def get_valid_moves(self):
        return [col for col in range(self.cols) if self.board[0][col] == 0]

    def make_move(self, col):
        if col not in self.get_valid_moves():
            return False

        for row in range(self.rows-1, -1, -1):
            if self.board[row][col] == 0:
                self.board[row][col] = self.current_player
                self.check_winner(row, col)
                self.current_player = 3 - self.current_player
                return True
        return False

    def check_winner(self, last_row, last_col):
        # Check horizontal
        for c in range(max(0, last_col-3), min(self.cols-3, last_col+1)):
            if all(self.board[last_row][c+i] == self.current_player for i in range(4)):
                self.game_over = True
                self.winner = self.current_player
                return self.winner

        # Check vertical
        for r in range(max(0, last_row-3), min(self.rows-3, last_row+1)):
            if all(self.board[r+i][last_col] == self.current_player for i in range(4)):
                self.game_over = True
                self.winner = self.current_player
                return self.winner

        # Check diagonal (top-left to bottom-right)
        for r in range(max(0, last_row-3), min(self.rows-3, last_row+1)):
            for c in range(max(0, last_col-3), min(self.cols-3, last_col+1)):
                if all(self.board[r+i][c+i] == self.current_player for i in range(4)):
                    self.game_over = True
                    self.winner = self.current_player
                    return self.winner

        # Check diagonal (top-right to bottom-left)
        for r in range(max(0, last_row-3), min(self.rows-3, last_row+1)):
            for c in range(max(3, last_col-3), min(self.cols, last_col+4)):
                if all(self.board[r+i][c-i] == self.current_player for i in range(4)):
                    self.game_over = True
                    self.winner = self.current_player
                    return self.winner

        # Check for draw
        if len(self.get_valid_moves()) == 0:
            self.game_over = True
            return 0

        return None

    def print_board(self):
        symbols = {0: ' ', 1: 'X', -1: 'O'}
        for row in self.board:
            print('|'.join(symbols[cell] for cell in row))
        print('-' * (2 * self.cols - 1))