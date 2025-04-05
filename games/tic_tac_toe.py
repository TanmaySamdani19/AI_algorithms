import numpy as np

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        self.game_over = False
        self.winner = None

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        self.game_over = False
        self.winner = None

    def get_valid_moves(self):
        return [(x, y) for x in range(3) for y in range(3) if self.board[x][y] == 0]

    def make_move(self, x, y):
        if self.board[x][y] == 0 and not self.game_over:
            self.board[x][y] = self.current_player
            self.check_winner()
            self.current_player = 3 - self.current_player
            return True
        return False

    def check_winner(self):
        # Check rows, columns, and diagonals
        for i in range(3):
            # Check rows
            if abs(sum(self.board[i, :])) == 3:
                self.game_over = True
                self.winner = self.board[i, 0]
                return self.winner
            
            # Check columns
            if abs(sum(self.board[:, i])) == 3:
                self.game_over = True
                self.winner = self.board[0, i]
                return self.winner
        
        # Check diagonals
        if abs(sum(np.diag(self.board))) == 3:
            self.game_over = True
            self.winner = self.board[1, 1]
            return self.winner
        
        if abs(sum(np.diag(np.fliplr(self.board)))) == 3:
            self.game_over = True
            self.winner = self.board[1, 1]
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
            print('-' * 5)