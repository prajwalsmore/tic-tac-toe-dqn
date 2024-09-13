# tictactoe_env.py
import numpy as np

class TicTacToe:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.done = False
        return self.board.flatten()

    def is_winner(self, player):
        for i in range(3):
            if np.all(self.board[i, :] == player) or np.all(self.board[:, i] == player):
                return True
        if np.all(np.diag(self.board) == player) or np.all(np.diag(np.fliplr(self.board)) == player):
            return True
        return False

    def is_draw(self):
        return np.all(self.board != 0)

    def step(self, action, player):
        x, y = divmod(action, 3)
        if self.board[x, y] == 0:
            self.board[x, y] = player
            if self.is_winner(player):
                self.done = True
                return self.board.flatten(), 1 if player == 1 else -1, self.done
            elif self.is_draw():
                self.done = True
                return self.board.flatten(), 0.5, self.done  # Draw reward
            elif self.preventing_opponent_win(player):  # Check if opponent would win next
                return self.board.flatten(), 0.2, False  # Reward for blocking
            else:
                return self.board.flatten(), 0, False
        else:
            return self.board.flatten(), -10, False  # Invalid move penalty

    def preventing_opponent_win(self, player):
        opponent = -1 if player == 1 else 1
        return self.is_winner(opponent)

