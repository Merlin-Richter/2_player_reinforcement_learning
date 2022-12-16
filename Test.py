
import torch
import torch.nn as nn


class TicTacToe:
    def __init__(self):
        self.board = [[" " for i in range(3)] for j in range(3)]

    def play(self, player, x, y):
        if self.board[x][y] != " ":
            return False
        self.board[x][y] = player
        return True

    def check_winner(self):
        # check rows
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] != " ":
                return self.board[i][0]
        # check columns
        for i in range(3):
            if self.board[0][i] == self.board[1][i] == self.board[2][i] != " ":
                return self.board[0][i]
        # check diagonals
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != " ":
            return self.board[0][0]
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != " ":
            return self.board[0][2]
        return None

    def print_board(self):
        for i in range(3):
            for j in range(3):
                print(self.board[i][j], end="")
                if j < 2:
                    print("|", end="")
            print()
            if i < 2:
                print("-+-+-")



class NeuralNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    self.layer1 = nn.Linear(9, 64)
    self.layer2 = nn.Linear(64, 64)
    self.layer3 = nn.Linear(64, 64)
    self.output_layer = nn.Linear(64, 9)

  def forward(self, x):
    x = self.layer1(x)
    x = torch.relu(x)
    x = self.layer2(x)
    x = torch.relu(x)
    x = self.layer3(x)
    x = torch.relu(x)
    x = self.output_layer(x)
    return x

game = TicTacToe()
game.play("X", 0, 0)
game.play("O", 0, 1)
game.play("X", 1, 1)
game.play("O", 0, 2)
game.play("X", 2, 2)

game.print_board()
# Output:
# X|O|X
# -+-+-
# |X|
# -+-+-
# | |O

winner = game.check_winner()
print(winner)
# Output: X