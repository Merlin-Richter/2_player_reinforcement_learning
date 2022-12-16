from environments.env_base import Env_base
import numpy as np


class Environment(Env_base):

    def __init__(self):
        super().__init__()
        self.player_turn = 1
        self.render_dict = {0:"-", 1:"♟", 2:"♙"}
        self.index_to_position = [(0, 1), (0, 3), (0, 5), (1, 0), (1, 2), (1, 4), (2, 1), (2, 3), (2, 5), (3, 0), (3, 2), (3, 4), (4, 1), (4, 3), (4, 5), (5, 0), (5, 2), (5, 4)]
        self.hash_matrix = np.array([np.power(3, x) for x in range(18)]).astype(int)
        self.reset()

    def reset(self):

        self.reward = 0
        self.done = False
        self.state = np.array([[1, 1, 1],
                               [1, 1, 1],
                               [0, 0, 0],
                               [0, 0, 0],
                               [2, 2, 2],
                               [2, 2, 2]])
        self.new_state = np.copy(self.state)
        self.player_turn = 1

    def piece_at_position(self, position):
        return self.new_state.flatten()[self.index_to_position.index(position)]

    def get_turn(self):
        return self.player_turn - 1

    def get_actions(self):
        possible_actions = []
        flat_board = self.new_state.flatten()

        for position_index, piece in enumerate(flat_board):
            if piece == self.player_turn:
                position = self.index_to_position[position_index]
                vertical_direction = 1 if self.player_turn == 1 else -1

                for horizontal_direction in [-1, 1]:
                    distance = 1
                    next_position = (position[0] + vertical_direction, position[1] + horizontal_direction)
                    while -1 < next_position[0] < 6 and -1 < next_position[1] < 6:

                        piece_at_pos = flat_board[self.index_to_position.index(next_position)]

                        if piece_at_pos != self.player_turn and not (distance != 1 and piece_at_pos != 0):

                            possible_actions.append((self.index_to_position.index(position), self.index_to_position.index(next_position)))
                            if distance == 1 and piece_at_pos != 0:
                                break
                            next_position = (next_position[0] + vertical_direction, next_position[1] + horizontal_direction)
                            distance += 1

                        else:
                            break
        return possible_actions

    def inner_workings(self, action):
        # No check for rather the move is legal or not for efficiency reasons
        from_position = action[0]
        to_position = action[1]
        self.new_state[np.floor(from_position / 3).astype(int), from_position % 3] = 0
        self.new_state[np.floor(to_position / 3).astype(int), to_position % 3] = self.player_turn
        self.player_turn = 1 if self.player_turn == 2 else 2
        if 1 in self.new_state[5]:
            self.reward = 1
            self.done = True
        elif 2 in self.new_state[0]:
            self.reward = 0
            self.done = True
        if len(np.unique(self.new_state)) == 2:
            self.done = True
            if np.unique(self.new_state)[1] == 1:
                self.reward = 1
            else:
                self.reward = 0

    def render(self):
        for x in [2,1,0]:
            print(str(2*x+2) + " " + " - ".join([self.render_dict[self.new_state[2*x + 1, y]] for y in range(3)]) + " - ")
            print(str(2*x+1) + " - " + " - ".join([self.render_dict[self.new_state[2*x, y]] for y in range(3)]))

        print("  a b c d e f ")

    def hash(self):
        return np.sum(self.new_state.flatten() * self.hash_matrix) + 500_000_000 * (self.player_turn-1)

    def vector_representation(self, polynomials=False):
        if not polynomials:
            return np.append(np.append(self.new_state.flatten() == 1, self.new_state.flatten() == 2).astype(int), 1)
        else:
            vector_rep = np.append(self.new_state.flatten() == 1, self.new_state.flatten() == 2).astype(int)
            outer = np.outer(vector_rep, vector_rep)
            return np.append(np.concatenate(np.array([outer[x, x:] for x in range(36)], dtype=object)), 1)


if __name__ == "__main__":

    Env = Environment()

    Env.render()

    alphabet = ['a', 'b', 'c', 'd', 'e', 'f']

    while True:
        while not Env.done:
            Env.render()
            Player_action = (0, 0)
            print(Env.get_actions())
            while Player_action not in Env.get_actions():
                start = input("which piece do you want to play?  ")
                end = input('where to?  ')
                try:
                    start_position = (int(start[0]) - 1, alphabet.index(start[1]))
                    end_position = (int(end[0]) - 1, alphabet.index(end[1]))
                except:
                    print("input incorrect")
                    continue
                Player_action = (Env.index_to_position.index(start_position), Env.index_to_position.index(end_position))
                if Player_action not in Env.get_actions():
                    print("\nThat move is not a legal move\n")
            Env.step(Player_action)

        Env.render()
        Env.reset()
        print("\nGAME OVER\n")
        input("press enter to play again")


