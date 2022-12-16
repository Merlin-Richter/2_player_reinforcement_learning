import environments.env_base as Env_base
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

images = {"Plain": mpimg.imread('../Uniwar_images/plain.png'),
          "Forest": mpimg.imread('../Uniwar_images/forest.png'),
          "Void": np.zeros((10, 10, 4)),
          "BMecha": mpimg.imread('../Uniwar_images/Bmecha.png'),
          "BSpeeder": mpimg.imread('../Uniwar_images/Bspeeder.png'),
          "BGuardian": mpimg.imread('../Uniwar_images/Bguardian.png'),
          "BPlasma": mpimg.imread('../Uniwar_images/Bplasma.png'),
          "RMecha": mpimg.imread('../Uniwar_images/Rmecha.png'),
          "RSpeeder": mpimg.imread('../Uniwar_images/Rspeeder.png'),
          "RGuardian": mpimg.imread('../Uniwar_images/Rguardian.png'),
          "RPlasma": mpimg.imread('../Uniwar_images/Rplasma.png')

          }

Global_Game_tiles = np.array([["Plain", "Forest"],
                              ["Forest", "Forest"],
                              ["Forest", "Forest"]])

Damage_table = np.load("../Look_up_table.npy")

class Board():

    def __init__(self):
        self.Game_units = np.array([["BSpeeder_", "x"],
                                    ["x", "RPlasma_"],
                                    ["x", "x"]])
        self.Game_hp = np.array([[10, 0],
                                 [0, 10],
                                 [0, 0]])
        self.credits = np.array([0, 0])
        self.turn = "B"

    def end_turn(self):
        self.turn = "B" if self.turn == "R" else "R"
        for i, j in np.array(np.where((self.Game_units.astype("<U1") == "_"))).T:
            self.Game_units[i, j] = self.Game_units[i, j][1:] + "_"

    def __eq__(self, other):
        if isinstance(other, Board):
            return (self.Game_units == other.Game_units).all() and (self.Game_hp == other.Game_hp).all() and (self.credits == other.credits).all() and self.turn == other.turn

    def render(self):
        for i in range(self.Game_units.shape[0]):
            for j in range(self.Game_units.shape[1]):
                x, y = j, i
                plt.imshow(images[Global_Game_tiles[i, j]], extent=[-(x % 2) * 50 + 100 * y+50, (y + 1) * 100 - (x % 2) * 50+50, x * 75, 100 + x * 75])
                if self.Game_units[i, j] != "x":
                    plt.imshow(images[self.Game_units[i, j].replace("_", "")], extent=[-(x % 2) * 50 + 100 * y + 13+50, (y + 1) * 100 - 12 - (x % 2) * 50+50, x * 75 + 13, 100 + x * 75 - 12])
                    plt.text(-(x % 2) * 50 + 100 * y+50, x*100, self.Game_hp[i, j])
        plt.xlim([0, 300])
        plt.ylim([0, 300])
        plt.show()




class Environment(Env_base.Env_base):

    def __init__(self):
        super().__init__()
        self.stats = {"Mecha": {"class":"light", "light":6, "heavy":3, "defence":6, "mobility":8, "repair":1, "piercing":1, "max_range":1},
                 "Speeder": {"class":"heavy", "light":10, "heavy":5, "defence":8, "mobility":16, "repair":2, "piercing":1, "max_range":1},
                 "Plasma": {"class": "heavy", "light": 10, "heavy": 12, "defence": 14, "mobility": 7, "repair":1, "piercing":1, "max_range":1},
                 "Guardian": {"class": "light", "light": 7, "heavy": 5, "defence": 3, "mobility": 10, "repair":0, "piercing":0.6, "max_range":2}}

        self.mobility = {"light" : {"Forest": 4, "Plain": 3, "BaseRed": 4, "BaseBlue": 4, "Base": 4, "Void":100},
                    "heavy" : {"Forest": 6, "Plain": 3, "BaseRed": 4, "BaseBlue": 4, "Base": 4, "Void":100}
                    }

        self.attack_bonus = {"light" : {"Forest": 2, "Plain": 0, "BaseRed": 2, "BaseBlue": 2, "Base": 2, "Void":0},
                    "heavy" : {"Forest": 0, "Plain": 0, "BaseRed": 0, "BaseBlue": 0, "Base": 0, "Void":0}
                    }

        self.defence_bonus = {"light": {"Forest": 3, "Plain": 0, "BaseRed": 2, "BaseBlue": 2, "Base": 2, "Void": 0},
                             "heavy": {"Forest": -3, "Plain": 0, "BaseRed": -1, "BaseBlue": -1, "Base": -1, "Void": 0}
                             }

        self.Game_tiles = Global_Game_tiles
        self.credits_per_base = 100

        self.boards = [Board()]

    def possible_board_transitions(self, board):
        liste = []
        enemy = "R" if board.turn == "B" else "B"
        team = board.turn
        indexes = np.array(np.where((board.Game_units.astype("<U1") == team))).T
        for i, j in indexes:
            unit = board.Game_units[i, j][1:].replace("_", "")

            new_board = copy.deepcopy(board)
            new_board.Game_units[i, j] = "_" + board.Game_units[i, j][:-1]

            liste += self.attack_neihbors(new_board, (i, j), unit, enemy)
            new_board.Game_hp[i, j] = min(10, new_board.Game_hp[i, j] + self.stats[unit]["repair"])
            liste.append(new_board)

            tiles = np.array(self.all_actions_of_unit_at(board, i, j, self.stats[unit]["mobility"], self.stats[unit]["class"], team))
            tiles = np.unique(tiles, axis=0)
            for tile in tiles:

                if board.Game_units[tile[0], tile[1]] == "x":
                    new_board = copy.deepcopy(board)
                    new_board.Game_units[tile[0], tile[1]] = "_" + board.Game_units[i, j][:-1]
                    new_board.Game_units[i, j] = "x"
                    new_board.Game_hp[tile[0], tile[1]] = board.Game_hp[i, j]
                    new_board.Game_hp[i, j] = 0
                    print(tile)
                    liste.append(new_board)
                    if tile[2]:
                        liste += self.attack_neihbors(new_board, tile, unit, enemy)


        return liste

    def attack_neihbors(self, board, tile, unit, enemy):
        N_tiles = self.get_neighbouring_tiles(tile[0], tile[1])
        liste = []
        print("now")
        board.render()
        print("not now")
        for N in N_tiles:
            if board.Game_units.astype("<U1")[N[0], N[1]] == enemy:
                new_board = copy.deepcopy(board)
                E_unit = new_board.Game_units[N[0], N[1]][1:].replace("_", "")
                new_board = copy.deepcopy(new_board)
                HP = new_board.Game_hp[N[0], N[1]]

                attack = self.stats[unit][self.stats[E_unit]["class"]] + self.attack_bonus[self.stats[unit]["class"]][self.Game_tiles[tile[0], tile[1]]]
                defence = self.stats[E_unit]["defence"] + self.defence_bonus[self.stats[E_unit]["class"]][self.Game_tiles[N[0], N[1]]]

                if self.stats[E_unit]["class"] == "heavy":
                    defence = int(round(defence * self.stats[unit]["piercing"]))
                difference = max(-10, min(10, int(attack - defence)))
                new_board.Game_hp[N[0], N[1]] -= int(
                    round(Damage_table[new_board.Game_hp[tile[0], tile[1]], difference + 10]))

                attack = self.stats[E_unit][self.stats[unit]["class"]] + self.attack_bonus[self.stats[E_unit]["class"]][self.Game_tiles[N[0], N[1]]]
                defence = self.stats[unit]["defence"] + self.defence_bonus[self.stats[unit]["class"]][self.Game_tiles[tile[0], tile[1]]]

                if self.stats[unit]["class"] == "heavy":
                    defence = int(round(defence * self.stats[E_unit]["piercing"]))
                difference = max(-10, min(10, int(attack - defence)))

                new_board.Game_hp[tile[0], tile[1]] -= int(round(Damage_table[HP, difference + 10]))
                if new_board.Game_hp[tile[0], tile[1]] < 1:
                    new_board.Game_hp[tile[0], tile[1]] = 0
                    new_board.Game_units[tile[0], tile[1]] = "x"
                if new_board.Game_hp[N[0], N[1]] < 1:
                    new_board.Game_hp[N[0], N[1]] = 0
                    new_board.Game_units[N[0], N[1]] = "x"
                liste.append(new_board)
        return liste

    def all_actions_of_unit_at(self, board, i, j, movement_points, class_type, team, first=True):
        enemy = "R" if team == "B" else "B"
        returny = [(i, j, False)]
        positions = self.get_neighbouring_tiles(i, j)
        for pos in positions:
            if board.Game_units[pos[0], pos[1]][0] == enemy:
                if first:
                    returny[0] = (i, j, True)
                    print(returny)
                else:
                    return [(i, j, True)]
            movement_cost = Env.mobility[class_type][Env.Game_tiles[pos[0], pos[1]]]
            if movement_points >= movement_cost:
                returny += self.all_actions_of_unit_at(board, pos[0], pos[1], movement_points - movement_cost,class_type, team, False)
        return returny

    def get_neighbouring_tiles(self, i, j):
        returny = []
        if j+1 < self.Game_tiles.shape[1]:
            returny.append((i, j+1))
        if j > 0:
            returny.append((i, j-1))
        if i > 0:
            returny.append((i-1, j))
            if i % 2 == 1 and j > 0:
                returny.append((i - 1, j-1))
            elif i % 2 == 0 and j+1 < self.Game_tiles.shape[1]:
                returny.append((i - 1, j + 1))
        if i+1 < self.Game_tiles.shape[0]:
            returny.append((i+1, j))
            if i % 2 == 1 and j > 0:
                returny.append((i + 1, j-1))
            elif i % 2 == 0 and j+1 < self.Game_tiles.shape[1]:
                returny.append((i + 1, j + 1))
        return returny


Env = Environment()

liste = []

board = Env.possible_board_transitions(Board())
board[-1].end_turn()
board[-1].render()
board2 = Env.possible_board_transitions(board[-1])
board2[-1].render()






