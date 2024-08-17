import random

from .keys import KEY_RIGHT, KEY_UP


class AutoPlayer:
    def __init__(self):
        self.available_moves = [KEY_UP, KEY_RIGHT, None]

    def choose_moves(self, player, next_obstacle):
        chosen_moves = []
        chosen_moves.append(self.roll_move(correct_move=KEY_RIGHT))

        if next_obstacle:
            distance_to_obstacle = (next_obstacle.x) - (player.x + player.w)
            is_player_near_obstacle_right = distance_to_obstacle >= 0 and distance_to_obstacle <= 2
            is_player_below_obstacle = (player.y + player.h) > next_obstacle.y
            if is_player_near_obstacle_right and is_player_below_obstacle:
                chosen_moves.append(self.roll_move(KEY_UP))
                return chosen_moves
        chosen_moves.append(self.roll_move(None))
        return chosen_moves

    def roll_move(self, correct_move, p_correct=0.9):
        # Equal probability for all wrong moves
        q = (1 - p_correct) / (len(self.available_moves) - 1)
        weights = [
            p_correct if available_move == correct_move else q
            for available_move in self.available_moves
        ]
        chosen_move = random.choices(self.available_moves, weights, k=1)[0]
        return chosen_move
