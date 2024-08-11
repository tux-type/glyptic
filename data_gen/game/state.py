import pyxel
from elements import Player, Platform, Obstacle
from keys import KEY_RIGHT, KEY_UP, KEY_QUIT
from collections import deque

from datetime import datetime
import json
from pathlib import Path
from bot import AutoPlayer


class BasicGame:
    def __init__(self, w=45, h=30, fps=5, collect_data=True, auto_play=True):
        self.collect_data = collect_data

        self.w = w
        self.h = h
        pyxel.init(width=w, height=h, fps=fps, display_scale=1)
        pyxel.fullscreen(False)

        self.update_counter = 0
        self.sc_flag = False
        self.base_y_position = pyxel.floor(pyxel.height * (2 / 3))
        self.base_platform = Platform(
            x=0,
            y=self.base_y_position,
            w=w,
            h=h // 15,
        )
        self.player = Player(
            x=w // 10,
            y=self.base_y_position - (h // 10),
            w=h // 15,
            h=h // 10,
            surface_y=self.base_platform.y,
        )
        self.offset = 0

        self.obstacle_locations = [4, 6, 10, 25, 35, 70]
        self.obstacles = deque()
        # Next obstacle is a misleading name, can be obstacle behind if no more ahead
        self.next_obstacle = None

        self.activated_keys = []

        self.auto_play = auto_play
        self.auto_player = AutoPlayer()

        if collect_data:
            data_dir_name = "collection_" + datetime.today().strftime("%Y%m%d-%H%M%S")

            self.data_collection_dir = (
                Path(__file__).parents[2] / "data" / "combined" / data_dir_name
            )
            # TODO: Replace with logger
            print(f"Assuming project directory: {self.data_collection_dir.parents[2]}")
            print(f"Creating directory: {self.data_collection_dir}")
            self.data_collection_dir.mkdir(parents=False, exist_ok=False)
            self.keys_with_id = []

    def render_background(self, col_a=7, col_b=13):
        col_a, col_b = (col_a, col_b) if self.offset % 2 == 0 else (col_b, col_a)
        [
            pyxel.rect(x=x, y=y, w=1, h=1, col=col_a)
            if (x % 2 == 0) != (y % 2 == 0)
            else pyxel.rect(x=x, y=y, w=1, h=1, col=col_b)
            for x in range(pyxel.width)
            for y in range(pyxel.height)
        ]

    def create_obstacle(self, immune=True):
        w = self.h // 15
        h = self.h // 15
        obstacle = Obstacle(x=self.w - w, y=self.base_y_position - h, w=w, h=h, immune=immune)
        self.obstacles.append(obstacle)

    def move_obstacles(self):
        assert len(self.obstacles) > 0, "trying to move obstacles when none present"
        for obstacle in self.obstacles:
            obstacle.x -= 1
        self.update_next_obstacle()
        if self.next_obstacle and self.next_obstacle.x < 0:
            self.destroy_obstacle()

    def destroy_obstacle(self):
        self.obstacles.popleft()

    def update_next_obstacle(self):
        if not self.obstacles:
            self.next_obstacle = None
            return
        else:
            for obstacle in self.obstacles:
                if (obstacle.x + obstacle.w) > self.player.x:
                    self.next_obstacle = obstacle
                    return

    def save_screenshot(self):
        pyxel.screenshot(scale=1)

        # pyxel only saves screenshots in Desktop
        screenshot_path = Path("~/Desktop").expanduser()
        screenshots = list(screenshot_path.glob("pyxel*.png"))
        assert len(screenshots) == 1, "there should be only one pyxel png screenshot in save dir"
        new_name = (
            str(self.data_collection_dir)
            + "/"
            + screenshots[0].stem
            + "_X"
            + str(self.update_counter)
            + screenshots[0].suffix
        )
        screenshots[0].rename(new_name)

    def save_pressed_keys(self, write=False):
        update_id = "X" + str(self.update_counter)
        self.keys_with_id.append({update_id: self.activated_keys})
        if write:
            keys_file = self.data_collection_dir / "key_activations.json"
            with open(keys_file, "w") as f_out:
                json.dump(self.keys_with_id, f_out)

    def update(self):
        if self.collect_data:
            # Acts as an ID for key press and image data
            self.update_counter += 1

        self.activated_keys = []
        if self.auto_play:
            self.activated_keys = self.auto_player.choose_moves(
                player=self.player, next_obstacle=self.next_obstacle
            )
        else:
            if pyxel.btnp(KEY_RIGHT, hold=0, repeat=1):
                self.activated_keys.append(KEY_RIGHT)
            if pyxel.btn(KEY_UP):
                self.activated_keys.append(KEY_UP)
        if pyxel.btn(KEY_QUIT):
            self.activated_keys.append(KEY_QUIT)


        self.update_next_obstacle()

        if KEY_RIGHT in self.activated_keys:
            if not self.obstacles:
                self.offset += 1
            else:
                assert self.next_obstacle is not None, "next_obstacle should be set"
                # print(f"player x: {self.player.x}")
                # print(f"player x + player.w: {self.player.x + self.player.w}")
                # print(f"player y: {self.player.y}")
                # print(f"player y + player.h: {self.player.y + self.player.h}")
                # print(f"player surface_y: {self.player.surface_y}")
                # print(f"obstacle y: {self.next_obstacle.y}")
                # print(f"obstacle x: {self.next_obstacle.x}")
                # print("-----")

                is_player_colliding_right = (self.player.x + self.player.w) == self.next_obstacle.x
                is_player_above_obstacle = (self.player.y + self.player.h) <= self.next_obstacle.y
                if not is_player_colliding_right or is_player_above_obstacle:
                    self.offset += 1
                    self.move_obstacles()

        # Check if player needs to fall
        if self.next_obstacle:
            assert self.next_obstacle is not None, "next_obstacle should be set"
            is_player_above_obstacle = (
                self.player.x + self.player.w > self.next_obstacle.x
                and self.player.x < self.next_obstacle.x + self.next_obstacle.w
            )
            if is_player_above_obstacle:
                self.player.surface_y = self.next_obstacle.y
            elif self.player.surface_y != self.base_platform.y:
                self.player.surface_y = self.base_platform.y
                self.player.fall()

        if KEY_UP in self.activated_keys:
            # TODO: Handle both keys pressed together better (UP AND RIGHT)
            self.player.jump()

        if self.offset in self.obstacle_locations:
            self.create_obstacle()

        self.player.update()

        quit_game = KEY_QUIT in self.activated_keys
        if self.collect_data:
            self.save_screenshot()
            self.save_pressed_keys(write=quit_game)

        if quit_game:
            pyxel.quit()

    def draw(self):
        self.render_background()
        self.base_platform.render(col=15)
        self.player.render()
        for obstacle in self.obstacles:
            obstacle.render()

    def run(self):
        pyxel.run(self.update, self.draw)
