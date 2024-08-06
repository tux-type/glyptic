import pyxel
from elements import Player, Platform, Obstacle
from keys import KEY_RIGHT, KEY_UP
from collections import deque


class BasicGame:
    def __init__(self, w=45, h=30, fps=5):
        self.w = w
        self.h = h
        pyxel.init(width=w, height=h, fps=fps)
        pyxel.fullscreen(False)

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
        if self.obstacles[0].x < 0:
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

    def update(self):
        self.update_next_obstacle()

        if pyxel.btnp(KEY_RIGHT, hold=0, repeat=1):
            if not self.obstacles:
                self.offset += 1
            else:
                assert self.next_obstacle is not None, "next_obstacle should be set"
                print(f"player x: {self.player.x}")
                print(f"player x + player.w: {self.player.x + self.player.w}")
                print(f"player y: {self.player.y}")
                # print(f"player y + player.h: {self.player.y + self.player.h}")
                print(f"obstacle y: {self.obstacles[0].y}")
                print(f"obstacle x: {self.obstacles[0].x}")
                print("-----")

                is_player_colliding_right = (self.player.x + self.player.w) == self.next_obstacle.x
                is_player_higher_than_obstacle = (self.player.y + self.player.h) <= self.obstacles[
                    0
                ].y
                if not is_player_colliding_right or is_player_higher_than_obstacle:
                    self.offset += 1
                    self.move_obstacles()

        if pyxel.btn(KEY_UP):
            # TODO: Handle both keys pressed together better (UP AND RIGHT)
            print("KEY_UP")
            if not self.player.jumping and not self.player.falling:
                self.player.jump()

        # Check if need to fall
        if self.next_obstacle:
            assert self.next_obstacle is not None, "next_obstacle should be set"
            is_player_above_obstacle = (
                self.player.x + self.player.w > self.next_obstacle.x
                and self.player.x < self.next_obstacle.x + self.next_obstacle.w
            )
            if is_player_above_obstacle:
                self.player.surface_y = self.next_obstacle.y
                print("Above obstacle")
            elif self.player.surface_y != self.base_platform.y:
                self.player.surface_y = self.base_platform.y
                self.player.falling = True

        if self.offset in self.obstacle_locations:
            self.create_obstacle()

        self.player.update()

        if pyxel.btnp(pyxel.KEY_Q):
            pyxel.quit()

    def draw(self):
        # pyxel.cls(0)
        self.render_background()
        self.base_platform.render(col=15)
        self.player.render()
        for obstacle in self.obstacles:
            obstacle.render()

    def run(self):
        pyxel.run(self.update, self.draw)
