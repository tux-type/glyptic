import pyxel
from components import Player, Platform, Obstacle
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
        )
        self.offset = 0

        self.obstacle_locations = [10, 25, 30, 70]
        self.obstacles = deque()

    def render_background(self, col_a=7, col_b=13):
        col_a, col_b = (col_a, col_b) if self.offset % 2 == 0 else (col_b, col_a)
        [
            pyxel.rect(x=x, y=y, w=1, h=1, col=col_a)
            if (x % 2 == 0) != (y % 2 == 0)
            else pyxel.rect(x=x, y=y, w=1, h=1, col=col_b)
            for x in range(pyxel.width)
            for y in range(pyxel.height)
        ]

    def create_obstacle(self, immune=False):
        w = self.h // 15
        h = self.h // 15
        obstacle = Obstacle(x=self.w - w, y=self.base_y_position - h, w=w, h=h, immune=immune)
        self.obstacles.append(obstacle)

    def move_obstacles(self):
        assert len(self.obstacles) > 0, "trying to move obstacles when none present"
        for obstacle in self.obstacles:
            obstacle.x -= 1
        if self.obstacles[0].x < 0:
            self.destroy_obstacle()

    def destroy_obstacle(self):
        self.obstacles.popleft()

    def update(self):
        if pyxel.btnp(KEY_RIGHT, hold=0, repeat=1):
            if not self.obstacles:
                self.offset += 1
            else:
                print(f"player y: {self.player.y}")
                print(f"obstacle y: {self.obstacles[0].y}")
                print(self.player.y)
                # If collide with object game over?
                if (self.player.y + self.player.h) > self.obstacles[0].y:
                    self.offset += 1
                    self.move_obstacles()
                # Player collision
                elif (self.player.x + self.player.w) == self.obstacles[0].x:
                    print("Cannot move")
                    pass
                else:
                    self.offset += 1
                    self.move_obstacles()

                # if not self.obstacles[0].x == self.player.x + self.player.w:

        if pyxel.btn(KEY_UP):
            print("KEY_UP")
            self.player.jump()

        if self.offset in self.obstacle_locations:
            self.create_obstacle()

        self.player.update()

        if pyxel.btnp(pyxel.KEY_Q):
            pyxel.quit()

    def draw(self):
        pyxel.cls(0)
        self.render_background()
        self.base_platform.render(col=15)
        self.player.render()
        for obstacle in self.obstacles:
            obstacle.render()

    def run(self):
        pyxel.run(self.update, self.draw)
