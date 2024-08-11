from dataclasses import dataclass, field

import pyxel


@dataclass
class Player:
    x: float
    y: float
    w: float
    h: float
    surface_y: float
    jumping: bool = field(default=False)
    falling: bool = field(default=False)
    max_jump_height: int = 3
    current_jump_height: int = 0

    def jump(self):
        is_on_surface = (self.y + self.h) == self.surface_y
        if is_on_surface:
            self.jumping = True

    def fall(self):
        self.falling = True

    def handle_jumping(self):
        still_jumping = self.current_jump_height < self.max_jump_height and not self.falling
        if still_jumping:
            self.y -= 1
            self.current_jump_height += 1
        else:
            self.current_jump_height = 0
            self.jumping = False

    def check_falling(self):
        is_above_surface = (self.y + self.h) < self.surface_y
        if is_above_surface and not self.jumping:
            self.falling = True

    def handle_falling(self):
        if self.y + self.h == self.surface_y:
            self.falling = False
        else:
            assert self.y + self.h < self.surface_y, "player not actually falling"
            self.y += 1

    def update(self):
        self.check_falling()
        if self.falling:
            self.handle_falling()
        if self.jumping:
            self.handle_jumping()

    def render(self, col=1):
        pyxel.rect(x=self.x, y=self.y, w=self.w, h=self.h, col=col)


@dataclass
class Platform:
    x: float
    y: float
    w: float
    h: float

    def render(self, col=13):
        pyxel.rect(x=self.x, y=self.y, w=self.w, h=self.h, col=col)


@dataclass
class Obstacle:
    x: float
    y: float
    w: float
    h: float
    immune: bool

    def render(self, col=0):
        pyxel.rect(x=self.x, y=self.y, w=self.w, h=self.h, col=col)
