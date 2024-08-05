import pyxel
from dataclasses import dataclass, field


@dataclass
class Player:
    x: float
    y: float
    w: float
    h: float
    surface_y: float  # TODO: Calculate whether airborne base on surface_y
    airborne: bool = field(default=False)
    jumping: bool = field(default=False)
    falling: bool = field(default=False)
    max_jump_height: int = 6
    current_jump_height: int = 0

    def jump(self):
        self.jumping = True

    def handle_jumping(self):
        self.jumping = self.current_jump_height < self.max_jump_height
        is_above_surface = self.y + self.h < self.surface_y
        if self.jumping:
            self.y -= 1
            self.current_jump_height += 1
        else:
            self.current_jump_height = 0
            self.jumping = False
            if is_above_surface:
                self.falling = True
                # TODO: Check if calling handle_falling() here has any benefits

    def handle_falling(self):
        if self.y + self.h == self.surface_y:
            self.falling = False
        else:
            assert (self.y + self.h < self.surface_y), "player not actually falling"
            self.y += 1

    def update(self):
        if self.jumping:
            self.handle_jumping()
        if self.falling:
            self.handle_falling()

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
