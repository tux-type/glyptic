import pyxel
from dataclasses import dataclass, field


@dataclass
class Player:
    x: float
    y: float
    w: float
    h: float
    airborne: bool = field(default=False)
    airborne_time: int = field(default=0)
    airborne_limit: int = field(default=6)

    def jump(self):
        self.airborne = True

    def handle_airborne(self, jump_height=3):
        assert self.airborne_limit >= 2 * jump_height, "not enough time to perform full jump"
        assert (
            self.airborne_time <= self.airborne_limit
        ), "spending longer airborne than airborne limit"
        if self.airborne_time == self.airborne_limit:
            self.airborne = False
            self.airborne_time = 0
            return
        elif self.airborne_time < jump_height:
            self.y -= 1
        elif self.airborne_time >= (self.airborne_limit - jump_height):
            self.y += 1
        self.airborne_time += 1

    def update(self):
        if self.airborne:
            self.handle_airborne()

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
