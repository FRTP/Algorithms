import pyglet
from pyglet.gl import *

from gym.envs.classic_control.rendering import Geom

RAD2DEG = 57.29577951308232


class Text(Geom):
    def __init__(self, start=(0.0, 0.0), end=(0.0, 0.0)):
        Geom.__init__(self)
        self.start = start
        self.end = end
        self.linewidth = LineWidth(1)
        self.add_attr(self.linewidth)

    def render1(self):
        text = pyglet.text.Label(
            'Hello',
            font_name='Action Man',
            font_size=36,
            x=200,
            y=200,
            anchor_x='center', anchor_y='center'
        )
        glBegin(GL_LINES)
        glVertex2f(*self.start)
        glVertex2f(*self.end)
        glEnd()
