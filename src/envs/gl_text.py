import os
import sys

from gym.envs.classic_control.rendering import Geom


if "Apple" in sys.version:
    if 'DYLD_FALLBACK_LIBRARY_PATH' in os.environ:
        os.environ['DYLD_FALLBACK_LIBRARY_PATH'] += ':/usr/lib'
        # (JDS 2016/04/15): avoid bug on Anaconda 2.3.0 / Yosemite

import pyglet

from pyglet.gl import *

RAD2DEG = 57.29577951308232

class Text(Geom):
    def __init__(self, start=(0.0, 0.0), end=(0.0, 0.0)):
        Geom.__init__(self)
        self.start = start
        self.end = end
        self.linewidth = LineWidth(1)
        self.add_attr(self.linewidth)

    def render1(self):
        text = pyglet.text.Label('Hello',
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
