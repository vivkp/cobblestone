import random
import time

from matplotlib import pyplot as plt
from matplotlib import animation


class RegrMagic(object):
    """Mock for function Regr_magic()
    """
    def __init__(self):
        self.x = 0
    def __call__(self):
        self.x += 1
        return self.x, random.random()

regr_magic = RegrMagic()

def frames():
    while True:
        yield regr_magic()

fig = plt.figure()

x = []
y = []
def animate(args):
    x.append(args[0])
    y.append(args[1])
    if len(x) > 40:
        x.pop(0)
        y.pop(0)
    return plt.plot(x, y, color='r')


anim = animation.FuncAnimation(fig, animate, frames=frames, interval=500)
plt.show()
