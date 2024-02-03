import random
import time
import math

from matplotlib import pyplot as plt
from matplotlib import animation

# class RegrMagic(object):
#     """Mock for function Regr_magic()
#     """
#     def __init__(self):
#         self.x = 0

#     def __call__(self):
    
#         self.x += 1
#         return self.x, random.random()

# regr_magic = RegrMagic()

def simulate_data_stream(num_iterations):
    concept_drift_rate = 0.1
    concept_drift_value = 0
    current_time = 0

    for _ in range(num_iterations):
        seasonal_component = 10 * math.sin(2 * math.pi * current_time / 60)
        concept_drift_value += concept_drift_rate
        drift_component = max(2,5 * concept_drift_value)
        random_noise = random.uniform(-5, 5)

        if random.random() < 0.0:
            value = random.uniform(1500, 2000) + seasonal_component + drift_component + random_noise
            is_anomaly = True
        else:
            value = 50 + seasonal_component + drift_component + random_noise
            is_anomaly = False

        yield current_time , value
        current_time += 1

def frames():
    for time, val in  simulate_data_stream(50000):
    
        yield time,val
      
        # yield regr_magic()

fig, ax = plt.subplots()
line, = ax.plot([], [], color='g')

x = []
y = []

def animate(args):

    

    global x, y


    x.append(args[0])
    y.append(args[1])

    # Keep only the last 40 points
    # if len(x) > 50:
    #     x = x[-50:]
    #     y = y[-50:]

    line.set_data(x, y)
    ax.relim()
    ax.autoscale_view()
    return line,

anim = animation.FuncAnimation(fig, animate, frames=frames, interval=100)
plt.show()

