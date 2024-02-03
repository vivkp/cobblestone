import random
import math
from collections import deque
from matplotlib import pyplot as plt
from matplotlib import animation
from statsmodels.tsa.seasonal import STL

class AnomalyDetector:
    def __init__(self, window_size, threshold_multiplier, stl_seasonal):
        self.window_size = window_size
        self.threshold_multiplier = threshold_multiplier
        self.data_window = deque(maxlen=window_size)
        self.stl_seasonal = stl_seasonal

    def update(self, value):
        self.data_window.append(value)


    def detect_anomaly(self, value):
        _, _, residuals = self.stl_decomposition()
        residual_std = residuals.std()
        threshold = residual_std * self.threshold_multiplier

        return value > threshold

def simulate_data_stream(num_iterations):
    concept_drift_rate = 0.1
    concept_drift_value = 0
    current_time = 0

    for _ in range(num_iterations):
        seasonal_component = 10 * math.sin(2 * math.pi * current_time / 60)
        concept_drift_value += concept_drift_rate
        drift_component = max(5, 5 * concept_drift_value)
        random_noise = random.uniform(-5, 5)

        if random.random() < 0.05:
            value = random.uniform(500, 800) + seasonal_component + drift_component + random_noise
            is_anomaly = True
        else:
            value = 50 + seasonal_component + drift_component + random_noise
            is_anomaly = False

        yield current_time, value, is_anomaly
        current_time += 1

def frames():
    for time, val, is_anomaly in simulate_data_stream(500):
        yield time, val, is_anomaly


fig, ax = plt.subplots(figsize=(12, 6))
line, = ax.plot([], [], color='g')

detector = AnomalyDetector(window_size=50, threshold_multiplier=3.0, stl_seasonal=12)
x = []
y = []

true_positives = 0
false_positives = 0
false_negatives = 0

def animate(args):
    global x, y, true_positives, false_negatives, false_positives

    time_val, value, is_anomaly = args
    x.append(time_val)
    y.append(value)

    plt.xlim(max(0, len(x) - 50), len(x))
    line.set_data(x, y)

    detector.update(value)
    anomaly_detected = detector.detect_anomaly(value)

    if is_anomaly:
        if anomaly_detected:
            true_positives += 1
        else:
            false_negatives += 1
    else:
        if anomaly_detected:
            false_positives += 1

    if is_anomaly:
        plt.scatter(time_val, value, color='red', marker='o', label='Anomaly' if not plt.gca().get_legend() else "")
        if anomaly_detected:
            plt.annotate('Detected', (time_val, value), textcoords="offset points", xytext=(0, 10), ha='center',
                         fontsize=8, color='red')

    ax.relim()
    ax.autoscale_view()

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    plt.legend(['Data', f'F1 Score: {f1:.3f}'], loc='upper right')
    return line,

anim = animation.FuncAnimation(fig, animate, frames=frames, interval=1, repeat=False)

plt.show()


