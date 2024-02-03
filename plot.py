import random
import time
import math
import numpy as np
from collections import deque
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.widgets import Slider

class AnomalyDetector:
    def __init__(self, window_size, threshold_multiplier,ema_alpha):
        self.window_size = window_size
        self.threshold_multiplier = threshold_multiplier
        self.mva_data_window = deque(maxlen=window_size)
        self.mva_sum = 0
        self.mva_value = None
        self.mva_std_dev = 0

        self.ema_data_window = deque(maxlen=window_size)
        self.ema_alpha = ema_alpha
        self.ema_value = 0
        self.ema_std_dev = 0

    def update(self,value):
        self.update_mva(value)
        self.update_ema(value)

    def update_mva(self, value):
        if len(self.mva_data_window) == self.window_size:
            self.mva_sum -= self.mva_data_window.popleft()

        self.mva_sum += value
        self.mva_data_window.append(value)

        self.mva_value = self.mva_sum / len(self.mva_data_window)
        self.mva_std_dev = np.std(list(self.mva_data_window))

    def update_ema(self, value):
        if len(self.ema_data_window) == self.window_size:
            self.ema_data_window.popleft()

        self.ema_value = self.ema_alpha * value + (1 - self.ema_alpha) * self.ema_value
        self.ema_data_window.append(self.ema_value)
        
        self.ema_std_dev = np.std(list(self.ema_data_window))

    def detect_anomaly(self, value):
        mva_threshold = self.mva_value + self.threshold_multiplier * self.mva_std_dev
        ema_threshold = self.ema_value + self.threshold_multiplier * self.ema_std_dev
        return (value > mva_threshold , value > ema_threshold) 

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

detector = AnomalyDetector(window_size=50, threshold_multiplier=3.0,ema_alpha=0.2)
x = []
y = []

mva_true_positives= 0
mva_false_negatives = 0
mva_false_positives = 0

ema_true_positives= 0
ema_false_negatives = 0
ema_false_positives = 0

def animate(args):
    global x, y , ema_true_positives, ema_false_negatives, ema_false_positives , mva_true_positives , mva_false_negatives  , mva_false_positives 

    # , true_positives_mva, false_positives_mva , false_negatives_mva
    time_val, value, is_anomaly = args
    x.append(time_val)
    y.append(value)

    # if len(x) > 50:
    #     x = x[-50:]
    #     y = y[-50:]
    plt.xlim(max(0, len(x) - 50), len(x))

    line.set_data(x, y)

    detector.update(value)
    mva_threshold , ema_threshold = detector.detect_anomaly(value)

    if is_anomaly:
        if mva_threshold:
            mva_true_positives += 1
        else:
            mva_false_negatives += 1
        
        if ema_threshold:
            ema_true_positives += 1
        else:
            ema_false_negatives += 1
    else:
        if mva_threshold:
            mva_false_positives += 1

        if ema_threshold:
            ema_false_positives += 1

    


    if is_anomaly:
        plt.scatter(time_val, value, color='red', marker='o', label='Anomaly' if not plt.gca().get_legend() else "")
        
        anotate = "Detected"
        anyTrue = False
        if mva_threshold:
            anotate += " MVA"
            anyTrue = True
        if ema_threshold:
            anotate += " EMA"
            anyTrue = True
        if anyTrue :
            plt.annotate(anotate, (time_val, value), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='red')

    ax.relim()
    ax.autoscale_view()
    

    mva_precision = mva_true_positives / (mva_true_positives + mva_false_positives) if mva_true_positives + mva_false_positives > 0 else 0
    mva_recall = mva_true_positives / (mva_true_positives + mva_false_negatives) if mva_true_positives + mva_false_negatives > 0 else 0
    mva_f1 = 2 * (mva_precision * mva_recall) / (mva_precision + mva_recall) if mva_precision + mva_recall > 0 else 0


    ema_precision = ema_true_positives / (ema_true_positives + ema_false_positives) if ema_true_positives + ema_false_positives > 0 else 0
    ema_recall = ema_true_positives / (ema_true_positives + ema_false_negatives) if ema_true_positives + ema_false_negatives > 0 else 0
    ema_f1 = 2 * (ema_precision * ema_recall) / (ema_precision + ema_recall) if ema_precision + ema_recall > 0 else 0

    plt.legend(['Data', f'F1 Score (MVA): {mva_f1:.3f},',f'F1 Score (EMA): {ema_f1:.3f}'], loc='upper right')
    # plt.xlim(0, max(x))
    # plt.ylim(min(y), max(y))
    return line,

anim = animation.FuncAnimation(fig, animate, frames=frames, interval=1, repeat = False)


plt.show()

mva_precision = mva_true_positives / (mva_true_positives + mva_false_positives) if mva_true_positives + mva_false_positives > 0 else 0
mva_recall = mva_true_positives / (mva_true_positives + mva_false_negatives) if mva_true_positives + mva_false_negatives > 0 else 0
mva_f1 = 2 * (mva_precision * mva_recall) / (mva_precision + mva_recall) if mva_precision + mva_recall > 0 else 0


ema_precision = ema_true_positives / (ema_true_positives + ema_false_positives) if ema_true_positives + ema_false_positives > 0 else 0
ema_recall = ema_true_positives / (ema_true_positives + ema_false_negatives) if ema_true_positives + ema_false_negatives > 0 else 0
ema_f1 = 2 * (ema_precision * ema_recall) / (ema_precision + ema_recall) if ema_precision + ema_recall > 0 else 0

print(f"Precision : MVA - {mva_precision}  EMA - {ema_precision}")
print(f"Recall : MVA - {mva_recall} EMA - {ema_recall}")
print(f"F1 Score : MVA - {mva_f1} EMA - {ema_f1}")
