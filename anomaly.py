import random
import math
import numpy as np
from collections import deque

class AnomalyDetector:
    def __init__(self, window_size, threshold_multiplier):
        self.window_size = window_size
        self.threshold_multiplier = threshold_multiplier
        self.data_window = deque(maxlen=window_size)
        self.mva_sum = 0
        self.mva_value = None
        self.std_dev = 0

    def update_mva(self, value):
        if len(self.data_window) == self.window_size:
            self.mva_sum -= self.data_window.popleft()

        self.mva_sum += value
        self.data_window.append(value)

        self.mva_value = self.mva_sum / len(self.data_window)
        self.std_dev = np.std(list(self.data_window))

    def detect_anomaly(self, value):
        threshold_mva = self.mva_value + self.threshold_multiplier * self.std_dev
        return value > threshold_mva

def simulate_data_stream(num_iterations):
    concept_drift_rate = 0.1
    concept_drift_value = 0
    current_time = 0

    for _ in range(num_iterations):
        seasonal_component = 10 * math.sin(2 * math.pi * current_time / 60)
        concept_drift_value += concept_drift_rate
        drift_component = 5 * concept_drift_value
        random_noise = random.uniform(-5, 5)

        if random.random() < 0.05:
            value = random.uniform(20000, 30000) + seasonal_component + drift_component + random_noise
            is_anomaly = True
        else:
            value = 50 + seasonal_component + drift_component + random_noise
            is_anomaly = False

        yield value, is_anomaly
        current_time += 1

def main():
    detector = AnomalyDetector(
        window_size=50, threshold_multiplier=3.0,
    )

    true_positives_mva = 0
    false_positives_mva = 0
    false_negatives_mva = 0

    num_iterations = 20000

    for value, is_anomaly_actual in simulate_data_stream(num_iterations):
        detector.update_mva(value)
  
        if is_anomaly_actual:
            if detector.detect_anomaly(value):
                true_positives_mva += 1
            else:
                false_negatives_mva += 1
        else:
            if detector.detect_anomaly(value):
                false_positives_mva += 1

    precision_mva = true_positives_mva / (true_positives_mva + false_positives_mva) if true_positives_mva + false_positives_mva > 0 else 0
    recall_mva = true_positives_mva / (true_positives_mva + false_negatives_mva) if true_positives_mva + false_negatives_mva > 0 else 0
    f1_mva = 2 * (precision_mva * recall_mva) / (precision_mva + recall_mva) if precision_mva + recall_mva > 0 else 0

    print(f"Precision (Moving Average): {precision_mva}")
    print(f"Recall (Moving Average): {recall_mva}")
    print(f"F1 Score (Moving Average): {f1_mva}")

if __name__ == "__main__":
    main()
