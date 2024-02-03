import random
import time
import math
import numpy as np
from collections import deque
from matplotlib import pyplot as plt
from matplotlib import animation
from sklearn.ensemble import IsolationForest
import argparse


class AnomalyDetector:
    """Class for detecting anomalies using Moving Average (MVA), Exponential Moving Average (EMA), and Isolation Forest."""

    def __init__(self, window_size, threshold_multiplier,ema_alpha,contamination):
        """Initialize the AnomalyDetector.

        Args:
        - window_size (int): Size of the data window for MVA and EMA.
        - threshold_multiplier (float): Multiplier for threshold in anomaly detection.
        - ema_alpha (float): Alpha parameter for EMA.
        - contamination (float): Proportion of outliers in the data for Isolation Forest.
        """
        self.window_size = window_size
        self.threshold_multiplier = threshold_multiplier

        # Moving Average (MVA) parameters
        self.mva_data_window = deque(maxlen=window_size)
        self.mva_sum = 0
        self.mva_value = None
        self.mva_std_dev = 0

        # Exponential Moving Average (EMA) parameters
        self.ema_data_window = deque(maxlen=window_size)
        self.ema_alpha = ema_alpha
        self.ema_value = 0
        self.ema_std_dev = 0

        # Isolation Forest parameters
        self.window_size = window_size 
        self.contamination = contamination * window_size/100 # Adjust contamination for Isolation Forest wrt window size
        self.model = None
        self.data_window = []

    def update(self,value):
        """Update MVA, EMA, and Isolation Forest with the new data point."""
       
        self.update_mva(value)
        self.update_ema(value)
        self.update_model(value)

    def update_mva(self, value):
        """ Update Moving Average (MVA) parameters """
        if len(self.mva_data_window) == self.window_size:
            self.mva_sum -= self.mva_data_window.popleft()

        self.mva_sum += value
        self.mva_data_window.append(value)

        self.mva_value = self.mva_sum / len(self.mva_data_window)
        self.mva_std_dev = np.std(list(self.mva_data_window))

    def update_ema(self, value):
        """Update Exponential Moving Average (EMA) parameters."""
        if len(self.ema_data_window) == self.window_size:
            self.ema_data_window.popleft()

        self.ema_value = self.ema_alpha * value + (1 - self.ema_alpha) * self.ema_value
        self.ema_data_window.append(self.ema_value)
        
        self.ema_std_dev = np.std(list(self.ema_data_window))

    def update_model(self, value):
        """Update Isolation Forest model with the new data point."""

        self.data_window.append(value)

        if len(self.data_window) > self.window_size:
            self.data_window.pop(0)
            window = np.array(self.data_window).reshape(-1, 1)

            if self.model is None:
                self.model = IsolationForest(contamination=self.contamination)
                
            self.model.fit(window)

    def isof_anomaly(self, value):
        """Predict anomaly using Isolation Forest."""
        if self.model is None:
            return 0
        prediction = self.model.predict([[value]])
        return 1 if prediction == -1 else 0

    def detect_anomaly(self, value):
        """Detect anomalies using MVA, EMA, and Isolation Forest."""

        mva_threshold = self.mva_value + self.threshold_multiplier * self.mva_std_dev
        ema_threshold = self.ema_value + self.threshold_multiplier * self.ema_std_dev
        isof_anomaly = self.isof_anomaly(value)
        return (value > mva_threshold , value > ema_threshold , isof_anomaly) 



def simulate_data_stream(num_iterations,anomaly_rate):
    """Generate a simulated data stream with concept drift and seasonal variation."""
   
    concept_drift_rate = 0.1
    concept_drift_value = 0
    current_time = 0
    period = 60

    for i in range(num_iterations):
        value = 50 ### Constant Value 
        random_noise = random.uniform(-5, 5)   ### Random Noise
        seasonal_component = 10 * math.sin(2 * math.pi * current_time / period) ### Seasonal Component with period 60
        concept_drift_value += concept_drift_rate
        drift_component =  5 * concept_drift_value  ### incremental Concept Drift 
        
        

        if random.random() < anomaly_rate and i > 50: #### roduce anomaly depending on the anomaly rate probability , unless the window size is 50 it would not produce anomaly
            value = random.uniform(500, 800) + seasonal_component + drift_component + random_noise
            is_anomaly = True ### variable storing the generated point is anomaly or not
        else:
            value = value + seasonal_component + drift_component + random_noise
            is_anomaly = False

        yield current_time, value, is_anomaly 
        current_time += 1

def frames(iterations):
    """Generate frames for the animation based on the simulated data stream."""
   
    for time, val, is_anomaly in simulate_data_stream(iterations,anomaly_rate):
        yield time, val, is_anomaly



def animate(args):
    """Animate the data stream and update anomaly detection metrics."""
    
    global x, y , ema_true_positives, ema_false_negatives, ema_false_positives , mva_true_positives , mva_false_negatives  , mva_false_positives ,  isof_true_positives, isof_false_negatives, isof_false_positives

    time_val, value, is_anomaly = args
    x.append(time_val)
    y.append(value)

    #### Showing the latest 50 points on the plot
    plt.xlim(max(0, len(x) - 50), len(x))

    ### Padded y so that anomaly annotation is in the frame
    if max(y) > 500:
        plt.ylim(0,max(y)+200)
    

    line.set_data(x, y)
  

    detector.update(value)
    mva_anomaly , ema_anomaly , isof_anomaly = detector.detect_anomaly(value)

    if is_anomaly:
        if mva_anomaly:
            mva_true_positives += 1
        else:
            mva_false_negatives += 1
        
        if ema_anomaly:
            ema_true_positives += 1
        else:
            ema_false_negatives += 1

        if isof_anomaly:
            isof_true_positives += 1
        else:
            isof_false_negatives += 1
    else:
        if mva_anomaly:
            mva_false_positives += 1

        if ema_anomaly:
            ema_false_positives += 1
        
        if isof_anomaly:
            isof_false_positives +=1

    
    notDetect = True

    if is_anomaly: ### plotting the real generated anomaly on the graph 
        plt.scatter(time_val, value, color='red', marker='o', label='Anomaly' if not plt.gca().get_legend() else "")
        notDetect = False

    anotate = "Detected - "
    anyTrue = False
    if mva_anomaly:
        anotate += " MVA"
        anyTrue = True
    if ema_anomaly:
        anotate += " EMA"
        anyTrue = True

    if isof_anomaly:
        anotate += " ISOF"
        anyTrue = True

    if anyTrue :
        if notDetect: ### code to plot the predicted anomaly - False Positive
            plt.scatter(time_val, value, color='blue', marker='*', label='FalseAnomaly' if not plt.gca().get_legend() else "")

        loc = random.randint(5,25) 
        ### Listing the algorithm that detected the anomaly.
        plt.annotate(anotate, (time_val, value), textcoords="offset points", xytext=(0,loc), ha='center', fontsize=8, color='orange')

    ax.relim()
    ax.autoscale_view()
    

    mva_precision = mva_true_positives / (mva_true_positives + mva_false_positives) if mva_true_positives + mva_false_positives > 0 else 0
    mva_recall = mva_true_positives / (mva_true_positives + mva_false_negatives) if mva_true_positives + mva_false_negatives > 0 else 0
    mva_f1 = 2 * (mva_precision * mva_recall) / (mva_precision + mva_recall) if mva_precision + mva_recall > 0 else 0


    ema_precision = ema_true_positives / (ema_true_positives + ema_false_positives) if ema_true_positives + ema_false_positives > 0 else 0
    ema_recall = ema_true_positives / (ema_true_positives + ema_false_negatives) if ema_true_positives + ema_false_negatives > 0 else 0
    ema_f1 = 2 * (ema_precision * ema_recall) / (ema_precision + ema_recall) if ema_precision + ema_recall > 0 else 0

    isof_precision = isof_true_positives / (isof_true_positives + isof_false_positives) if isof_true_positives + isof_false_positives > 0 else 0
    isof_recall = isof_true_positives / (isof_true_positives + isof_false_negatives) if isof_true_positives + isof_false_negatives > 0 else 0
    isof_f1 = 2 * (isof_precision * isof_recall) / (isof_precision + isof_recall) if isof_precision + isof_recall > 0 else 0


    #### printing f1 score , precision , recall for mva , ema and isof respectively on the graph

    plt.legend(['Data', f'MVA  F1 score: {mva_f1:.3f} Pr: {mva_precision:.3f} Recall - {mva_recall:.3f} ,',f'EMA F1 score: {ema_f1:.3f} Pr: {ema_precision:.3f} Recall: {ema_recall:.3f}',f'ISOF F1 score: {isof_f1:.3f} Pr: {isof_precision:.3f} Recall: {isof_recall:.3f}'], loc='upper right') 
  
    return line,


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--anomaly_rate', dest='anomaly_rate',default= 0.08, type=float, help='Add anomaly rate')
    parser.add_argument('--interval', dest='interval', type=int,default = 100, help='Interval to show live data')
    parser.add_argument('--iterations', dest='iterations', type=int,default =1000, help='Number of time stream data called')
    
    args = parser.parse_args()
    anomaly_rate = args.anomaly_rate
    interval = args.interval
    iterations = args.iterations


    #### Data validation is performed to handle cases of incorrect user input; default values will be used if the provided input is invalid."
    if interval < 1 :
        interval = 100
    
    if iterations < 1:
        iterations = 500
    
    if anomaly_rate > 1 or anomaly_rate < 0 :
        anomaly_rate = 0.08


    fig, ax = plt.subplots(figsize=(12, 6))
    line, = ax.plot([], [], color='g')

    x = []
    y = []

    

    detector = AnomalyDetector(window_size=50, threshold_multiplier=3.0,ema_alpha=0.2,contamination = anomaly_rate)
    

    mva_true_positives= 0
    mva_false_negatives = 0
    mva_false_positives = 0

    ema_true_positives= 0
    ema_false_negatives = 0
    ema_false_positives = 0

    isof_true_positives= 0
    isof_false_negatives = 0
    isof_false_positives = 0

    plt.title("Anomaly Detection on Live Data Stream")
    plt.xlabel("Time")
    plt.ylabel("Values")
    anim = animation.FuncAnimation(fig, animate, frames=frames(iterations), interval=interval, repeat = False)
    plt.show()
    ##### The Red point on the graph shows the real generated anomaly. 
    #### The "Detected" followd by mva , ema or isof shows the real anomaly was successfully detected by respective algorithms