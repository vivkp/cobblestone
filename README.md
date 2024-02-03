Steps to run

# Set up a virtual environment and clone the repository
```bash
python3 -m venv env
cd env
source bin/activate
git clone https://github.com/vivkp/cobblestone.git
cd cobblestone

# Install dependencies and run the Python script with specified parameters
pip install -r requirements.txt

python plot.py ## default --anomaly_rate 0.08 --interval 100 --iterations 1000
python plot.py --anomaly_rate 0.08 --interval 200 --iterations 500
```

For the assignment titled "Efficient Data Stream Anomaly Detection," three algorithms were chosen to address the dynamic nature of continuous data streams:

1. **Moving Average (MA):**
   - **Purpose:** Establish a baseline understanding of overall trend and seasonality in the data, crucial for identifying anomalies.
   - **Advantages:** Simple and efficient, suitable for real-time analysis of data streams.

2. **Exponential Moving Average (EMA):**
   - **Purpose:** React quickly to recent changes, making it suitable for capturing concept drift and evolving patterns.
   - **Advantages:** Adapts to changing data streams, ensuring effective anomaly detection.

3. **Isolation Forest:**
   - **Purpose:** Identifies various types of anomalies regardless of their distribution or location.
   - **Advantages:** Isolates anomalies by finding data points significantly different from their neighbors, useful for deviations from established trends.

**Summary:**
- The MA provides a robust baseline, EMA adapts to concept drift, and Isolation Forest excels in detecting anomalies deviating from established trends or seasonality.

**Data Stream Simulation Details:**
- Incorporated regular patterns, seasonal elements, and random noise in the simulated data stream to mimic real-world scenarios and challenges.

**Real-time Anomaly Detection Mechanism:**
- Implemented a real-time mechanism that efficiently flags anomalies as data streams continuously, ensuring timely detection.

**Adaptability to Concept Drift:**
- EMA and Isolation Forest demonstrate adaptability to concept drift, crucial for handling shifts in underlying patterns over time.

**Efficiency and Optimization:**
- Employed optimization techniques to balance speed and efficiency, addressing the challenges posed by continuous data streams.

**Visualization Tool Features:**
- Developed a real-time visualization tool that aids in monitoring the data stream and visualizing detected anomalies, enhancing user understanding.
- The generated annotated graph displays the most recent 50 data streams.
- Anomalies, represented by red markers, are based on the real simulated anomaly.
- Annotations (Starting with Detected ) provide insights into the specific algorithms predicting anomaly.
- The upper-right corner box presents live statistics, including F1 score, precision, and recall metrics, for the different algorithms used.  
  Note: the Y axis autoscales

**Testing and Validation:**
- Conducted thorough testing and validation procedures, utilizing performance metrics to evaluate the accuracy and reliability of the anomaly detection system.
