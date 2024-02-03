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

The generated annotated graph displays the most recent 50 data streams.
Anomalies, represented by red markers, are based on the simulated data.
Annotations (Starting with Detected ) provide insights into the specific algorithms predicting anomaly.
The upper-right corner box presents live statistics, including F1 score, precision, and recall metrics, for the different algorithms used.
Note: the Y axis autoscales


For the assignment following three algorithms are used :

1. Moving Average (MA):

  Establishes a baseline understanding of the overall trend and seasonality in the data. This is crucial for identifying anomalies as deviations from what's considered "normal".
  It's a simple and efficient algorithm that can be easily implemented for real-time analysis of data streams.

2. Exponential Moving Average (EMA):
  Reacts quicker to recent changes compared to the MA, making it suitable for capturing concept drift where the underlying patterns shift over time.
  This is important for adapting to evolving data streams and ensuring that anomaly detection remains effective.

3. Isolation Forest:
  Excels at identifying various types of anomalies, regardless of their distribution or location in the data. This is because it isolates anomalies by finding data points that are significantly different from their neighbors.
  This is especially useful for detecting anomalies that deviate from the established trends or seasonality captured by the MA and EMA.

