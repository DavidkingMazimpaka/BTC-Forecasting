# Forecasting Bitcoin Prices with Time Series Analysis

## Introduction

Time series forecasting plays a crucial role in financial markets, allowing investors and analysts to predict future price movements based on historical data. Given the volatile nature of Bitcoin, effective forecasting can provide valuable insights for trading strategies and investment decisions. This project explores the use of time series analysis to forecast Bitcoin prices, leveraging advanced machine learning techniques.

## 1. An Introduction to Time Series Forecasting

Time series forecasting is the process of predicting future values based on previously observed values. It is particularly significant in financial markets where understanding trends, seasonality, and cyclical patterns can lead to better decision-making.

Forecasting Bitcoin prices is valuable due to:
- **Market Volatility**: Bitcoin prices are highly volatile, making accurate predictions crucial for risk management.
- **Investment Strategies**: Forecasting can inform trading strategies, helping investors maximize returns.
- **Market Analysis**: Understanding price trends can assist in broader market analysis and sentiment evaluation.

## 2. Preprocessing Method

Preprocessing is a vital step in preparing data for modeling. In this project, the following preprocessing techniques were employed:

- **Normalization**: Scaling the data to a range between 0 and 1 to improve model performance.
- **Handling Missing Values**: Filling or interpolating missing data points to maintain the integrity of the time series.

### Code Snippet
```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load data
data = pd.read_csv('bitcoin_prices.csv')

# Handle missing values
data.fillna(method='ffill', inplace=True)

# Normalize data
scaler = MinMaxScaler()
data['Close'] = scaler.fit_transform(data[['Close']])
```
## 3. Setting Up tf.data.Dataset for Model Inputs

Using TensorFlow's tf.data.Dataset allows for efficient data loading and preprocessing. Key aspects include:

- **Batching:** Grouping samples into batches to optimize training.
- **Shuffling:** Randomizing the order of samples to prevent the model from learning unintended patterns.
- **Windowing:** Creating time windows to structure the dataset for sequential learning.

### Code Example
```python
import tensorflow as tf

# Create tf.data.Dataset
def create_dataset(data, time_step):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Setup dataset
X, y = create_dataset(data['Close'].values.reshape(-1, 1), time_step=30)
dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(32).shuffle(buffer_size=1000)
```

## 4. Model Architecture

For forecasting, an LSTM (Long Short-Term Memory) network was employed due to its effectiveness in capturing temporal dependencies in sequential data.

***Model Summary**
```python
model=Sequential()

model.add(LSTM(10,input_shape=(None,1),activation="relu"))

model.add(Dense(1))

model.compile(loss="mean_squared_error",optimizer="adam")
```
## 5. Results and Evaluation
The model's performance was evaluated using metrics such as Mean Absolute Error (MAE) and Root Mean Square Error (RMSE).

### Performance Metrics

- Train data RMSE:  1887.9166018845806
- Train data MSE:  3564229.0956714223
- Train data MAE:  1468.6586168509853
- Test data RMSE:  1831.431546424147
- Test data MSE:  3354141.5092375427
- Test data MAE:  1410.0935997884612

### Visualization
![BTC Price Predictions](data/btc%20forecasting.pngbtc forecasting.png)

In this section, the predicted Bitcoin prices are compared to the actual prices. The graph illustrates the model's effectiveness in capturing overall trends in Bitcoin's price movements. However, some discrepancies are evident, particularly during periods of high volatility, where the model's predictions may deviate from actual market behavior. These insights highlight both the strengths and limitations of the forecasting model.

## 6. Conclusion
Working on this forecasting task provided valuable insights into the complexities of predicting Bitcoin prices. The challenges included data preprocessing and tuning the model architecture for optimal performance. However, the potential for accurate forecasting in such a volatile market is significant.

For the complete code and further details, please visit my GitHub repository: [GitHub Repository Link](https://github.com/DavidkingMazimpaka/btc_forecasting).

## Examples and Visuals
Throughout this README, code snippets have been provided for clarity, along with visualizations to illustrate the model's performance and the importance of preprocessing in time series forecasting.