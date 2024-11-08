
# Stock Returns Anomaly Detection App

We will have built the web application with Streamlit to detect anomalies in stock returns data. The app uses the Isolation Forest algorithm to classify input data as "Normal" or "Anomaly." The goal is to help identify outliers in stock returns data, which may signify unusual market behavior or potential issues with the data.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Screenshots](#screenshots)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## Features

- Allows users to input a stock "Returns" value and identifies if it is an anomaly.
- Calculates historical daily returns using stock data.
- Visualizes anomalies in historical data and highlights user input for better insight.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/anomaly-detection-app.git
   cd anomaly-detection-app
   ```

2. **Set up a virtual environment:**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows use: env\Scripts\activate
   ```

3. **Install required libraries:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Place the required files:**
   - Ensure you have `Google Dataset.xlsx` containing the historical stock prices with a column labeled `Close`.
   - A pre-trained `model.pkl` file for Isolation Forest model.

## Usage

1. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

2. **Input a "Returns" value** in the app, and it will identify whether it is an anomaly.

3. **View the anomaly visualization** based on historical data, with anomalies highlighted in red.

## How It Works

- The app calculates daily returns from the stock data and trains an Isolation Forest model.
- The model identifies anomalies based on the spread and clustering of return values.
- Users can input a single "Returns" value, which is classified as "Anomaly" or "Normal" based on model predictions.
- The app provides a scatter plot to visualize historical returns with identified anomalies.

## Screenshots

### Normal Class:

![image](https://github.com/user-attachments/assets/d4800849-6b18-4c99-893f-6ace3c476675)

### Anomality Class:

![image](https://github.com/user-attachments/assets/2ca16071-171e-4dd6-b734-e04f1eedc152)



## Technologies Used

- **Streamlit**: Web framework for creating interactive applications.
- **Pandas**: Data manipulation and analysis.
- **Scikit-Learn**: Machine learning library for the Isolation Forest algorithm.
- **Matplotlib & Seaborn**: Data visualization libraries.


