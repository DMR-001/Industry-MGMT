# Energy Consumption Analysis & Prediction System

## CS5593 Project - Energy Efficiency Optimization in Industrial Settings

This project implements a comprehensive energy consumption analysis system with three main components:

1. **Linear Regression** - Energy consumption forecasting
2. **K-Means Clustering** - Operating modes detection
3. **Outlier Detection** - Anomaly detection using Z-score and IQR methods

## Features

### 📈 **ENERGY FORECASTING:**
- Custom linear regression implementation from scratch
- Gradient descent optimization with adjustable parameters
- Time series prediction with lag features and rolling averages
- Performance metrics: RMSE, MAE, R²
- **Enhanced Future Predictions:** 1-30 days ahead with intelligent feature updating
- **Session State Management:** Persistent model storage for interactive predictions
- **Prediction Bounds:** Automatic outlier prevention and reasonable value constraints
- **Interactive Controls:** Real-time parameter adjustment without retraining

### 🎯 Operating Modes Analysis
- K-means clustering algorithm implemented from scratch
- Silhouette score evaluation
- Cluster visualization and statistics
- Operating mode identification

### 🚨 Anomaly Detection
- Z-score method for outlier detection
- Interquartile Range (IQR) method
- Modified Z-score using median
- Ensemble detection methods
- Interactive threshold adjustment

## Project Structure

```
Eng/
├── app.py                    # Main Streamlit application
├── data/
│   ├── energy_data.csv        # Generated energy consumption data
│   └── industrial_energy_data.csv
├── src/
│   └── data_processor.py      # Data preprocessing utilities
├── algorithms/
│   ├── linear_regression.py   # Linear regression from scratch
│   ├── kmeans.py             # K-means clustering from scratch
│   └── outlier_detector.py   # Outlier detection methods
├── utils/
│   ├── data_generator.py     # Synthetic data generation
│   └── visualizer.py         # Plotting and visualization
└── requirements.txt       # Python dependencies
```

## Installation

1. **Clone or download the project**
```bash
cd "c:\Users\dorankulamukteshwara\Desktop\gam\Eng"
```

2. **Create virtual environment (already created)**
```bash
python -m venv .venv
.venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

```bash
streamlit run app.py
```

The application will start on `http://localhost:8501`

### Application Pages

#### 1. Data Overview
- View dataset statistics and sample data
- Visualize data distribution and time series
- Check data quality metrics

#### 2. Energy Forecasting
- Train linear regression model with custom parameters
- Adjust learning rate and training epochs
- View model performance metrics
- Generate future predictions

#### 3. Operating Modes Analysis
- Perform K-means clustering
- Adjust number of clusters and iterations
- Visualize cluster results and statistics
- Identify different operating patterns

#### 4. Anomaly Detection
- Choose detection method (Z-score or IQR)
- Adjust detection thresholds
- Visualize anomalies in time series
- Review anomaly statistics

## Algorithms Implemented

### Linear Regression
- **Gradient Descent**: Custom implementation with configurable learning rate
- **Feature Engineering**: Time-based features, lag features, rolling averages
- **Metrics**: RMSE, MAE, R² score calculation
- **Prediction**: Multi-step ahead forecasting

### K-Means Clustering
- **Initialization**: Random centroid initialization
- **Assignment**: Distance-based cluster assignment
- **Update**: Centroid recalculation
- **Evaluation**: Silhouette score implementation
- **Convergence**: Automatic convergence detection

### Outlier Detection
- **Z-Score**: Statistical outlier detection using standard deviation
- **IQR Method**: Quartile-based outlier detection
- **Modified Z-Score**: Median-based robust detection
- **Ensemble**: Multi-method voting system

## Data Generation

The system generates realistic synthetic energy consumption data including:
- **Seasonal patterns**: Higher usage in extreme temperatures
- **Weekly patterns**: Weekday vs weekend differences
- **Environmental factors**: Temperature and humidity effects
- **Operating modes**: Normal, high production, maintenance, weekend
- **Random outliers**: Simulated equipment failures or unusual events

## Performance Metrics

### Regression Metrics
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of determination

### Clustering Metrics
- **Silhouette Score**: Cluster quality measurement
- **Inertia**: Within-cluster sum of squares
- **Cluster Statistics**: Size, centroid, variance

### Anomaly Detection Metrics
- **Detection Rate**: Percentage of data flagged as anomalous
- **Threshold Sensitivity**: Adjustable detection parameters
- **Statistical Comparison**: Normal vs anomalous data statistics

## Team Members

- **Anvitha Reddy Thummalapally** - Regression Model Development
- **Lakshmi Sahasra Jangoan** - Clustering Analysis & Dashboard Integration
- **Soumya Kulkarni** - Outlier Detection & System Optimization

## Implementation Notes

- **No External ML Libraries**: All algorithms implemented from scratch using only NumPy and Pandas
- **Interactive Dashboard**: Full Streamlit integration with real-time parameter adjustment
- **Comprehensive Visualization**: Plotly-based interactive charts and graphs
- **Modular Design**: Clean separation of concerns with dedicated modules
- **Error Handling**: Robust error handling and user feedback

## Future Enhancements

1. **Advanced Models**: Implement polynomial regression and neural networks
2. **Real-time Data**: Integration with live energy monitoring systems
3. **Automated Reporting**: Scheduled analysis and report generation
4. **Model Optimization**: Hyperparameter tuning and cross-validation
5. **Export Features**: Model saving and prediction export capabilities

## License

This project is developed for academic purposes as part of CS5593 coursework.
