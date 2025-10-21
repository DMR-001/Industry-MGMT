import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'algorithms'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from data_processor import DataProcessor
from linear_regression import LinearRegression
from kmeans import KMeans
from outlier_detector import OutlierDetector
from visualizer import Visualizer
from data_generator import generate_energy_data

# Page configuration
st.set_page_config(
    page_title="Energy Consumption Analysis",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("🏭 Steel Industry Energy Analysis & Prediction System")
    st.markdown("### Advanced Analytics for Industrial Energy Consumption")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("🏭 Steel Industry Analytics")
    page = st.sidebar.radio(
        "Select Analysis Type:",
        ["📊 Data Overview", "📈 Energy Forecasting", "🎯 Operating Modes", "🚨 Anomaly Detection"]
    )
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📋 Supported Features:**")
    st.sidebar.markdown("• Energy consumption forecasting")
    st.sidebar.markdown("• Power factor analysis") 
    st.sidebar.markdown("• CO2 emissions tracking")
    st.sidebar.markdown("• Load type classification")
    st.sidebar.markdown("• Operating mode detection")
    st.sidebar.markdown("• Anomaly identification")
    
    # Initialize components
    @st.cache_data
    def load_data():
        processor = DataProcessor()
        
        # Try to load user's CSV data first
        possible_paths = [
            "data/steel_industry_data.csv",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    if path.endswith('.xlsx'):
                        # Load Excel file
                        df = pd.read_excel(path)
                        df.to_csv("data/temp_data.csv", index=False)
                        return processor.load_data("data/temp_data.csv")
                    else:
                        return processor.load_data(path)
                except Exception as e:
                    st.warning(f"Could not load {path}: {str(e)}")
                    continue
        
        # Fallback to generated sample data
        data_path = "data/energy_data.csv"
        if not os.path.exists(data_path):
            st.info("Generating sample energy consumption data...")
            generate_energy_data(data_path)
        
        return processor.load_data(data_path)
    
    try:
        df = load_data()
        processor = DataProcessor()
        visualizer = Visualizer()
        
        if page == "📊 Data Overview":
            show_data_overview(df, processor, visualizer)
        elif page == "📈 Energy Forecasting":
            show_forecasting(df, processor, visualizer)
        elif page == "🎯 Operating Modes":
            show_clustering(df, processor, visualizer)
        elif page == "🚨 Anomaly Detection":
            show_anomaly_detection(df, processor, visualizer)
            
    except Exception as e:
        st.error(f"Error loading application: {str(e)}")
        st.info("Please check if all required files are present in the project directory.")

def show_data_overview(df, processor, visualizer):
    st.header("📊 Data Overview")
    
    # File upload section
    st.subheader("📁 Upload Your Data")
    uploaded_file = st.file_uploader(
        "Upload your industrial energy data (Excel or CSV)", 
        type=['xlsx', 'csv'],
        help="Upload your Excel or CSV file with energy consumption data"
    )
    
    if uploaded_file is not None:
        try:
            # Save uploaded file
            if uploaded_file.name.endswith('.xlsx'):
                df_new = pd.read_excel(uploaded_file)
                df_new.to_csv("data/uploaded_data.csv", index=False)
            else:
                df_new = pd.read_csv(uploaded_file)
                df_new.to_csv("data/uploaded_data.csv", index=False)
            
            st.success(f"✅ Successfully uploaded {uploaded_file.name}")
            st.info("Please refresh the page to use the new data")
            
            # Show preview of uploaded data
            st.subheader("Data Preview")
            st.dataframe(df_new.head())
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    # Data metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        if 'Usage_kWh' in df.columns:
            st.metric("Avg Energy Usage", f"{df['Usage_kWh'].mean():.2f} kWh")
        else:
            st.metric("Columns", len(df.columns))
    with col3:
        if 'Usage_kWh' in df.columns:
            st.metric("Max Usage", f"{df['Usage_kWh'].max():.2f} kWh")
        else:
            st.metric("Data Types", len(df.dtypes.unique()))
    with col4:
        if 'Date' in df.columns:
            date_range = (df['Date'].max() - df['Date'].min()).days
            st.metric("Date Range", f"{date_range} days")
        else:
            st.metric("Missing Values", f"{df.isnull().sum().sum()}")
    
    # Hide detailed dataset information - just show basic stats
    # Data info section removed to clean up UI
    
    # Sample data section removed to clean up UI
    
    # Visualizations
    st.subheader("Data Distribution")
    fig = visualizer.plot_distribution(df['Usage_kWh'])
    st.plotly_chart(fig, use_container_width=True)
    
    if 'Date' in df.columns:
        st.subheader("Time Series")
        fig = visualizer.plot_time_series(df)
        st.plotly_chart(fig, use_container_width=True)
    
    # Steel Industry specific visualizations
    st.subheader("🏭 Steel Industry Analytics")
    
    # Check if we have steel industry data
    steel_columns = ['Power_Factor', 'CO2_tCO2', 'Reactive_Power_kVarh', 'Load_Type', 'Week_Status']
    available_steel_cols = [col for col in steel_columns if col in df.columns]
    
    if len(available_steel_cols) == 0:
        st.info("No steel industry specific columns detected. This might be synthetic data.")
        return
    
    # Power Factor Analysis
    if 'Power_Factor' in df.columns:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Lagging Power Factor")
            fig = visualizer.plot_distribution(df['Power_Factor'])
            st.plotly_chart(fig, use_container_width=True)
        
        if 'Leading_Power_Factor' in df.columns:
            with col2:
                st.subheader("Leading Power Factor")
                fig = visualizer.plot_distribution(df['Leading_Power_Factor'])
                st.plotly_chart(fig, use_container_width=True)
        elif 'NSM' in df.columns:
            with col2:
                st.subheader("NSM Values")
                fig = visualizer.plot_distribution(df['NSM'])
                st.plotly_chart(fig, use_container_width=True)
    
    # CO2 Emissions Analysis
    if 'CO2_tCO2' in df.columns and 'Usage_kWh' in df.columns:
        st.subheader("CO2 Emissions Analysis")
        col1, col2 = st.columns(2)
        with col1:
            fig = visualizer.plot_distribution(df['CO2_tCO2'])
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            # Show CO2 vs Energy relationship
            import plotly.express as px
            fig = px.scatter(df, x='Usage_kWh', y='CO2_tCO2', 
                           title='Energy vs CO2 Emissions',
                           labels={'Usage_kWh': 'Energy Usage (kWh)', 'CO2_tCO2': 'CO2 (tCO2)'},
                           opacity=0.6)
            st.plotly_chart(fig, use_container_width=True)
    
    # Reactive Power Analysis
    reactive_cols = ['Reactive_Power_kVarh', 'Leading_Reactive_Power_kVarh']
    available_reactive = [col for col in reactive_cols if col in df.columns]
    
    if available_reactive:
        st.subheader("Reactive Power Analysis")
        if len(available_reactive) == 2:
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Lagging Reactive Power**")
                fig = visualizer.plot_distribution(df['Reactive_Power_kVarh'])
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.write("**Leading Reactive Power**")
                fig = visualizer.plot_distribution(df['Leading_Reactive_Power_kVarh'])
                st.plotly_chart(fig, use_container_width=True)
        else:
            fig = visualizer.plot_distribution(df[available_reactive[0]])
            st.plotly_chart(fig, use_container_width=True)
    
    # Load Type Distribution
    if 'Load_Type' in df.columns:
        st.subheader("Load Type Distribution")
        load_counts = df['Load_Type'].value_counts()
        import plotly.express as px
        fig = px.pie(values=load_counts.values, names=load_counts.index,
                    title='Distribution of Load Types',
                    color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show load type statistics
        st.subheader("Load Type Statistics")
        load_stats = df.groupby('Load_Type')['Usage_kWh'].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
        st.dataframe(load_stats)
    
    # Week Status Analysis
    if 'Week_Status' in df.columns and 'Usage_kWh' in df.columns:
        st.subheader("Weekday vs Weekend Analysis")
        weekend_stats = df.groupby('Week_Status')['Usage_kWh'].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
        st.dataframe(weekend_stats)
        
        # Visual comparison
        import plotly.express as px
        fig = px.box(df, x='Week_Status', y='Usage_kWh', 
                    title='Energy Usage: Weekday vs Weekend',
                    color='Week_Status')
        st.plotly_chart(fig, use_container_width=True)

def show_forecasting(df, processor, visualizer):
    st.header("📈 Energy Forecasting")
    
    # Prepare data for regression
    X, y = processor.prepare_regression_data(df)
    
    # Train-test split
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Model parameters
    col1, col2 = st.columns(2)
    with col1:
        learning_rate = st.slider("Learning Rate", 0.001, 0.1, 0.01)
    with col2:
        epochs = st.slider("Training Epochs", 100, 2000, 1000)
    
    # Initialize session state for model
    if 'trained_model' not in st.session_state:
        st.session_state.trained_model = None
        st.session_state.model_metrics = None
        st.session_state.X_test = None
        st.session_state.y_test = None
    
    if st.button("Train Regression Model"):
        with st.spinner("Training linear regression model..."):
            # Train model
            model = LinearRegression(learning_rate=learning_rate, epochs=epochs)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_rmse = model.rmse(y_train, y_pred_train)
            test_rmse = model.rmse(y_test, y_pred_test)
            train_mae = model.mae(y_train, y_pred_train)
            test_mae = model.mae(y_test, y_pred_test)
            
            # Store in session state
            st.session_state.trained_model = model
            st.session_state.model_metrics = {
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae
            }
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.y_pred_test = y_pred_test
            
            st.success("Model trained successfully!")
    
    # Display metrics if model is trained
    if st.session_state.trained_model is not None:
        metrics = st.session_state.model_metrics
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Train RMSE", f"{metrics['train_rmse']:.3f}")
        with col2:
            st.metric("Test RMSE", f"{metrics['test_rmse']:.3f}")
        with col3:
            st.metric("Train MAE", f"{metrics['train_mae']:.3f}")
        with col4:
            st.metric("Test MAE", f"{metrics['test_mae']:.3f}")
        
        # Plot results
        fig = visualizer.plot_regression_results(st.session_state.y_test, st.session_state.y_pred_test)
        st.plotly_chart(fig, use_container_width=True)
        
        # Future predictions section
        st.subheader("Future Predictions")
        days_ahead = st.slider("Days to predict ahead", 1, 30, 7, key="days_slider")
        
        # Generate predictions when slider changes
        if st.button("Generate Predictions") or days_ahead:
            with st.spinner(f"Generating {days_ahead} days of predictions..."):
                model = st.session_state.trained_model
                
                # Use the last few data points to start predictions
                last_features = X_test[-1:].copy()
                future_predictions = []
                
                for i in range(days_ahead):
                    pred = model.predict(last_features)[0]
                    future_predictions.append(pred)
                    
                    # Update features for next prediction more carefully
                    new_features = last_features.copy()
                    
                    # Feature order from data_processor.py:
                    # ['Temperature', 'Humidity', 'Day_of_Year', 'Day_of_Week', 
                    #  'Month', 'Quarter', 'Usage_lag1', 'Usage_lag2', 'Usage_lag3',
                    #  'Usage_rolling_3', 'Usage_rolling_7']
                    
                    if new_features.shape[1] >= 11:  # We have all features
                        # Shift lag features: lag3 = lag2, lag2 = lag1, lag1 = current pred
                        new_features[0, 8] = new_features[0, 7]  # Usage_lag3 = old Usage_lag2
                        new_features[0, 7] = new_features[0, 6]  # Usage_lag2 = old Usage_lag1
                        new_features[0, 6] = pred  # Usage_lag1 = new prediction
                        
                        # Update rolling averages more conservatively
                        # For rolling_3: use last 2 lag features + current prediction
                        rolling_3 = (new_features[0, 6] + new_features[0, 7] + new_features[0, 8]) / 3
                        new_features[0, 9] = rolling_3
                        
                        # For rolling_7: use more conservative update
                        old_rolling_7 = new_features[0, 10]
                        new_features[0, 10] = (old_rolling_7 * 6 + pred) / 7
                        
                        # Keep other features relatively stable or update minimally
                        # Day_of_Year, Day_of_Week, Month, Quarter should change for future days
                        new_features[0, 2] = (new_features[0, 2] + 1) % 365  # Day_of_Year
                        new_features[0, 3] = (new_features[0, 3] + 1) % 7     # Day_of_Week
                        
                        # Temperature and humidity - add small random variation
                        new_features[0, 0] += np.random.normal(0, 0.5)  # Temperature
                        new_features[0, 1] += np.random.normal(0, 1.0)  # Humidity
                    
                    last_features = new_features
                
                # Ensure predictions are reasonable (add bounds)
                future_predictions = np.array(future_predictions)
                
                # Apply reasonable bounds based on historical data
                historical_mean = np.mean(y)
                historical_std = np.std(y)
                lower_bound = max(0, historical_mean - 3 * historical_std)
                upper_bound = historical_mean + 3 * historical_std
                
                future_predictions = np.clip(future_predictions, lower_bound, upper_bound)
                
                future_df = pd.DataFrame({
                    'Day': range(1, days_ahead + 1),
                    'Predicted_Usage_kWh': future_predictions
                })
                
                # Display prediction results
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Days Predicted", days_ahead)
                with col2:
                    st.metric("Avg Predicted Usage", f"{np.mean(future_predictions):.2f} kWh")
                
                fig = visualizer.plot_future_predictions(future_df)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show prediction details
                with st.expander("View Detailed Predictions"):
                    st.dataframe(future_df)
    
    else:
        st.info("Please train a model first to see predictions and metrics.")

def show_clustering(df, processor, visualizer):
    st.header("🎯 Operating Modes Analysis")
    
    try:
        # Prepare clustering features
        features = processor.prepare_clustering_features(df)
        
        # Verify feature dimensions
        st.info(f"Dataset shape: {df.shape}, Features shape: {features.shape}")
        
        # Clustering parameters
        col1, col2 = st.columns(2)
        with col1:
            n_clusters = st.slider("Number of Clusters", 2, 10, 3)
        with col2:
            max_iters = st.slider("Max Iterations", 50, 500, 100)
        
        if st.button("Perform Clustering"):
            with st.spinner("Performing K-means clustering..."):
                # Perform clustering
                kmeans = KMeans(n_clusters=n_clusters, max_iters=max_iters)
                clusters = kmeans.fit_predict(features)
                
                # Verify cluster dimensions
                if len(clusters) != len(df):
                    st.error(f"Cluster length mismatch: {len(clusters)} clusters for {len(df)} data points")
                    return
                
                # Calculate silhouette score
                silhouette_score = kmeans.silhouette_score(features, clusters)
                
                # Display results
                st.metric("Silhouette Score", f"{silhouette_score:.3f}")
                
                # Add clusters to dataframe
                df_clustered = df.copy()
                df_clustered['Cluster'] = clusters
                
                # Plot clustering results
                fig = visualizer.plot_clustering_results(features, clusters, kmeans.centroids)
                st.plotly_chart(fig, use_container_width=True)
                
                # Cluster statistics
                st.subheader("Cluster Statistics")
                cluster_stats = df_clustered.groupby('Cluster')['Usage_kWh'].agg([
                    'count', 'mean', 'std', 'min', 'max'
                ]).round(2)
                st.dataframe(cluster_stats)
                
                # Time series by cluster
                fig = visualizer.plot_clusters_time_series(df_clustered)
                st.plotly_chart(fig, use_container_width=True)
                
    except Exception as e:
        st.error(f"Error in clustering analysis: {str(e)}")
        st.info("Please check the data format and try again.")

def show_anomaly_detection(df, processor, visualizer):
    st.header("🚨 Anomaly Detection")
    
    try:
        # Method selection
        method = st.selectbox(
            "Select Detection Method:",
            ["Z-Score", "IQR (Interquartile Range)"]
        )
        
        col1, col2 = st.columns(2)
        
        if method == "Z-Score":
            with col1:
                threshold = st.slider("Z-Score Threshold", 1.0, 4.0, 2.0, 0.1)
        else:
            with col1:
                multiplier = st.slider("IQR Multiplier", 1.0, 3.0, 1.5, 0.1)
        
        if st.button("Detect Anomalies"):
            with st.spinner("Detecting anomalies..."):
                detector = OutlierDetector()
                
                if method == "Z-Score":
                    anomalies = detector.detect_zscore(df['Usage_kWh'].values, threshold=threshold)
                    method_params = f"Threshold: {threshold}"
                else:
                    anomalies = detector.detect_iqr(df['Usage_kWh'].values, multiplier=multiplier)
                    method_params = f"IQR Multiplier: {multiplier}"
                
                # Verify anomaly array length
                if len(anomalies) != len(df):
                    st.error(f"Anomaly array length mismatch: {len(anomalies)} anomalies for {len(df)} data points")
                    return
                
                # Calculate statistics
                n_anomalies = np.sum(anomalies)
                anomaly_percentage = (n_anomalies / len(df)) * 100
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Anomalies", n_anomalies)
                with col2:
                    st.metric("Anomaly Rate", f"{anomaly_percentage:.2f}%")
                with col3:
                    st.metric("Method", method)
                
                st.info(f"Detection Parameters: {method_params}")
                
                # Add anomalies to dataframe
                df_anomalies = df.copy()
                df_anomalies['Is_Anomaly'] = anomalies
                
                # Plot results
                fig = visualizer.plot_anomalies(df_anomalies)
                st.plotly_chart(fig, use_container_width=True)
                
                # Anomaly details
                st.subheader("Anomaly Details")
                anomaly_data = df_anomalies[df_anomalies['Is_Anomaly']]
                
                if len(anomaly_data) > 0:
                    # Show available columns for anomalies
                    display_columns = ['Date', 'Usage_kWh']
                    
                    # Add industrial columns if available
                    industrial_cols = ['Power_Factor', 'CO2_tCO2', 'Reactive_Power_kVarh', 'Load_Type']
                    for col in industrial_cols:
                        if col in anomaly_data.columns:
                            display_columns.append(col)
                    
                    # Add Temperature and Humidity if they exist (they might be synthetic)
                    if 'Temperature' in anomaly_data.columns:
                        display_columns.append('Temperature')
                    if 'Humidity' in anomaly_data.columns:
                        display_columns.append('Humidity')
                    
                    st.dataframe(anomaly_data[display_columns])
                    
                    # Statistics of anomalies
                    st.subheader("Anomaly Statistics")
                    stats = {
                        'Mean Usage (Anomalies)': anomaly_data['Usage_kWh'].mean(),
                        'Mean Usage (Normal)': df_anomalies[~df_anomalies['Is_Anomaly']]['Usage_kWh'].mean(),
                        'Max Anomaly Value': anomaly_data['Usage_kWh'].max(),
                        'Min Anomaly Value': anomaly_data['Usage_kWh'].min()
                    }
                    
                    for key, value in stats.items():
                        st.metric(key, f"{value:.2f} kWh")
                else:
                    st.info("No anomalies detected with current parameters.")
                    
    except Exception as e:
        st.error(f"Error in anomaly detection: {str(e)}")
        st.info("Please check the data format and try again.")

if __name__ == "__main__":
    main()