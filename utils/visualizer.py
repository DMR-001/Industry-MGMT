import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class Visualizer:
    """Visualization utilities for the energy consumption analysis"""
    
    def __init__(self):
        self.color_palette = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff9800',
            'info': '#17a2b8'
        }
    
    def plot_time_series(self, df, column='Usage_kWh'):
        """Plot time series data"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df[column],
            mode='lines',
            name=f'{column}',
            line=dict(color=self.color_palette['primary'], width=2)
        ))
        
        fig.update_layout(
            title=f'{column} Over Time',
            xaxis_title='Date',
            yaxis_title=column,
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def plot_distribution(self, data, title='Data Distribution'):
        """Plot histogram and box plot of data distribution"""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Histogram', 'Box Plot'),
            column_widths=[0.7, 0.3]
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(
                x=data,
                nbinsx=30,
                name='Distribution',
                marker_color=self.color_palette['primary'],
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Box plot
        fig.add_trace(
            go.Box(
                y=data,
                name='Box Plot',
                marker_color=self.color_palette['secondary']
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title=title,
            template='plotly_white',
            showlegend=False
        )
        
        return fig
    
    def plot_regression_results(self, y_true, y_pred):
        """Plot regression results"""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Actual vs Predicted', 'Residuals'),
            column_widths=[0.5, 0.5]
        )
        
        # Actual vs Predicted
        fig.add_trace(
            go.Scatter(
                x=y_true,
                y=y_pred,
                mode='markers',
                name='Predictions',
                marker=dict(color=self.color_palette['primary'], opacity=0.6)
            ),
            row=1, col=1
        )
        
        # Perfect prediction line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color=self.color_palette['danger'], dash='dash')
            ),
            row=1, col=1
        )
        
        # Residuals
        residuals = y_true - y_pred
        fig.add_trace(
            go.Scatter(
                x=y_pred,
                y=residuals,
                mode='markers',
                name='Residuals',
                marker=dict(color=self.color_palette['secondary'], opacity=0.6)
            ),
            row=1, col=2
        )
        
        # Zero line for residuals
        fig.add_trace(
            go.Scatter(
                x=[min(y_pred), max(y_pred)],
                y=[0, 0],
                mode='lines',
                name='Zero Line',
                line=dict(color=self.color_palette['danger'], dash='dash')
            ),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Actual Values", row=1, col=1)
        fig.update_yaxes(title_text="Predicted Values", row=1, col=1)
        fig.update_xaxes(title_text="Predicted Values", row=1, col=2)
        fig.update_yaxes(title_text="Residuals", row=1, col=2)
        
        fig.update_layout(
            title='Regression Model Performance',
            template='plotly_white',
            showlegend=False
        )
        
        return fig
    
    def plot_future_predictions(self, future_df):
        """Plot future predictions"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=future_df['Day'],
            y=future_df['Predicted_Usage_kWh'],
            mode='lines+markers',
            name='Future Predictions',
            line=dict(color=self.color_palette['success'], width=3),
            marker=dict(size=8, symbol='circle'),
            hovertemplate='<b>Day %{x}</b><br>Predicted Usage: %{y:.2f} kWh<extra></extra>'
        ))
        
        # Add confidence band (simple approach using standard deviation)
        std_dev = np.std(future_df['Predicted_Usage_kWh'])
        upper_bound = future_df['Predicted_Usage_kWh'] + std_dev
        lower_bound = future_df['Predicted_Usage_kWh'] - std_dev
        
        fig.add_trace(go.Scatter(
            x=future_df['Day'],
            y=upper_bound,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=future_df['Day'],
            y=lower_bound,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(46, 160, 44, 0.2)',
            name='Confidence Band',
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            title=f'Future Energy Usage Predictions ({len(future_df)} days)',
            xaxis_title='Days Ahead',
            yaxis_title='Predicted Usage (kWh)',
            template='plotly_white',
            hovermode='x unified'
        )
        
        # Add annotations for min and max predictions
        max_pred = future_df['Predicted_Usage_kWh'].max()
        min_pred = future_df['Predicted_Usage_kWh'].min()
        max_day = future_df.loc[future_df['Predicted_Usage_kWh'].idxmax(), 'Day']
        min_day = future_df.loc[future_df['Predicted_Usage_kWh'].idxmin(), 'Day']
        
        fig.add_annotation(
            x=max_day,
            y=max_pred,
            text=f"Max: {max_pred:.1f} kWh",
            showarrow=True,
            arrowhead=2,
            arrowcolor=self.color_palette['danger']
        )
        
        fig.add_annotation(
            x=min_day,
            y=min_pred,
            text=f"Min: {min_pred:.1f} kWh",
            showarrow=True,
            arrowhead=2,
            arrowcolor=self.color_palette['info']
        )
        
        return fig
    
    def plot_clustering_results(self, X, labels, centroids):
        """Plot clustering results (2D projection)"""
        # For visualization, use first two features if more than 2D
        if X.shape[1] > 2:
            X_plot = X[:, :2]
            centroids_plot = centroids[:, :2]
            feature_names = ['Feature 1', 'Feature 2']
        else:
            X_plot = X
            centroids_plot = centroids
            feature_names = ['Feature 1', 'Feature 2']
        
        fig = go.Figure()
        
        # Plot data points
        unique_labels = np.unique(labels)
        colors = px.colors.qualitative.Set1[:len(unique_labels)]
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            fig.add_trace(go.Scatter(
                x=X_plot[mask, 0],
                y=X_plot[mask, 1],
                mode='markers',
                name=f'Cluster {label}',
                marker=dict(color=colors[i], opacity=0.6, size=8)
            ))
        
        # Plot centroids
        fig.add_trace(go.Scatter(
            x=centroids_plot[:, 0],
            y=centroids_plot[:, 1],
            mode='markers',
            name='Centroids',
            marker=dict(
                color='black',
                size=15,
                symbol='x',
                line=dict(width=2, color='white')
            )
        ))
        
        fig.update_layout(
            title='K-Means Clustering Results',
            xaxis_title=feature_names[0],
            yaxis_title=feature_names[1],
            template='plotly_white'
        )
        
        return fig
    
    def plot_clusters_time_series(self, df_clustered):
        """Plot time series colored by clusters"""
        fig = go.Figure()
        
        unique_clusters = sorted(df_clustered['Cluster'].unique())
        colors = px.colors.qualitative.Set1[:len(unique_clusters)]
        
        for i, cluster in enumerate(unique_clusters):
            cluster_data = df_clustered[df_clustered['Cluster'] == cluster]
            fig.add_trace(go.Scatter(
                x=cluster_data['Date'],
                y=cluster_data['Usage_kWh'],
                mode='markers',
                name=f'Cluster {cluster}',
                marker=dict(color=colors[i], opacity=0.7, size=6)
            ))
        
        fig.update_layout(
            title='Energy Usage by Operating Mode (Cluster)',
            xaxis_title='Date',
            yaxis_title='Usage (kWh)',
            template='plotly_white'
        )
        
        return fig
    
    def plot_anomalies(self, df_anomalies):
        """Plot time series with anomalies highlighted"""
        fig = go.Figure()
        
        # Normal points
        normal_data = df_anomalies[~df_anomalies['Is_Anomaly']]
        fig.add_trace(go.Scatter(
            x=normal_data['Date'],
            y=normal_data['Usage_kWh'],
            mode='markers',
            name='Normal',
            marker=dict(color=self.color_palette['primary'], opacity=0.6, size=6)
        ))
        
        # Anomalous points
        anomaly_data = df_anomalies[df_anomalies['Is_Anomaly']]
        if len(anomaly_data) > 0:
            fig.add_trace(go.Scatter(
                x=anomaly_data['Date'],
                y=anomaly_data['Usage_kWh'],
                mode='markers',
                name='Anomalies',
                marker=dict(
                    color=self.color_palette['danger'],
                    size=10,
                    symbol='diamond',
                    line=dict(width=2, color='white')
                )
            ))
        
        fig.update_layout(
            title='Energy Usage with Anomaly Detection',
            xaxis_title='Date',
            yaxis_title='Usage (kWh)',
            template='plotly_white'
        )
        
        return fig
    
    def plot_correlation_matrix(self, df):
        """Plot correlation matrix heatmap"""
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Feature Correlation Matrix',
            template='plotly_white'
        )
        
        return fig
    
    def plot_feature_importance(self, features, importance_values):
        """Plot feature importance"""
        fig = go.Figure(go.Bar(
            x=importance_values,
            y=features,
            orientation='h',
            marker_color=self.color_palette['primary']
        ))
        
        fig.update_layout(
            title='Feature Importance',
            xaxis_title='Importance',
            yaxis_title='Features',
            template='plotly_white'
        )
        
        return fig