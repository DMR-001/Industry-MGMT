import numpy as np

class OutlierDetector:
    """Outlier detection algorithms implemented from scratch"""
    
    def __init__(self):
        self.outliers_info = {}
        
    def detect_zscore(self, data, threshold=2.0):
        """Detect outliers using Z-score method"""
        data = np.array(data)
        
        # Calculate mean and standard deviation
        mean = np.mean(data)
        std = np.std(data)
        
        # Calculate Z-scores
        z_scores = np.abs((data - mean) / std)
        
        # Identify outliers
        outliers = z_scores > threshold
        
        # Store information
        self.outliers_info['zscore'] = {
            'mean': mean,
            'std': std,
            'threshold': threshold,
            'z_scores': z_scores,
            'outlier_indices': np.where(outliers)[0],
            'outlier_values': data[outliers],
            'n_outliers': np.sum(outliers)
        }
        
        return outliers
    
    def detect_iqr(self, data, multiplier=1.5):
        """Detect outliers using Interquartile Range (IQR) method"""
        data = np.array(data)
        
        # Calculate quartiles
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        
        # Calculate bounds
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        # Identify outliers
        outliers = (data < lower_bound) | (data > upper_bound)
        
        # Store information
        self.outliers_info['iqr'] = {
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'multiplier': multiplier,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'outlier_indices': np.where(outliers)[0],
            'outlier_values': data[outliers],
            'n_outliers': np.sum(outliers)
        }
        
        return outliers
    
    def detect_modified_zscore(self, data, threshold=3.5):
        """Detect outliers using Modified Z-score method (using median)"""
        data = np.array(data)
        
        # Calculate median and median absolute deviation
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        
        # Calculate modified Z-scores
        modified_z_scores = 0.6745 * (data - median) / mad if mad != 0 else np.zeros_like(data)
        
        # Identify outliers
        outliers = np.abs(modified_z_scores) > threshold
        
        # Store information
        self.outliers_info['modified_zscore'] = {
            'median': median,
            'mad': mad,
            'threshold': threshold,
            'modified_z_scores': modified_z_scores,
            'outlier_indices': np.where(outliers)[0],
            'outlier_values': data[outliers],
            'n_outliers': np.sum(outliers)
        }
        
        return outliers
    
    def detect_isolation_forest_simple(self, data, contamination=0.1):
        """Simple isolation-based outlier detection"""
        data = np.array(data)
        n_samples = len(data)
        n_outliers = int(contamination * n_samples)
        
        # Calculate distances from median
        median = np.median(data)
        distances = np.abs(data - median)
        
        # Get indices of points with largest distances
        outlier_indices = np.argsort(distances)[-n_outliers:]
        outliers = np.zeros(n_samples, dtype=bool)
        outliers[outlier_indices] = True
        
        # Store information
        self.outliers_info['isolation_simple'] = {
            'median': median,
            'contamination': contamination,
            'distances': distances,
            'outlier_indices': outlier_indices,
            'outlier_values': data[outliers],
            'n_outliers': np.sum(outliers)
        }
        
        return outliers
    
    def detect_ensemble(self, data, methods=['zscore', 'iqr'], min_votes=2):
        """Ensemble outlier detection using multiple methods"""
        data = np.array(data)
        n_samples = len(data)
        votes = np.zeros(n_samples)
        
        # Apply each method
        if 'zscore' in methods:
            outliers_zscore = self.detect_zscore(data)
            votes += outliers_zscore.astype(int)
        
        if 'iqr' in methods:
            outliers_iqr = self.detect_iqr(data)
            votes += outliers_iqr.astype(int)
        
        if 'modified_zscore' in methods:
            outliers_modified = self.detect_modified_zscore(data)
            votes += outliers_modified.astype(int)
        
        # Final decision based on minimum votes
        ensemble_outliers = votes >= min_votes
        
        # Store information
        self.outliers_info['ensemble'] = {
            'methods': methods,
            'min_votes': min_votes,
            'votes': votes,
            'outlier_indices': np.where(ensemble_outliers)[0],
            'outlier_values': data[ensemble_outliers],
            'n_outliers': np.sum(ensemble_outliers)
        }
        
        return ensemble_outliers
    
    def get_outlier_statistics(self, data, outliers):
        """Get statistics about detected outliers"""
        data = np.array(data)
        outliers = np.array(outliers, dtype=bool)
        
        normal_data = data[~outliers]
        outlier_data = data[outliers]
        
        stats = {
            'total_samples': len(data),
            'n_outliers': np.sum(outliers),
            'outlier_percentage': (np.sum(outliers) / len(data)) * 100,
            'normal_data': {
                'mean': np.mean(normal_data) if len(normal_data) > 0 else None,
                'std': np.std(normal_data) if len(normal_data) > 0 else None,
                'min': np.min(normal_data) if len(normal_data) > 0 else None,
                'max': np.max(normal_data) if len(normal_data) > 0 else None
            },
            'outlier_data': {
                'mean': np.mean(outlier_data) if len(outlier_data) > 0 else None,
                'std': np.std(outlier_data) if len(outlier_data) > 0 else None,
                'min': np.min(outlier_data) if len(outlier_data) > 0 else None,
                'max': np.max(outlier_data) if len(outlier_data) > 0 else None
            }
        }
        
        return stats
    
    def plot_outlier_analysis(self, data, outliers, method='zscore'):
        """Prepare data for plotting outlier analysis"""
        data = np.array(data)
        outliers = np.array(outliers, dtype=bool)
        
        plot_data = {
            'data': data,
            'outliers': outliers,
            'normal_indices': np.where(~outliers)[0],
            'outlier_indices': np.where(outliers)[0],
            'method': method,
            'info': self.outliers_info.get(method, {})
        }
        
        return plot_data