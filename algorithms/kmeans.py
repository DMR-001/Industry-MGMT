import numpy as np

class KMeans:
    """K-Means clustering implementation from scratch"""
    
    def __init__(self, n_clusters=3, max_iters=100, random_state=42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        
    def fit(self, X):
        """Fit the K-means model to the data"""
        np.random.seed(self.random_state)
        
        # Initialize centroids randomly
        n_samples, n_features = X.shape
        self.centroids = np.random.uniform(
            np.min(X, axis=0), np.max(X, axis=0), 
            size=(self.n_clusters, n_features)
        )
        
        for iteration in range(self.max_iters):
            # Assign points to closest centroid
            distances = self.calculate_distances(X)
            new_labels = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.zeros((self.n_clusters, n_features))
            for k in range(self.n_clusters):
                if np.sum(new_labels == k) > 0:
                    new_centroids[k] = np.mean(X[new_labels == k], axis=0)
                else:
                    new_centroids[k] = self.centroids[k]
            
            # Check for convergence
            if np.allclose(self.centroids, new_centroids):
                print(f"Converged after {iteration + 1} iterations")
                break
                
            self.centroids = new_centroids
            self.labels = new_labels
        
        return self
    
    def fit_predict(self, X):
        """Fit the model and return cluster labels"""
        self.fit(X)
        return self.labels
    
    def predict(self, X):
        """Predict cluster labels for new data"""
        distances = self.calculate_distances(X)
        return np.argmin(distances, axis=1)
    
    def calculate_distances(self, X):
        """Calculate distances from each point to each centroid"""
        distances = np.zeros((X.shape[0], self.n_clusters))
        
        for i, centroid in enumerate(self.centroids):
            distances[:, i] = np.sqrt(np.sum((X - centroid)**2, axis=1))
        
        return distances
    
    def inertia(self, X):
        """Calculate within-cluster sum of squares"""
        if self.labels is None:
            return None
            
        inertia = 0
        for k in range(self.n_clusters):
            cluster_points = X[self.labels == k]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - self.centroids[k])**2)
        
        return inertia
    
    def silhouette_score(self, X, labels):
        """Calculate silhouette score for clustering evaluation"""
        n_samples = X.shape[0]
        silhouette_scores = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Calculate a(i) - mean intra-cluster distance
            same_cluster = X[labels == labels[i]]
            if len(same_cluster) > 1:
                a_i = np.mean([self.euclidean_distance(X[i], point) 
                              for point in same_cluster if not np.array_equal(X[i], point)])
            else:
                a_i = 0
            
            # Calculate b(i) - mean nearest-cluster distance
            b_i = float('inf')
            for cluster_id in range(self.n_clusters):
                if cluster_id != labels[i]:
                    other_cluster = X[labels == cluster_id]
                    if len(other_cluster) > 0:
                        mean_dist = np.mean([self.euclidean_distance(X[i], point) 
                                           for point in other_cluster])
                        b_i = min(b_i, mean_dist)
            
            # Calculate silhouette score for sample i
            if max(a_i, b_i) > 0:
                silhouette_scores[i] = (b_i - a_i) / max(a_i, b_i)
            else:
                silhouette_scores[i] = 0
        
        return np.mean(silhouette_scores)
    
    def euclidean_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt(np.sum((point1 - point2)**2))
    
    def get_cluster_info(self, X):
        """Get information about each cluster"""
        if self.labels is None:
            return None
            
        cluster_info = {}
        for k in range(self.n_clusters):
            cluster_points = X[self.labels == k]
            cluster_info[k] = {
                'size': len(cluster_points),
                'centroid': self.centroids[k],
                'mean': np.mean(cluster_points, axis=0) if len(cluster_points) > 0 else None,
                'std': np.std(cluster_points, axis=0) if len(cluster_points) > 0 else None
            }
        
        return cluster_info