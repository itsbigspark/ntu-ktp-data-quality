import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances



def run_clustering(df, text_cols, numeric_cols):
    """
    Perform clustering using KMeans and DBSCAN, and add clustering labels to the dataframe.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - text_cols (list): List of text columns to use for text-based features.
    - numeric_cols (list): List of numerical columns to use for clustering.

    Returns:
    - df (pd.DataFrame): DataFrame with added cluster labels and similarity features.
    - dict: Dictionary with clustering model objects (kmeans and dbscan).
    """
    # --- Step 1: Text Vectorization and Cosine Similarity Calculation ---
    vectorizer = TfidfVectorizer()
    combined_text = df[text_cols].fillna("").apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    
    # Check if combined_text has any meaningful content after joining
    if combined_text.str.strip().replace('', np.nan).isna().all():
        print("Warning: All text columns are empty or contain only stop words.")
        text_vectors = np.zeros((len(df), 0))  # Fallback to an empty array if no text features
    else:
        text_vectors = vectorizer.fit_transform(combined_text)
        if text_vectors.shape[1] == 0:
            print("Warning: Empty vocabulary; all documents may contain only stop words or are empty.")
            text_vectors = np.zeros((len(df), 0))

    # Calculate cosine similarity features only if text_vectors has valid features
    if text_vectors.shape[1] > 0:
        df['cosine_similarity_listing_title'] = cosine_similarity(text_vectors).mean(axis=1)
        # Adding default cosine similarity columns to prevent missing column issues
        df['cosine_similarity_status'] = cosine_similarity(text_vectors).mean(axis=1)  # Example addition
    else:
        # Set default values if similarity features are not calculable
        df['cosine_similarity_listing_title'] = 0
        df['cosine_similarity_status'] = 0  # Ensure column exists with a default value

    # --- Step 2: Prepare Data for Clustering ---
    if not all(col in df.columns for col in numeric_cols):
        raise ValueError(f"One or more numeric columns {numeric_cols} not found in the DataFrame.")
    
    df[numeric_cols] = df[numeric_cols].fillna(0)
    clustering_data = np.hstack((text_vectors.toarray(), df[numeric_cols].values)) if text_vectors.shape[1] > 0 else df[numeric_cols].values

    # --- Step 3: Dimensionality Reduction using PCA ---
    pca = PCA(n_components=min(10, clustering_data.shape[1]))
    X_pca = pca.fit_transform(clustering_data)

    # --- Step 4: Apply KMeans Clustering ---
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_pca)
    df['kmeans_cluster'] = kmeans_labels

    # --- Step 5: Apply DBSCAN Clustering ---
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_pca)
    df['dbscan_cluster'] = dbscan_labels

    # --- Step 6: Distance to KMeans Centroids ---
    df['euclidean_dist_to_centroid'] = np.min(euclidean_distances(X_pca, kmeans.cluster_centers_), axis=1)

    return df, {'kmeans': kmeans, 'dbscan': dbscan}




def cluster_analysis(df, cluster_col, X_pca, cluster_model=None):
    """
    Analyze the clusters and print details about each cluster.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data with clustering labels.
    - cluster_col (str): Column name for the cluster labels to analyze.
    - X_pca (np.ndarray): PCA-transformed data used for clustering.
    - cluster_model: Clustering model (optional) to get centroid information for KMeans.
    """
    clusters = df[cluster_col].unique()
    print(f"\n--- {cluster_col} Cluster Analysis ---")

    for cluster in clusters:
        # Filter records within the current cluster
        cluster_indices = df[df[cluster_col] == cluster].index
        cluster_records = df.loc[cluster_indices]

        # Count classes within the cluster
        class_counts = cluster_records['record_class'].value_counts()
        unique_count = class_counts.get(0, 0)
        similar_count = class_counts.get(1, 0)
        duplicate_count = class_counts.get(2, 0)

        print(f"\nCluster {cluster}:")
        print(f"  Number of records: {len(cluster_records)}")
        print(f"  Unique records: {unique_count}, Overly similar records: {similar_count}, Duplicate records: {duplicate_count}")
        
        # Display example records
        print("  Sample records:")
        print(cluster_records.head())

        # Calculate mean feature values for cluster-level analysis
        if cluster_model is not None and hasattr(cluster_model, 'cluster_centers_'):
            centroid = cluster_model.cluster_centers_[cluster]
            cluster_points = X_pca[cluster_indices]
            avg_dist = np.mean(euclidean_distances(cluster_points, centroid.reshape(1, -1)).flatten())
            print(f"  Average Euclidean distance to centroid: {avg_dist}")

        print("  Mean feature values:\n", cluster_records[['cosine_similarity_listing_title', 'euclidean_dist_to_centroid']].mean())
