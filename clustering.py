import pandas as pd
from sklearn.cluster import KMeans

# Load preprocessed data
df_scaled = pd.read_csv('preprocessed_data.csv')

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df_scaled['Cluster'] = kmeans.fit_predict(df_scaled)

# Save results with cluster assignments
df_scaled.to_csv('clustered_data.csv', index=False)
