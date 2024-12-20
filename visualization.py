import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load clustered data
df = pd.read_csv('customer_data.csv')
df_clustered = pd.read_csv('clustered_data.csv')
df['Cluster'] = df_clustered['Cluster']

# Scatter plot for Spending vs Frequency
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['spending'], y=df['frequency'], hue=df['Cluster'], palette='viridis')
plt.title('Customer Segments')
plt.xlabel('Spending')
plt.ylabel('Frequency')
plt.legend(title='Cluster')
plt.show()

# Cluster count visualization
plt.figure(figsize=(8, 5))
sns.countplot(x='Cluster', data=df, palette='viridis')
plt.title('Customer Distribution Across Clusters')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.show()
