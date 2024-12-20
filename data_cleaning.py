import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('customer_data.csv')

# Drop missing values
df = df.dropna()

# Select features for clustering
features = ['spending', 'frequency', 'recency']
X = df[features]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save cleaned and preprocessed data
df_scaled = pd.DataFrame(X_scaled, columns=features)
df_scaled.to_csv('preprocessed_data.csv', index=False)
