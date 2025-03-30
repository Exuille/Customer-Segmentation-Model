import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from datetime import datetime
from yellowbrick.cluster import SilhouetteVisualizer

df = pd.read_excel('Online Retail.xlsx')
df = df.dropna(subset=['CustomerID'])
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
}).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalPrice': 'Monetary'})

train_data, temp_data = train_test_split(rfm, test_size=0.3, random_state=42)
test_data, val_data = train_test_split(temp_data, test_size=0.5, random_state=42)

sns.pairplot(rfm)
plt.show()

correlation_matrix = rfm.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

rfm_features = ['Recency', 'Frequency', 'Monetary']
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_data[rfm_features])

inertia = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(train_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(range(2, 10), inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

kmeans = KMeans(n_clusters=9, random_state=42, n_init=10)
train_data['Cluster'] = kmeans.fit_predict(train_scaled)

visualizer = SilhouetteVisualizer(kmeans)
visualizer.fit(train_scaled)
visualizer.show()

best_k = None
lowest_inertia = float('inf')

for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(train_scaled)
    
    if kmeans.inertia_ < lowest_inertia:
        lowest_inertia = kmeans.inertia_
        best_k = k

print(f'Best number of clusters: {best_k}')

val_scaled = scaler.transform(val_data[rfm_features])
rfm_scaled = scaler.transform(rfm[rfm_features])

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
kmeans.fit(train_scaled)

val_data['Cluster'] = kmeans.predict(val_scaled)

final_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(n_clusters=best_k, random_state=42, n_init=10))
])

final_pipeline.fit(train_scaled)
rfm['Cluster'] = final_pipeline.predict(rfm_scaled)

rfm.to_csv('customer_segments.csv', index=True)
print("Pipeline complete! Results saved to 'customer_segments.csv'.")
