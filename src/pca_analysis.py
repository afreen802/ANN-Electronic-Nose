import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load dataset
data = pd.read_excel("Afreen_Training_Data.xlsx")

# Sensor columns used for PCA
sensor_columns = ['Sensor1', 'Sensor2', 'Sensor3', 'Sensor4']

# Handle missing values and standardize data
X = data[sensor_columns].fillna(data[sensor_columns].mean())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA
pca = PCA(n_components=3)
pca_result = pca.fit_transform(X_scaled)

# Create dataframe for PCA output
df_pca = pd.DataFrame(pca_result, columns=['PC1', 'PC2', 'PC3'])
df_pca['Category'] = data['Category']

# 3D PCA Visualization
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

categories = df_pca['Category'].unique()
colors = ['blue', 'red', 'green', 'orange']

for i, cat in enumerate(categories):
    subset = df_pca[df_pca['Category'] == cat]

    ax.scatter(
        subset['PC1'],
        subset['PC2'],
        subset['PC3'],
        label=cat,
        s=60,
        color=colors[i % len(colors)]
    )

# Axis labels
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.2f}%)')

ax.set_title('3D PCA of Sensor Data')

plt.legend()
plt.show()

# Explained variance table
variance_table = pd.DataFrame({
    'Principal Component': ['PC1', 'PC2', 'PC3'],
    'Explained Variance Ratio (%)':
        np.round(pca.explained_variance_ratio_ * 100, 2)
})

print("\nExplained Variance Table:")
print(variance_table)

# Sensor loading analysis
sensor_loading = pd.DataFrame(
    np.round(pca.components_, 3),
    columns=sensor_columns,
    index=variance_table['Principal Component']
)

print("\nSensor contribution to each PC:")
print(sensor_loading)

# Redundancy analysis
redundant_sensors = []

for sensor in sensor_columns:
    if max(abs(sensor_loading[sensor])) < 0.3:
        redundant_sensors.append(sensor)

if redundant_sensors:
    print(f"\nPotentially redundant sensor(s): "
          f"{', '.join(redundant_sensors)}")
else:
    print("\nNo sensor appears strongly redundant.")
