import numpy as np
import matplotlib.pyplot as plt
import cond_color

plt.rc('font', size=14)  
plt.rc('axes', titlesize=16)  
plt.rc('axes', labelsize=14)  
plt.rc('xtick', labelsize=12) 
plt.rc('ytick', labelsize=12)  
plt.rc('legend', fontsize=12)  

data = np.load('psths.npz', allow_pickle=True)
X = data['X'] 
times = data['times'] 

time_interval = (times >= -150) & (times <= 300)
X_limited = X[:, :, time_interval] 
N, C, T = X_limited.shape 

a = X_limited.max(axis=(1, 2), keepdims=True) 
b = X_limited.min(axis=(1, 2), keepdims=True)
X_normalized = (X_limited - b) / (a - b + 5)

X_mean_centered = X_normalized - X_normalized.mean(axis=1, keepdims=True)

X_reshaped = X_mean_centered.reshape(N, C * T)

X_centered = X_reshaped - X_reshaped.mean(axis=1, keepdims=True) 
sample_covariance = np.cov(X_centered, rowvar=True) 

eigenvalues, eigenvectors = np.linalg.eigh(sample_covariance)
sorted_indices = np.argsort(-eigenvalues)
eigenvectors = eigenvectors[:, sorted_indices]
V_M = eigenvectors[:, :12]

Z = np.dot(V_M.T, X_centered)

Z_reshaped = Z.reshape(12, C, T)
PC1, PC2 = Z_reshaped[0], Z_reshaped[1]

initial_points_x = PC1[:, 0]
initial_points_y = PC2[:, 0] 
colors = cond_color.get_colors(initial_points_x, initial_points_y)

plt.figure(figsize=(10, 10))
for condition in range(C):
    plt.plot(PC1[condition], PC2[condition], color=colors[condition], alpha=0.9, linewidth=1.5)
    cond_color.plot_start(
        [PC1[condition, 0]], [PC2[condition, 0]], [colors[condition]], markersize=100
    )
    cond_color.plot_end(
        [PC1[condition, -1]], [PC2[condition, -1]], [colors[condition]], markersize=20
    )

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Trajectories in PC1-PC2 Space")
plt.grid(True)
plt.show()
