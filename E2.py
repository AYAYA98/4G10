from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

plt.rc('font', size=14)  
plt.rc('axes', titlesize=16)  
plt.rc('axes', labelsize=14)  
plt.rc('xtick', labelsize=12) 
plt.rc('ytick', labelsize=12)  
plt.rc('legend', fontsize=12)  


data = np.load('psths.npz', allow_pickle=True)
X = data['X']
times = data['times']

a = X.max(axis=(1, 2), keepdims=True)  
b = X.min(axis=(1, 2), keepdims=True)  
X_normalized = (X - b) / (a - b + 5)

X_mean_centered = X_normalized - X_normalized.mean(axis=1, keepdims=True)

time_interval = (times >= -150) & (times <= 300)
X_limited = X_mean_centered[:, :, time_interval]  
N, C, T = X_limited.shape

X_reshaped = X_limited.reshape(N, C * T)

M = 12  
pca = PCA(n_components=M)
Z = pca.fit_transform(X_reshaped)
Z = Z.T

'''# Variance explained by PCA
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = explained_variance_ratio.cumsum()'''

plt.figure(figsize=(10, 6))
plt.hist(a.flatten(), bins=30, color='blue', alpha=0.7, edgecolor='black')
plt.xlabel("Maximum Firing Rate (Hz)")
plt.ylabel("Number of Neurons")
plt.title("Histogram of Maximum Firing Rates Across Neurons")
plt.show()

'''# Plot cumulative explained variance
plt.figure(figsize=(10, 6))
plt.plot(range(1, M + 1), cumulative_variance, marker='o', linestyle='--', label="Cumulative Variance")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Explained Variance by Principal Components")
plt.grid()
plt.legend()
plt.show()'''

print(f"Shape of Normalized PSTHs: {X_normalized.shape}")
print(f"Shape of Mean-Centered PSTHs: {X_mean_centered.shape}")
print(f"Shape of Limited PSTHs: {X_limited.shape}")
print(f"Shape of Z (PCA result): {Z.shape}")
'''print(f"Explained variance by the first {M} components: {explained_variance_ratio}")
print(f"Cumulative variance explained: {cumulative_variance[-1]}")'''
