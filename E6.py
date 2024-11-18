import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import cond_color 

plt.rc('font', size=14)  
plt.rc('axes', titlesize=16)  
plt.rc('axes', labelsize=14)  
plt.rc('xtick', labelsize=12) 
plt.rc('ytick', labelsize=12)  
plt.rc('legend', fontsize=12)  

psths_data = np.load('psths.npz')
X = psths_data['X'] 
times = psths_data['times']  

time_start = -150
time_end = 200
time_mask = (times >= time_start) & (times <= time_end)
X_limited = X[:, :, time_mask]  


N, C, T = X_limited.shape

a = X_limited.max(axis=(1, 2), keepdims=True)
b = X_limited.min(axis=(1, 2), keepdims=True)
X_normalized = (X_limited - b) / (a - b + 5)
X_mean_centered = X_normalized - X_normalized.mean(axis=1, keepdims=True)

X_reshaped = X_mean_centered.reshape(N, C * T)  

M = 12
pca = PCA(n_components=M)
Z_pca = pca.fit_transform(X_reshaped.T).T  
Z = Z_pca.reshape(M, C, T)  

VM = pca.components_

def construct_H(M):
    K = M * (M - 1) // 2
    H = np.zeros((K, M, M))
    idx = 0
    for i in range(M):
        for j in range(i + 1, M):
            H[idx, i, j] = 1
            H[idx, j, i] = -1
            idx += 1
    return H

def compute_gradient(Z, H, M):
    C, T = Z.shape[1], Z.shape[2]
    Delta_Z = Z[:, :, 1:] - Z[:, :, :-1] 
    Z_sliced = Z[:, :, :-1]  
    W = np.tensordot(H, Z_sliced, axes=1)  
    W_flattened = W.reshape(W.shape[0], -1)  
    Delta_Z_flattened = Delta_Z.reshape(-1)  
    b = np.dot(W_flattened, Delta_Z_flattened)  
    Q = np.tensordot(W, W, axes=([1, 2, 3], [1, 2, 3]))  
    return b, Q

def estimate_A(Z, M):
    H = construct_H(M)
    b, Q = compute_gradient(Z, H, M)
    beta = np.linalg.solve(Q, b)  
    A = np.einsum('a,aij->ij', beta, H)  
    return A

A_est = estimate_A(Z, M)

eigenvalues, eigenvectors = np.linalg.eig(A_est)

index_fastest = np.argmax(np.imag(eigenvalues))
P_FR = np.vstack([np.real(eigenvectors[:, index_fastest]), np.imag(eigenvectors[:, index_fastest])])
P_FR = P_FR / np.linalg.norm(P_FR, axis=1, keepdims=True) 

Z_projected = P_FR @ Z.reshape(M, -1)  
Z_projected = Z_projected.reshape(2, C, T)  

time_pre_start = -800
time_pre_end = -150
time_pre_mask = (times >= time_pre_start) & (times <= time_pre_end)
X_pre_movement = X[:, :, time_pre_mask]

a_pre = X_pre_movement.max(axis=(1, 2), keepdims=True)
b_pre = X_pre_movement.min(axis=(1, 2), keepdims=True)
X_pre_normalized = (X_pre_movement - b_pre) / (a_pre - b_pre + 5)
X_pre_mean_centered = X_pre_normalized - X_pre_normalized.mean(axis=1, keepdims=True)

X_pre_reshaped = X_pre_mean_centered.reshape(N, -1)
Z_pre_pca = pca.transform(X_pre_reshaped.T).T
Z_pre = Z_pre_pca.reshape(M, C, -1)

Z_pre_projected = P_FR @ Z_pre.reshape(M, -1)  
Z_pre_projected = Z_pre_projected.reshape(2, C, -1)  

plt.figure(figsize=(10, 10))


colors = cond_color.get_colors(Z_projected[0, :, 0], Z_projected[1, :, 0])
for condition in range(Z_projected.shape[1]):
    plt.plot(
        Z_projected[0, condition, :], Z_projected[1, condition, :],
        color=colors[condition], alpha=0.2, linewidth=0.75
    )


final_points_x = Z_pre_projected[0, :, -1]
final_points_y = Z_pre_projected[1, :, -1]
alt_colors = cond_color.get_colors(final_points_x, final_points_y, alt_colors=True)

for condition in range(Z_pre_projected.shape[1]):
    plt.plot(
        Z_pre_projected[0, condition, :], Z_pre_projected[1, condition, :],
        color=alt_colors[condition], alpha=0.8, linewidth=0.75
    )
    cond_color.plot_start(
        [Z_pre_projected[0, condition, 0]],
        [Z_pre_projected[1, condition, 0]],
        [alt_colors[condition]],
        markersize=50
    )
    cond_color.plot_end(
        [Z_pre_projected[0, condition, -1]],
        [Z_pre_projected[1, condition, -1]],
        [alt_colors[condition]],
        markersize=20
    )

plt.xlabel("Projected Real Part")
plt.ylabel("Projected Imaginary Part")
plt.title("Pre-Movement and Movement Trajectories in FR Plane")
plt.show()
