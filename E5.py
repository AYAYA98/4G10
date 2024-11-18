import numpy as np
import matplotlib.pyplot as plt
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
from sklearn.decomposition import PCA
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
    """
    Computes the gradient of the log-likelihood w.r.t. beta.
    """
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
    """
    Estimates the antisymmetric matrix A from Z.
    """
    H = construct_H(M)
    b, Q = compute_gradient(Z, H, M)
    beta = np.linalg.solve(Q, b)  
    A = np.einsum('a,aij->ij', beta, H)  
    return A

A_est = estimate_A(Z, M)

eigenvalues, eigenvectors = np.linalg.eig(A_est)

def plot_2d_trajectories(Z, eigenvector, title):
    """
    Projects Z onto the 2D plane defined by the real and imaginary parts of the eigenvector and plots trajectories.
    """
    real_part = np.real(eigenvector)
    imag_part = np.imag(eigenvector)
    P = np.vstack([real_part / np.linalg.norm(real_part), imag_part / np.linalg.norm(imag_part)]) 
    Z_projected = P @ Z.reshape(Z.shape[0], -1)  
    Z_projected = Z_projected.reshape(2, Z.shape[1], Z.shape[2])  
    
    
    initial_points_x = Z_projected[0, :, 0] 
    initial_points_y = Z_projected[1, :, 0]  
    colors = cond_color.get_colors(initial_points_x, initial_points_y)


    plt.figure(figsize=(10, 8))
    for condition in range(Z_projected.shape[1]):
        plt.plot(
            Z_projected[0, condition, :], Z_projected[1, condition, :],
            color=colors[condition], alpha=0.8, linewidth=0.75
        )
        
        cond_color.plot_start(
            [Z_projected[0, condition, 0]],  
            [Z_projected[1, condition, 0]],  
            [colors[condition]],
            markersize=100
        )
        
        cond_color.plot_end(
            [Z_projected[0, condition, -1]],  
            [Z_projected[1, condition, -1]],  
            [colors[condition]],
            markersize=20
        )
    plt.xlabel("Projected Real Part")
    plt.ylabel("Projected Imaginary Part")
    plt.title(title)
    plt.show()


index_fastest = np.argmax(np.imag(eigenvalues))
plot_2d_trajectories(Z, eigenvectors[:, index_fastest], "2D Trajectories in Fastest Rotational Plane")


index_second = np.argsort(np.imag(eigenvalues))[-2]
index_third = np.argsort(np.imag(eigenvalues))[-3]

plot_2d_trajectories(Z, eigenvectors[:, index_second], "2D Trajectories in Second Rotational Plane")
plot_2d_trajectories(Z, eigenvectors[:, index_third], "2D Trajectories in Third Rotational Plane")


eigenvalues, eigenvectors = np.linalg.eig(A_est)


print("Eigenvalues:")
print(eigenvalues)

print("\nEigenvectors (showing all columns):")
print(eigenvectors)
