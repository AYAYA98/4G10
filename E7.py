import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import cond_color  

import matplotlib.pyplot as plt

plt.rc('font', size=14)  
plt.rc('axes', titlesize=16)  
plt.rc('axes', labelsize=14)  
plt.rc('xtick', labelsize=12) 
plt.rc('ytick', labelsize=12)  
plt.rc('legend', fontsize=12)  


psths_data = np.load('psths.npz')
X = psths_data['X']  
times = psths_data['times']  


N, C, T = X.shape
t0 = np.where(times == -150)[0][0] 
np.random.seed(42)  


random_conditions = [np.random.choice(C, C // 2, replace=False) for _ in range(N)]  
X_distorted = X.copy()

for neuron in range(N):
    for condition in random_conditions[neuron]:
        X_distorted[neuron, condition, t0:] = (
            2 * X_distorted[neuron, condition, t0] - X_distorted[neuron, condition, t0:]
        )


a_distorted = X_distorted.max(axis=(1, 2), keepdims=True)
b_distorted = X_distorted.min(axis=(1, 2), keepdims=True)
X_normalized_distorted = (X_distorted - b_distorted) / (a_distorted - b_distorted + 5)
X_mean_centered_distorted = X_normalized_distorted - X_normalized_distorted.mean(axis=1, keepdims=True)


X_reshaped_distorted = X_mean_centered_distorted.reshape(N, -1)


M = 12 
pca = PCA(n_components=M)
Z_pca_distorted = pca.fit_transform(X_reshaped_distorted.T).T  
Z_distorted = Z_pca_distorted.reshape(M, C, T)  


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


A_distorted = estimate_A(Z_distorted, M)
eigenvalues_distorted, eigenvectors_distorted = np.linalg.eig(A_distorted)


index_fastest = np.argmax(np.imag(eigenvalues_distorted))
P_FR = np.vstack([np.real(eigenvectors_distorted[:, index_fastest]), np.imag(eigenvectors_distorted[:, index_fastest])])
P_FR = P_FR / np.linalg.norm(P_FR, axis=1, keepdims=True)  

Z_projected_distorted = P_FR @ Z_distorted.reshape(M, -1)  
Z_projected_distorted = Z_projected_distorted.reshape(2, C, T)  


plt.figure(figsize=(10, 10))
colors = cond_color.get_colors(Z_projected_distorted[0, :, 0], Z_projected_distorted[1, :, 0])
for condition in range(Z_projected_distorted.shape[1]):
    plt.plot(
        Z_projected_distorted[0, condition, :], Z_projected_distorted[1, condition, :],
        color=colors[condition], alpha=0.8, linewidth=0.75
    )
    
    cond_color.plot_start(
        [Z_projected_distorted[0, condition, 0]],
        [Z_projected_distorted[1, condition, 0]],
        [colors[condition]],
        markersize=50
    )
    
    cond_color.plot_end(
        [Z_projected_distorted[0, condition, -1]],
        [Z_projected_distorted[1, condition, -1]],
        [colors[condition]],
        markersize=20
    )

plt.xlabel("Projected Real Part")
plt.ylabel("Projected Imaginary Part")
plt.title("Distorted Trajectories in FR Plane")
plt.show()
