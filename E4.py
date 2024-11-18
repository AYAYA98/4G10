import numpy as np
import matplotlib.pyplot as plt

plt.rc('font', size=14)  
plt.rc('axes', titlesize=16)  
plt.rc('axes', labelsize=14)  
plt.rc('xtick', labelsize=12) 
plt.rc('ytick', labelsize=12)  
plt.rc('legend', fontsize=12)  

def construct_H(M):
    """
    Constructs the 3D antisymmetric H array.
    Parameters:
        M (int): Number of dimensions in Z
    Returns:
        H (numpy array): Shape (K, M, M), where K = M * (M - 1) // 2
    """
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
    Parameters:
        Z (numpy array): Input data, shape (M, C, T)
        H (numpy array): Antisymmetric basis, shape (K, M, M)
        M (int): Number of dimensions in Z
    Returns:
        b (numpy array): Row vector, shape (K,)
        Q (numpy array): Matrix, shape (K, K)
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
    Parameters:
        Z (numpy array): Input data, shape (M, C, T)
        M (int): Number of dimensions in Z
    Returns:
        A (numpy array): Antisymmetric matrix, shape (M, M)
    """
    H = construct_H(M)
    b, Q = compute_gradient(Z, H, M)
    beta = np.linalg.solve(Q, b)
    A = np.einsum('a,aij->ij', beta, H) 
    return A


def test_A(Z_test, A_test):
    """
    Tests the algorithm for estimating A using Z_test and compares it with A_test.
    Parameters:
        Z_test (numpy array): Test data, shape (M, C, T)
        A_test (numpy array): Ground truth matrix, shape (M, M)
    Returns:
        A_est (numpy array): Estimated matrix, shape (M, M)
        error (float): Maximum absolute error between A_est and A_test
    """
    A_est = estimate_A(Z_test, Z_test.shape[0])  
    
    error = np.max(np.abs(A_est - A_test))
    
    return A_est, error

'''# Load test data
test_data = np.load('test.npz')
Z_test = test_data['Z_test']  # Shape: (M, C, T)
A_test = test_data['A_test']  # Ground truth matrix, shape (M, M)


A_est, error = test_A(Z_test, A_test)


print(f"Max absolute error between A_est and A_test: {error}")

# Color plot of A_est
plt.imshow(A_est, cmap='seismic', interpolation='nearest')
plt.colorbar(label="Value")
plt.title("Estimated Antisymmetric Matrix A")
plt.xlabel("Dimension Index")
plt.ylabel("Dimension Index")
plt.show()'''


from sklearn.decomposition import PCA

psths_data = np.load('psths.npz')
X = psths_data['X']  


time_start = -150
time_end = 300
times = psths_data['times']  
time_mask = (times >= time_start) & (times <= time_end)
X_limited = X[:, :, time_mask]  

N, C, T = X_limited.shape

X_reshaped = X_limited.reshape(N, C * T)  

M = 12
pca = PCA(n_components=M)
Z_pca = pca.fit_transform(X_reshaped.T).T 

Z = Z_pca.reshape(M, C, T)  

A_est = estimate_A(Z, M)

print("Estimated Antisymmetric Matrix A:")
print(A_est)

plt.imshow(A_est, cmap='seismic', interpolation='nearest')
plt.colorbar(label="Value")
plt.title("Estimated Antisymmetric Matrix A")
plt.xlabel("Dimension Index")
plt.ylabel("Dimension Index")
plt.show()
