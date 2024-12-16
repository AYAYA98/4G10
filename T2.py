import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import requests
# Load the provided data
r = requests.get('http://4G10.cbl-cambridge.org/data.npz', stream=True)
data = np.load(BytesIO(r.raw.read()))

hand_train = data["hand_train"]      # shape: (2, 400, 16)
neural_train = data["neural_train"]  # shape: (N, 400, 16)
neural_test = data["neural_test"]    # shape: (N, 100, 16)



A = data["hand_KF_A"]          # 10x10
C = data["hand_KF_C"]          # 2x10
mu0 = data["hand_KF_mu0"]      # 10x1
Sigma0 = data["hand_KF_Sigma0"]# 10x10
Q = data["hand_KF_Q"]          # 10x10
R = data["hand_KF_R"]          # 2x2

mu0 = data["hand_KF_mu0"].reshape(-1)  # produces shape (10,)


def kalman_smoother_for_velocities(v_data, A, C, mu0, Sigma0, Q, R):
    T = v_data.shape[1]
    z_filt = np.zeros((10, T))
    Sigma_filt = np.zeros((10, 10, T))

    z_pred = mu0.copy()        # z_pred: (10,)
    Sigma_pred = Sigma0.copy()

    for t in range(T):
        # Prediction step
        if t > 0:
            z_pred = A @ z_filt[:, t-1]            # (10,) @ (10,) = (10,)
            Sigma_pred = A @ Sigma_filt[:, :, t-1] @ A.T + Q

        # Update step
        y = v_data[:, t]        # y: (2,)
        S = C @ Sigma_pred @ C.T + R   # (2x10)(10x10)(10x2)+(2x2)=(2x2)
        K = Sigma_pred @ C.T @ np.linalg.inv(S)    # (10x10)(10x2)(2x2) = (10x2)
        innovation = y - C @ z_pred                # (2,) - (2,) = (2,)
        z_filt[:, t] = z_pred + K @ innovation     # (10,) + (10x2)(2,)=(10,)

        Sigma_filt[:, :, t] = Sigma_pred - K @ C @ Sigma_pred

    # RTS smoother
    z_smooth = np.zeros((10, T))
    Sigma_smooth = np.zeros((10,10,T))
    z_smooth[:, -1] = z_filt[:, -1]
    Sigma_smooth[:, :, -1] = Sigma_filt[:, :, -1]

    for t in range(T-2, -1, -1):
        Sigma_f = Sigma_filt[:, :, t]
        Sigma_fp = A @ Sigma_f @ A.T + Q
        J = Sigma_f @ A.T @ np.linalg.inv(Sigma_fp) # shape (10x10)(10x10)(10x10), still (10x10)
        z_smooth[:, t] = z_filt[:, t] + J @ (z_smooth[:, t+1] - A @ z_filt[:, t])
        Sigma_smooth[:, :, t] = Sigma_f + J @ (Sigma_smooth[:, :, t+1] - Sigma_fp) @ J.T

    return z_smooth, Sigma_smooth



N_neurons, K_train, T = neural_train.shape  
z_train = np.zeros((10, K_train, T))

for k in range(K_train):
    v_trial = hand_train[:, k, :]  # shape: 2 x T
    z_smooth, _ = kalman_smoother_for_velocities(v_trial, A, C, mu0, Sigma0, Q, R)
    z_train[:, k, :] = z_smooth


N, K_train, T = neural_train.shape 

#  Center the neural data
neural_mean = np.mean(neural_train, axis=(1,2), keepdims=True)  # shape (N,1,1)
neural_train_centered = neural_train - neural_mean
neural_test_centered = neural_test - neural_mean  # same mean as training set


X = neural_train_centered.reshape(N, K_train*T)   # N x (K_train*T)
Z = z_train.reshape(10, K_train*T)                # 10 x (K_train*T)


# D^* = (X Z^T)(Z Z^T)^{-1}
ZZT = Z @ Z.T        # 10 x 10
XZ_T = X @ Z.T        # N x 10
D = XZ_T @ np.linalg.inv(ZZT)  # D: N x 10


# S^* = (1/(K*T)) [ (X X^T) - D (Z X^T) ]
XXT = X @ X.T  # N x N
ZX_T = Z @ X.T  # 10 x N

S = (XXT - D @ ZX_T) / (K_train * T)  # S: N x N

'''
print("D shape:", D.shape)
print("S shape:", S.shape)
'''



def kalman_filter_for_neural(x_data, A, D, mu0, Sigma0, Q, S):
    N, T = x_data.shape
    z_dim = A.shape[0]  # 10
    z_filt = np.zeros((z_dim, T))
    Sigma_filt = np.zeros((z_dim, z_dim, T))
    
    # Initial conditions
    z_pred = mu0.copy()        # shape (10,)
    Sigma_pred = Sigma0.copy() # shape (10,10)
    
    for t in range(T):
       
        if t > 0:
            z_pred = A @ z_filt[:, t-1]
            Sigma_pred = A @ Sigma_filt[:, :, t-1] @ A.T + Q
        
       
        x_t = x_data[:, t]  # shape (N,)
        S_t = D @ Sigma_pred @ D.T + S  # (N,N)
        K_t = Sigma_pred @ D.T @ np.linalg.inv(S_t)  # (10,N)
        
        innovation = x_t - D @ z_pred  # (N,)
        z_filt[:, t] = z_pred + K_t @ innovation
        Sigma_filt[:, :, t] = Sigma_pred - K_t @ D @ Sigma_pred

    return z_filt, Sigma_filt


N, K_test, T = neural_test_centered.shape
v_pred_test = np.zeros((2, K_test, T))

for k in range(K_test):
    x_trial = neural_test_centered[:, k, :]  # N x T
    z_filt, _ = kalman_filter_for_neural(x_trial, A, D, mu0, Sigma0, Q, S)
    v_trial_pred = C @ z_filt  # 2 x T
    v_pred_test[:, k, :] = v_trial_pred


v_pred_test = v_pred_test.astype(np.float64)
np.save("my_kalman_predictions.npy", v_pred_test)

