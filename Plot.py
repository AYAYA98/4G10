import numpy as np
import matplotlib.pyplot as plt

# Assuming you already have:
# hand_train: shape (2, 400, 16) - Actual hand velocities for training set
# V_pred_baseline: shape (2, 400, 16) - Predicted velocities from baseline method
# V_pred_kalman: shape (2, 400, 16) - Predicted velocities from Kalman filter approach

import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import requests
import scipy.signal as signal

#-----------------------------------------------
# Load Data
#-----------------------------------------------
r = requests.get('http://4G10.cbl-cambridge.org/data.npz', stream=True)
data = np.load(BytesIO(r.raw.read()))

hand_train = data["hand_train"]       # shape: (2, 400, 16)
neural_train = data["neural_train"]   # shape: (N, 400, 16)
neural_test = data["neural_test"]     # shape: (N, 100, 16)

A = data["hand_KF_A"]
C = data["hand_KF_C"]
mu0 = data["hand_KF_mu0"].squeeze()    # Ensure shape (10,)
Sigma0 = data["hand_KF_Sigma0"]
Q = data["hand_KF_Q"]
R = data["hand_KF_R"]

N, K_train, T = neural_train.shape
_, K_test, _ = neural_test.shape

#-----------------------------------------------
# Baseline Decoder (Gaussian smoothing + Ridge Regression)
#-----------------------------------------------
# Define a Gaussian smoothing function
def smooth_neural_data(neural_data, gauss_filter):
    N, K, T = neural_data.shape
    smoothed = np.zeros_like(neural_data, dtype=float)
    for i in range(N):
        for k in range(K):
            smoothed[i, k, :] = signal.convolve(neural_data[i, k, :], gauss_filter, mode='same')
    return smoothed

# Choose smoothing parameters
sigma_ms = 80
sigma_in_bins = sigma_ms / 50.0
filter_radius = int(np.ceil(4*sigma_in_bins))
t = np.arange(-filter_radius, filter_radius+1)
gauss_filter = np.exp(-t**2/(2*sigma_in_bins**2))
gauss_filter /= gauss_filter.sum()

# Smooth training and test neural data
smoothed_neural_train = smooth_neural_data(neural_train, gauss_filter)
smoothed_neural_test = smooth_neural_data(neural_test, gauss_filter)

# Flatten for ridge regression
X_train = smoothed_neural_train.reshape(N, K_train*T)   # N x (400*16)
V_train = hand_train.reshape(2, K_train*T)              # 2 x (400*16)

lambda_reg = 10.0
XXT = X_train @ X_train.T
VX_T = V_train @ X_train.T
W = VX_T @ np.linalg.inv(XXT + lambda_reg * np.eye(N))  # 2xN

X_test = smoothed_neural_test.reshape(N, K_test*T)       # N x (100*16)
V_pred_baseline_flat = W @ X_test                        # 2 x (100*16)
V_pred_baseline = V_pred_baseline_flat.reshape(2, K_test, T)  # shape (2,100,16)

#-----------------------------------------------
# Kalman-based Decoder
#-----------------------------------------------
# 1. Run Kalman smoothing on hand velocities to get z_train (as per Section 3.1).
# (You should have implemented kalman_smoother_for_velocities previously.)

def kalman_smoother_for_velocities(v_data, A, C, mu0, Sigma0, Q, R):
    T = v_data.shape[1]
    z_dim = A.shape[0]
    
    z_filt = np.zeros((z_dim, T))
    Sigma_filt = np.zeros((z_dim, z_dim, T))
    
    z_pred = mu0.copy()
    Sigma_pred = Sigma0.copy()
    
    for t in range(T):
        if t > 0:
            z_pred = A @ z_filt[:, t-1]
            Sigma_pred = A @ Sigma_filt[:, :, t-1] @ A.T + Q
        
        y = v_data[:, t]
        S = C @ Sigma_pred @ C.T + R
        K = Sigma_pred @ C.T @ np.linalg.inv(S)
        z_filt[:, t] = z_pred + K @ (y - C @ z_pred)
        Sigma_filt[:, :, t] = Sigma_pred - K @ C @ Sigma_pred

    z_smooth = np.zeros((z_dim, T))
    Sigma_smooth = np.zeros((z_dim, z_dim, T))
    z_smooth[:, -1] = z_filt[:, -1]
    Sigma_smooth[:, :, -1] = Sigma_filt[:, :, -1]

    for t in range(T-2, -1, -1):
        Sigma_f = Sigma_filt[:, :, t]
        Sigma_fp = A @ Sigma_f @ A.T + Q
        J = Sigma_f @ A.T @ np.linalg.inv(Sigma_fp)
        
        z_smooth[:, t] = z_filt[:, t] + J @ (z_smooth[:, t+1] - A @ z_filt[:, t])
        Sigma_smooth[:, :, t] = Sigma_f + J @ (Sigma_smooth[:, :, t+1] - Sigma_fp) @ J.T

    return z_smooth, Sigma_smooth

z_train = np.zeros((10, K_train, T))
for k in range(K_train):
    v_trial = hand_train[:, k, :]
    z_smooth, _ = kalman_smoother_for_velocities(v_trial, A, C, mu0, Sigma0, Q, R)
    z_train[:, k, :] = z_smooth

# 2. Fit D and S (Section 3.2)
neural_mean = np.mean(neural_train, axis=(1,2), keepdims=True)
neural_train_centered = neural_train - neural_mean
neural_test_centered = neural_test - neural_mean

X = neural_train_centered.reshape(N, K_train*T)
Z = z_train.reshape(10, K_train*T)

ZZT = Z @ Z.T
XZ_T = X @ Z.T
D = XZ_T @ np.linalg.inv(ZZT)

XXT = X @ X.T
ZX_T = Z @ X.T
S = (XXT - D @ ZX_T) / (K_train*T)

# 3. Kalman filter on test neural data (Section 3.3)
def kalman_filter_for_neural(x_data, A, D, mu0, Sigma0, Q, S):
    N, T = x_data.shape
    z_dim = A.shape[0]
    z_filt = np.zeros((z_dim, T))
    Sigma_filt = np.zeros((z_dim, z_dim, T))
    
    z_pred = mu0.copy()
    Sigma_pred = Sigma0.copy()
    
    for t in range(T):
        if t > 0:
            z_pred = A @ z_filt[:, t-1]
            Sigma_pred = A @ Sigma_filt[:, :, t-1] @ A.T + Q
        
        x_t = x_data[:, t]
        S_t = D @ Sigma_pred @ D.T + S
        K_t = Sigma_pred @ D.T @ np.linalg.inv(S_t)
        
        innovation = x_t - D @ z_pred
        z_filt[:, t] = z_pred + K_t @ innovation
        Sigma_filt[:, :, t] = Sigma_pred - K_t @ D @ Sigma_pred

    return z_filt, Sigma_filt

V_pred_kalman = np.zeros((2, K_test, T))
for k in range(K_test):
    x_trial = neural_test_centered[:, k, :]
    z_filt, _ = kalman_filter_for_neural(x_trial, A, D, mu0, Sigma0, Q, S)
    v_trial_pred = C @ z_filt
    V_pred_kalman[:, k, :] = v_trial_pred

# At this point you have:
# hand_train          (2,400,16)
# V_pred_baseline     (2,100,16)
# V_pred_kalman       (2,100,16)

# These arrays can now be used for plotting.


trial_idx = 8  # choose a trial to visualize
time = np.arange(16)*50  # 16 time bins * 50 ms per bin = array of time in ms

# Extract velocities for the chosen trial
true_vel = hand_train[:, trial_idx, :]  # shape (2,16)
pred_vel_baseline_trial = V_pred_baseline[:, trial_idx, :]  # shape (2,16)
pred_vel_kalman_trial = V_pred_kalman[:, trial_idx, :]      # shape (2,16)

plt.figure(figsize=(6,4))
plt.plot(time, true_vel[0,:], 'k-o', label='True Vx')
plt.plot(time, pred_vel_baseline_trial[0,:], 'r--o', label='Baseline Vx')
plt.plot(time, pred_vel_kalman_trial[0,:], 'b--o', label='Kalman Vx')
plt.xlabel('Time (ms)')
plt.ylabel('Velocity (a.u.)')
plt.title('Comparison of Predicted vs True Hand Velocity (X-component)')
plt.legend()
plt.tight_layout()
plt.savefig('velocity_comparison_x.png', dpi=300)
plt.close()

print("Plot saved as velocity_comparison_x.png")
