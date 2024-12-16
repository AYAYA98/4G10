import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import requests
import scipy.signal as signal

#--------------------------------------------
# Load data (including LDS parameters)
#--------------------------------------------
r = requests.get('http://4G10.cbl-cambridge.org/data.npz', stream=True)
data = np.load(BytesIO(r.raw.read()))

hand_train = data["hand_train"]      # (2, 400, 16)
neural_train = data["neural_train"]  # (N, 400, 16)
neural_test = data["neural_test"]    # (N, 100, 16)

A = data["hand_KF_A"]          # 10x10
C = data["hand_KF_C"]          # 2x10
mu0 = data["hand_KF_mu0"].reshape(-1)  # (10,)
Sigma0 = data["hand_KF_Sigma0"]# 10x10
Q = data["hand_KF_Q"]          # 10x10
R = data["hand_KF_R"]          # 2x2

def smooth_neural_data(neural_data, gauss_filter):
    N, K, T = neural_data.shape
    smoothed = np.zeros_like(neural_data, dtype=float)
    for i in range(N):
        for k in range(K):
            smoothed[i, k, :] = signal.convolve(neural_data[i, k, :], gauss_filter, mode='same')
    return smoothed

def kalman_smoother_for_velocities(v_data, A, C, mu0, Sigma0, Q, R):
    K = v_data.shape[1]
    T = v_data.shape[2]
    z_dim = A.shape[0]

    z_filt = np.zeros((z_dim, K, T))
    Sigma_filt = np.zeros((z_dim, z_dim, K, T))

    # Forward pass (Kalman filter)
    for k in range(K):
        v_trial = v_data[:, k, :]
        z_pred = mu0.copy()
        Sigma_pred = Sigma0.copy()
        for t in range(T):
            y = v_trial[:, t]
            S = C @ Sigma_pred @ C.T + R
            K_gain = Sigma_pred @ C.T @ np.linalg.inv(S)
            z_filt[:, k, t] = z_pred + K_gain @ (y - C @ z_pred)
            Sigma_filt[:, :, k, t] = Sigma_pred - K_gain @ C @ Sigma_pred
            if t < T-1:
                z_pred = A @ z_filt[:, k, t]
                Sigma_pred = A @ Sigma_filt[:, :, k, t] @ A.T + Q

    # Backward pass (RTS smoother)
    z_smooth = np.zeros_like(z_filt)
    Sigma_smooth = np.zeros_like(Sigma_filt)
    for k in range(K):
        z_smooth[:, k, -1] = z_filt[:, k, -1]
        Sigma_smooth[:, :, k, -1] = Sigma_filt[:, :, k, -1]
        for t in range(T-2, -1, -1):
            Sigma_f = Sigma_filt[:, :, k, t]
            Sigma_fp = A @ Sigma_f @ A.T + Q
            J = Sigma_f @ A.T @ np.linalg.inv(Sigma_fp)
            z_smooth[:, k, t] = z_filt[:, k, t] + J @ (z_smooth[:, k, t+1] - A @ z_filt[:, k, t])
            Sigma_smooth[:, :, k, t] = Sigma_f + J @ (Sigma_smooth[:, :, k, t+1] - Sigma_fp) @ J.T

    return z_smooth

#--------------------------------------------
# Smoothing parameters
#--------------------------------------------
sigma_ms = 80
sigma_in_bins = sigma_ms / 50.0
filter_radius = int(np.ceil(4 * sigma_in_bins))
t = np.arange(-filter_radius, filter_radius+1)
gauss_filter = np.exp(-t**2/(2*sigma_in_bins**2))
gauss_filter /= gauss_filter.sum()

# Smooth neural data
smoothed_neural_train = smooth_neural_data(neural_train, gauss_filter)
smoothed_neural_test = smooth_neural_data(neural_test, gauss_filter)

N, K_train, T = smoothed_neural_train.shape
_, K_test, _ = smoothed_neural_test.shape

X_train = smoothed_neural_train.reshape(N, K_train*T)
V_train = hand_train.reshape(2, K_train*T)

# Baseline ridge regression
lambda_reg = 100
XXT = X_train @ X_train.T
VX_T = V_train @ X_train.T
XXT_reg = XXT + lambda_reg * np.eye(N)
W = VX_T @ np.linalg.inv(XXT_reg)

# Predict on test set (baseline)
X_test = smoothed_neural_test.reshape(N, K_test*T)
V_pred_test_flat = W @ X_test
V_pred_test = V_pred_test_flat.reshape(2, K_test, T)
V_pred_test = V_pred_test.astype(np.float64)
np.save("my_predictions_baseline.npy", V_pred_test)

# Predict on training set (baseline)
V_pred_train_flat = W @ X_train
V_pred_train = V_pred_train_flat.reshape(2, K_train, T)

#--------------------------------------------
# Kalman smoothing on hand_train to get z_train
#--------------------------------------------
z_train = kalman_smoother_for_velocities(hand_train, A, C, mu0, Sigma0, Q, R)
# z_train: (10, K_train, T)

# Fit D, S for neural likelihood
Z_train = z_train.reshape(10, K_train*T)
X_train_centered = X_train.copy()
mean_neural = np.mean(X_train_centered, axis=1, keepdims=True)
X_train_centered = X_train_centered - mean_neural

ZzT = Z_train @ Z_train.T
XzT = X_train_centered @ Z_train.T
D_star = XzT @ np.linalg.inv(ZzT)
XxT = X_train_centered @ X_train_centered.T
S_star = (XxT - D_star @ (Z_train @ X_train_centered.T)) / (K_train*T)

# Kalman filter on training neural data for comparison
X_train_centered_3d = X_train_centered.reshape(N, K_train, T)

def kalman_filter_for_neural(x_data, A, D, mu0, Sigma0, Q, S):
    N, K, T = x_data.shape
    z_dim = A.shape[0]
    z_filt = np.zeros((z_dim, K, T))
    Sigma_filt = np.zeros((z_dim, z_dim, K, T))

    for k in range(K):
        x_trial = x_data[:, k, :]
        z_pred = mu0.copy()
        Sigma_pred = Sigma0.copy()
        for t in range(T):
            x_t = x_trial[:, t]
            S_t = D @ Sigma_pred @ D.T + S
            K_gain = Sigma_pred @ D.T @ np.linalg.inv(S_t)
            z_filt[:, k, t] = z_pred + K_gain @ (x_t - D @ z_pred)
            Sigma_filt[:, :, k, t] = Sigma_pred - K_gain @ D @ Sigma_pred
            if t < T-1:
                z_pred = A @ z_filt[:, k, t]
                Sigma_pred = A @ Sigma_filt[:, :, k, t] @ A.T + Q

    return z_filt

z_filt_train = kalman_filter_for_neural(X_train_centered_3d, A, D_star, mu0, Sigma0, Q, S_star)
z_filt_train_2d = z_filt_train.reshape(10, K_train*T)
V_pred_kalman_train_flat = C @ z_filt_train_2d  # (2, K_train*T)
V_pred_kalman_train = V_pred_kalman_train_flat.reshape(2, K_train, T)

# Choose a trial to compare all three: True, Baseline, Kalman
trial_idx = 1
time_bins = np.arange(T)*50
true_vel_x = hand_train[0, trial_idx, :]
pred_vel_x_baseline = V_pred_train[0, trial_idx, :]
pred_vel_x_kalman = V_pred_kalman_train[0, trial_idx, :]

true_vel_y = hand_train[1, trial_idx, :]
pred_vel_y_baseline = V_pred_train[1, trial_idx, :]
pred_vel_y_kalman = V_pred_kalman_train[1, trial_idx, :]

plt.figure(figsize=(6,6))
# X component
plt.subplot(2,1,1)
plt.plot(time_bins, true_vel_x, 'k-', label='True Vx')
plt.plot(time_bins, pred_vel_x_baseline, 'r--', label='Baseline Vx')
plt.plot(time_bins, pred_vel_x_kalman, 'b-.', label='Kalman Vx')
plt.xlabel('Time (ms)')
plt.ylabel('Vel (X)')
plt.title('X-Comp: True vs Baseline vs Kalman (Train)')
plt.legend()
plt.grid(True)

# Y component
plt.subplot(2,1,2)
plt.plot(time_bins, true_vel_y, 'k-', label='True Vy')
plt.plot(time_bins, pred_vel_y_baseline, 'r--', label='Baseline Vy')
plt.plot(time_bins, pred_vel_y_kalman, 'b-.', label='Kalman Vy')
plt.xlabel('Time (ms)')
plt.ylabel('Vel (Y)')
plt.title('Y-Comp: True vs Baseline vs Kalman (Train)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('train_comparison_baseline_kalman.png', dpi=300)
plt.show()

print("Comparison plot with Kalman predictions saved as 'train_comparison_baseline_kalman.png'.")



trials_to_plot = [0, 50, 100, 150]  # pick any 4 trial indices

# Create a figure with 4 rows and 2 columns:
# Each row = one trial
# Column 1 = X-component, Column 2 = Y-component
fig, axes = plt.subplots(len(trials_to_plot), 2, figsize=(8, 10))
fig.suptitle('Comparison of True vs. Baseline vs. Kalman Predictions for 4 Trials (Training Data)')

for i, trial_idx in enumerate(trials_to_plot):
    true_vel_x = hand_train[0, trial_idx, :]
    pred_vel_x_baseline = V_pred_train[0, trial_idx, :]
    pred_vel_x_kalman = V_pred_kalman_train[0, trial_idx, :]

    true_vel_y = hand_train[1, trial_idx, :]
    pred_vel_y_baseline = V_pred_train[1, trial_idx, :]
    pred_vel_y_kalman = V_pred_kalman_train[1, trial_idx, :]

    # X-component subplot
    ax_x = axes[i, 0]
    ax_x.plot(time_bins, true_vel_x, 'k-', label='True Vx')
    ax_x.plot(time_bins, pred_vel_x_baseline, 'r--', label='Baseline Vx')
    ax_x.plot(time_bins, pred_vel_x_kalman, 'b-.', label='Kalman Vx')
    ax_x.set_xlabel('Time (ms)')
    ax_x.set_ylabel('Vel (X)')
    ax_x.set_title(f'Trial {trial_idx}: X-Comp')
    ax_x.grid(True)
    if i == 0:
        ax_x.legend()

    # Y-component subplot
    ax_y = axes[i, 1]
    ax_y.plot(time_bins, true_vel_y, 'k-', label='True Vy')
    ax_y.plot(time_bins, pred_vel_y_baseline, 'r--', label='Baseline Vy')
    ax_y.plot(time_bins, pred_vel_y_kalman, 'b-.', label='Kalman Vy')
    ax_y.set_xlabel('Time (ms)')
    ax_y.set_ylabel('Vel (Y)')
    ax_y.set_title(f'Trial {trial_idx}: Y-Comp')
    ax_y.grid(True)
    if i == 0:
        ax_y.legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for suptitle
plt.savefig('four_trials_comparison.png', dpi=300)
plt.show()