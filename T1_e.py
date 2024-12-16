import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import requests
import scipy.signal as signal

#--------------------------------------------
# Load the provided data
#--------------------------------------------
r = requests.get('http://4G10.cbl-cambridge.org/data.npz', stream=True)
data = np.load(BytesIO(r.raw.read()))

hand_train = data["hand_train"]      # shape: (2, 400, 16)
neural_train = data["neural_train"]  # shape: (N, 400, 16)
neural_test = data["neural_test"]    # shape: (N, 100, 16)

#--------------------------------------------
# Create an exponential decay filter (causal)
#--------------------------------------------
def create_exp_decay_filter(decay_ms=100, bin_size_ms=50):
    decay_in_bins = decay_ms / bin_size_ms
    # cover about 4*decay_in_bins for the kernel length
    length = int(np.ceil(4 * decay_in_bins))
    t = np.arange(0, length + 1)
    exp_filter = np.exp(-t / decay_in_bins)
    exp_filter /= exp_filter.sum()
    return exp_filter

#--------------------------------------------
# Causal smoothing function using the provided filter
# Uses 'full' convolution and truncates to ensure only past data are used
#--------------------------------------------
def smooth_neural_data_causal(neural_data, causal_filter):
    # neural_data shape: (N, K, T)
    N, K, T = neural_data.shape
    smoothed = np.zeros_like(neural_data, dtype=float)
    for i in range(N):
        for k in range(K):
            # Full convolution since we are causal
            conv_res = np.convolve(neural_data[i, k, :], causal_filter, mode='full')
            # conv_res length = T + length of filter - 1
            # Since it's causal, conv_res[t] includes data from [0...t]
            # We only take first T elements to match original length
            smoothed[i, k, :] = conv_res[:T]
    return smoothed

#--------------------------------------------
# Apply exponential decay smoothing
#--------------------------------------------
exp_filter = create_exp_decay_filter(decay_ms=60, bin_size_ms=50)
smoothed_neural_train = smooth_neural_data_causal(neural_train, exp_filter)
smoothed_neural_test = smooth_neural_data_causal(neural_test, exp_filter)

N, K_train, T = smoothed_neural_train.shape  # N = number of neurons, K_train=400, T=16
_, K_test, _ = smoothed_neural_test.shape     # K_test=100

#--------------------------------------------
# Flatten training data and perform ridge regression
#--------------------------------------------
X_train = smoothed_neural_train.reshape(N, K_train * T)  # N x (400*16)
V_train = hand_train.reshape(2, K_train * T)             # 2 x (400*16)

lambda_reg = 100
XXT = X_train @ X_train.T  # N x N
VX_T = V_train @ X_train.T   # 2 x N

XXT_reg = XXT + lambda_reg * np.eye(N)
W = VX_T @ np.linalg.inv(XXT_reg)  # W is 2 x N

#--------------------------------------------
# Predict on test set using exponential decay smoothing
#--------------------------------------------
X_test = smoothed_neural_test.reshape(N, K_test * T)  # N x (100*16)
V_pred = W @ X_test  # 2 x (100*16)
V_pred = V_pred.reshape(2, K_test, T)  # 2 x 100 x 16

V_pred = V_pred.astype(np.float64)
np.save("my_predictions_exp.npy", V_pred)

print("Exponential decay smoothing and prediction complete. Predictions saved as 'my_predictions_exp.npy'.")
