import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import requests
import scipy.signal as signal

# Load the provided data
r = requests.get('http://4G10.cbl-cambridge.org/data.npz', stream=True)
data = np.load(BytesIO(r.raw.read()))

hand_train = data["hand_train"]
neural_train = data["neural_train"]
neural_test = data["neural_test"]

sigma_ms = 60
sigma_in_bins = sigma_ms / 50.0
filter_radius = int(np.ceil(4 * sigma_in_bins))
t = np.arange(-filter_radius, filter_radius+1)
gauss_filter = np.exp(-t**2/(2*sigma_in_bins**2))
gauss_filter /= gauss_filter.sum()

def smooth_neural_data(neural_data, gauss_filter):
    N, K, T = neural_data.shape
    smoothed = np.zeros_like(neural_data, dtype=float)
    for i in range(N):
        for k in range(K):
            smoothed[i, k, :] = signal.convolve(neural_data[i, k, :], gauss_filter, mode='same')
    return smoothed

smoothed_neural_train = smooth_neural_data(neural_train, gauss_filter)
smoothed_neural_test = smooth_neural_data(neural_test, gauss_filter)

N, K_train, T = smoothed_neural_train.shape
_, K_test, _ = smoothed_neural_test.shape

X_train = smoothed_neural_train.reshape(N, K_train * T)
V_train = hand_train.reshape(2, K_train * T)

lambda_reg = 100
XXT = X_train @ X_train.T
VX_T = V_train @ X_train.T

XXT_reg = XXT + lambda_reg * np.eye(N)
W = VX_T @ np.linalg.inv(XXT_reg)

X_test = smoothed_neural_test.reshape(N, K_test * T)
V_pred = W @ X_test
V_pred = V_pred.reshape(2, K_test, T)

V_pred = V_pred.astype(np.float64)
np.save("my_predictions.npy", V_pred)
