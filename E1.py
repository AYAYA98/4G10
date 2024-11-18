import numpy as np
import matplotlib.pyplot as plt


plt.rc('font', size=14)  
plt.rc('axes', titlesize=16)  
plt.rc('axes', labelsize=14)  
plt.rc('xtick', labelsize=12) 
plt.rc('ytick', labelsize=12)  
plt.rc('legend', fontsize=12)  

data = np.load('psths.npz')  
X = data['X'] 
times = data['times']  

subset_neurons = [0, 1, 2, 3]  
subset_conditions = [0, 1, 2, 3] 

fig, axes = plt.subplots(len(subset_neurons), len(subset_conditions), figsize=(15, 10), sharex=True, sharey=True)
for i, neuron in enumerate(subset_neurons):
    for j, condition in enumerate(subset_conditions):
        axes[i, j].plot(times, X[neuron, condition, :], label=f"Neuron {neuron}, Condition {condition}")
        axes[i, j].axvline(x=0, color='r', linestyle='--', label='Movement Onset') 
        axes[i, j].set_title(f"N{neuron}, C{condition}")
        axes[i, j].set_xlabel("Time (ms)")
        axes[i, j].set_ylabel("Firing rate (Hz)")
        axes[i, j].grid(True)
plt.tight_layout()

population_avg = np.mean(X, axis=(0, 1))  

baseline = np.mean(population_avg[(times >= -800) & (times <= -600)]) 

plt.figure(figsize=(10, 8))
plt.plot(times, population_avg, label="Population Average")
plt.axvline(x=0, color='r', linestyle='--', label='Movement Onset') 
plt.axhline(y=baseline, color='g', linestyle='--', label=f"Baseline ({baseline:.2f} Hz)") 
plt.xlabel("Time (ms)")
plt.ylabel("Average Firing Rate (Hz)")
plt.title("Population Average Firing Rate Over Time")
plt.legend()
plt.grid(True)
plt.show()

threshold_increment = 2 
while True:
    rising_indices = np.where(population_avg > baseline + threshold_increment)[0]
    if rising_indices.size > 0:
        significant_rise_time = times[rising_indices[0]]
        break
    threshold_increment -= 0.5 
    if threshold_increment <= 0:  
        significant_rise_time = None
        break

print(f"Significant rise in population firing rate occurs at approximately {significant_rise_time} ms.")
