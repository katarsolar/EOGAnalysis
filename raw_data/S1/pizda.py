import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from numpy import mean

matplotlib.use('TkAgg')


target_data = scipy.io.loadmat('Target_GA_stream.mat')
time_data = scipy.io.loadmat('ControlSignal.mat')
mat_data = scipy.io.loadmat('EOG.mat')

target_df = pd.DataFrame(target_data['Target_GA_stream'])
time_df = pd.DataFrame(time_data['ControlSignal'])
eog_df = pd.DataFrame(mat_data['EOG'])

E_1 = eog_df.iloc[0, :10000]
E_1_mean = E_1.mean()
E_1_normalized = E_1 / E_1_mean

E_2 = eog_df.iloc[1, :10000]
E_2_mean = E_1.mean()
E_2_normalized = E_2 / E_2_mean

target_df_mean = target_df.mean(axis=1)
target_df_normalized = target_df.div(target_df_mean, axis=0)

plt.figure(figsize=(10, 8))

plt.plot(target_df_normalized.iloc[0, :10000], label='Target Gaze Angle')
plt.plot(E_1_normalized, label='EOG Signal E1')
plt.plot(E_2_normalized, label='EOG Signal E2')
plt.plot(time_df.iloc[0, :10000], color='red', label='Control Signal')

plt.legend()
plt.show()

# if 'EOG' in mat_data:
#     data = mat_data['EOG']
#     df = pd.DataFrame(data)
#     for i in range(4):
#         plt.subplot(4, 1, i + 1)
#         plt.plot(df.iloc[i,:50])
#         plt.title(f'Time series {i}')
#         plt.xlabel('Time')
#         plt.ylabel('Signal value')
#     plt.tight_layout()
#     plt.show()