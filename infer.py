import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix
from keras.models import Model, load_model

materialName = 'Wood'

def preprocess(fileName):
    data = pd.read_csv(fileName, usecols=['accel_x', 'accel_y', 'accel_z', 'rota_x', 'rota_y', 'rota_z'])
    data = np.array(data)
    formatted_data = np.empty((600, 10, 6), dtype=np.float32)

    for i in range(0, 600):
        formatted_data[i] = data[(i * 10):((i + 1) * 10), :]
    
    return formatted_data

model = load_model('model_ext_100-0.86.h5')

X_data = np.empty((600, 10, 6), dtype=np.float32)
X_data = preprocess(materialName + ' 1.csv')
Accel_data = X_data[:, :, :3]
Gyro_data = X_data[:, :, 3:]
count, count_correct = 0, 0
x = np.arange(1, 601, 1)
correctness = np.empty(600, dtype=np.float32)
dur = 0

for i in range(0, 600):
    start = time.time()
    result = model.predict([Accel_data[i:(i + 1), :, :], Gyro_data[i:(i + 1), :, :]])
    end = time.time()
    dur += end - start
    count += 1
    if result[0, 0] == np.amax(result[0]):
        count_correct += 1
    correctness[i] = count_correct / count * 100
    print(result[0])

dur /= count
plt.plot(x, correctness, label=materialName)
plt.xlabel('data inferred')
plt.ylabel('accuracy(%)')
plt.legend(loc='best')
plt.savefig(materialName + '_ext.png')
plt.show()

final_res = count_correct / count * 100
print(materialName + ': ', final_res, '%')
print('Average inference time: ', dur)

# Fabric = 96.67%
# Leather = 94.83%
# Metal = 99.50%
# Paper = 88.00%
# Wood = 87.00%
# 89.082 ms ave.

