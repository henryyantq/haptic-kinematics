import tensorflow as tf
import numpy as np
import pandas as pd
import random
import time 
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model
from keras.layers import Input, Add, Conv1D, LeakyReLU, BatchNormalization, Dense, Flatten, Dropout
from keras.optimizers import Adam

def res_bloc(inputLayer, filts, kSize, padding='causal'):
    l = Conv1D(filters=filts, kernel_size=kSize, padding=padding)(inputLayer)
    inputLayer = BatchNormalization()(inputLayer)
    l = BatchNormalization()(l)
    l = LeakyReLU(alpha=0.2)(l)
    l = Conv1D(filters=filts, kernel_size=kSize, padding=padding)(l)
    l = BatchNormalization()(l)
    l = LeakyReLU(alpha=0.2)(l)
    l = Conv1D(filters=filts, kernel_size=kSize, padding=padding)(l)
    l = BatchNormalization()(l)
    l = Add()([inputLayer, l])
    l = LeakyReLU(alpha=0.2)(l)
    return l

def preprocess(fileName):
    data = pd.read_csv(fileName, usecols=['accel_x', 'accel_y', 'accel_z', 'rota_x', 'rota_y', 'rota_z'])
    data = np.array(data)
    formatted_data = np.empty((600, 10, 6), dtype=np.float32)

    for i in range(0, 600):
        formatted_data[i] = data[(i * 10):((i + 1) * 10), :]
    
    return formatted_data

# Here for Data Preprocessing
X_data = np.empty((3000, 10, 6), dtype=np.float32)
y_data = np.zeros((3000, 5), dtype=int)
X_data_final = np.empty((3000, 10, 6), dtype=np.float32)
y_data_final = np.zeros((3000, 5), dtype=int)

X_data[0:600, :, :] = preprocess('Fabric 1.csv')
X_data[600:1200, :, :] = preprocess('Leather 1.csv')
X_data[1200:1800, :, :] = preprocess('Metal 1.csv')
X_data[1800:2400, :, :] = preprocess('Paper 1.csv')
X_data[2400:3000, :, :] = preprocess('Wood 1.csv')

for i in range(0, 5):
    for j in range(i * 600, (i + 1) * 600):
        y_data[j, 4 - i] = 1

index = np.arange(3000)
random.shuffle(index)

for i in range(0, 3000):
    X_data_final[i] = X_data[index[i]]
    y_data_final[i] = y_data[index[i]]

Accel_data = X_data_final[:, :, :3]
Gyro_data = X_data_final[:, :, 3:]
out_data = y_data_final

split = int(0.7 * 3000)
Accel_train, Gyro_train, y_train = Accel_data[:split], Gyro_data[:split], out_data[:split]
Accel_test, Gyro_test, y_test = Accel_data[split:], Gyro_data[split:], out_data[split:]
# Here for Data Preprocessing

# Model Construction
accel_in = Input((10, 3), dtype='float32')
gyro_in = Input((10, 3), dtype='float32')

# callback = EarlyStopping(monitor='val_accuracy', baseline=0.80, restore_best_weights=True)
filepath = 'model_ext_{epoch:02d}-{val_accuracy:.2f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', save_best_only=True, mode='max', period=1)

la = Conv1D(filters=3, kernel_size=1, padding='causal', data_format='channels_last')(accel_in)
la = LeakyReLU(alpha=0.2)(la)
la = Conv1D(filters=16, kernel_size=7, padding='causal', data_format='channels_last')(la)
la = LeakyReLU(alpha=0.2)(la)
la = Conv1D(filters=32, kernel_size=5, padding='causal', data_format='channels_last')(la)
la = LeakyReLU(alpha=0.2)(la)
la = res_bloc(la, 32, 5)
la = Conv1D(filters=16, kernel_size=3, padding='causal', data_format='channels_last')(la)
la = LeakyReLU(alpha=0.2)(la)
la = res_bloc(la, 16, 3)
la = Conv1D(filters=8, kernel_size=1, padding='valid', data_format='channels_last')(la)
la = LeakyReLU(alpha=0.2)(la)
la = Conv1D(filters=1, kernel_size=1, padding='valid', data_format='channels_last')(la)
la = LeakyReLU(alpha=0.2)(la)
la = Flatten()(la)

lg = Conv1D(filters=3, kernel_size=1, padding='causal', data_format='channels_last')(gyro_in)
lg = LeakyReLU(alpha=0.2)(lg)
lg = Conv1D(filters=16, kernel_size=7, padding='causal', data_format='channels_last')(lg)
lg = LeakyReLU(alpha=0.2)(lg)
lg = Conv1D(filters=32, kernel_size=5, padding='causal', data_format='channels_last')(lg)
lg = LeakyReLU(alpha=0.2)(lg)
lg = res_bloc(lg, 32, 5)
lg = Conv1D(filters=16, kernel_size=3, padding='causal', data_format='channels_last')(lg)
lg = LeakyReLU(alpha=0.2)(lg)
lg = res_bloc(lg, 16, 3)
lg = Conv1D(filters=8, kernel_size=1, padding='valid', data_format='channels_last')(lg)
lg = LeakyReLU(alpha=0.2)(lg)
lg = Conv1D(filters=1, kernel_size=1, padding='valid', data_format='channels_last')(lg)
lg = LeakyReLU(alpha=0.2)(lg)
lg = Flatten()(lg)

l = Add()([la, lg])

l = Dense(24, activation='relu')(l)
l = Dense(96, activation='relu')(l)
l = Dropout(rate=0.2)(l)
l = Dense(96, activation='relu')(l)
l = Dropout(rate=0.2)(l)
l = Dense(24, activation='relu')(l)
final_out = Dense(5, activation='softmax')(l)

hapnet = Model([accel_in, gyro_in], final_out, name='hapnet')
plot_model(hapnet, to_file='HapNet_structure.png', show_shapes=True)

adam = Adam(learning_rate=5e-4)
'''
hapnet.compile(
    optimizer=adam,
    loss='categorical_crossentropy',
    metrics=['accuracy']
    )

history = hapnet.fit(
    [Accel_train, Gyro_train],
    y_train,
    epochs=120,
    batch_size=16,
    validation_data=([Accel_test, Gyro_test], y_test),
    callbacks=[checkpoint]
    )

# hapnet.evaluate([Accel_test, Gyro_test], y_test) 
# hapnet.save('HapNet_all.h5')

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='best')
plt.savefig('loss_all.png')
plt.close()

plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc='best')
plt.savefig('accuracy_all.png')
'''

