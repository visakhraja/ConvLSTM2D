#load necessary libraries
import tensorflow as tf
# Enable dynamic GPU memory allocation
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

import numpy as np
import pandas as pd
import xarray as xr
from tensorflow.keras.layers import Input, ConvLSTM2D, Reshape, LeakyReLU, SpatialDropout2D, Conv2D, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential, Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.metrics import RootMeanSquaredError
#from global_land_mask import globe
import matplotlib.pyplot as plt
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
import datetime

# Enable mixed precision
from tensorflow.keras.mixed_precision import set_global_policy, LossScaleOptimizer
set_global_policy('mixed_float16')

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# Create a log directory with a timestamp to create unique folders for different runs
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

def create_sequences(data, start_date, end_date, lookback_window, days_after, target_variable):
    extended_start = pd.to_datetime(start_date) - pd.Timedelta(days=lookback_window)
    extended_end = pd.to_datetime(end_date) + pd.Timedelta(days=days_after)

    # Slice the data for extended period
    sliced_data = data.sel(time=slice(extended_start, extended_end))
    X_data=sliced_data[['ascat_smoothed', 'ocean_mask', 'day_of_year']]

    X_sequences = []
    Y_sequences = []
    for i in range(lookback_window, len(sliced_data['time']) - days_after):
        start_idx = i - lookback_window
        end_idx = i + days_after
        # create a sequence from the sliced_data
        X_seq = X_data.isel(time=slice(start_idx, end_idx + 1)).to_array().transpose('time', 'Latitude', 'Longitude', 'variable')
        X_seq_np = X_seq.values 
        Y_seq = sliced_data[target_variable].isel(time=i).values
        # Replace 'target_variable' with your actual target variable

        X_sequences.append(X_seq_np)
        Y_sequences.append(Y_seq)

    return np.array(X_sequences), np.array(Y_sequences)

# Define date ranges and parameters
train_start, train_end = '2014-10-03', '2023-12-22'
validation_start, validation_end = '2013-10-03', '2014-10-02'
test_start, test_end = '2012-10-03', '2013-10-02'
prediction_start, prediction_end = '2007-01-07', '2023-12-22'
lookback_window = 3
days_after = 3
sequence_length = lookback_window + days_after+1
channels = 3
target_variable = 'amsr2_smoothed'  

#loading the datasets
amsr = xr.open_dataset("/p/scratch/share/sivaprasad1/niesel1/EU_DATA/AMSR2_A_25_mask_uncert_cor_2012_2023_smoothed_compressed.nc")
ascat = xr.open_dataset('/p/scratch/share/sivaprasad1/niesel1/EU_DATA/ASCAT_25_2007_2023_smoothed_compressed.nc')
ocean = xr.open_dataset('/p/scratch/share/sivaprasad1/niesel1/EU_DATA/ocean_mask_AMSR.nc')
doy = xr.open_dataset('/p/scratch/share/sivaprasad1/niesel1/EU_DATA/dayofyear_scaled.nc')

#ocean['ocean_mask'] = 1 - ocean['ocean_mask']

combined_dataset = xr.merge([amsr, ascat, ocean, doy])
land_mask = combined_dataset['ocean_mask'] == 1
combined_dataset['amsr2_smoothed'] = combined_dataset['amsr2_smoothed'].where(land_mask, 0)
combined_dataset['ascat_smoothed'] = combined_dataset['ascat_smoothed'].where(land_mask, 0)
combined_dataset['day_of_year'] = combined_dataset['day_of_year'].where(land_mask, 0)
combined_dataset['amsr2_smoothed'] = combined_dataset['amsr2_smoothed'].fillna(0)
combined_dataset['ascat_smoothed'] = combined_dataset['ascat_smoothed'].fillna(0)
combined_dataset['day_of_year'] = combined_dataset['day_of_year'].fillna(0)

# Get the latitude and longitude dimensions in integer
lats = int(combined_dataset['Latitude'].shape[0])
lons = int(combined_dataset['Longitude'].shape[0])

# Create sequences for training, validation, and test sets
X_train, Y_train = create_sequences(combined_dataset, train_start, train_end, lookback_window, days_after, target_variable)
X_val, Y_val = create_sequences(combined_dataset, validation_start, validation_end, lookback_window, days_after, target_variable)
X_test, Y_test = create_sequences(combined_dataset, test_start, test_end, lookback_window, days_after, target_variable)
#X_pred, _ = create_sequences(combined_dataset, prediction_start, prediction_end, lookback_window, days_after, target_variable)


#convert to tensorflow dataset
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(4, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE).cache()
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(4, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE).cache()
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(4, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE).cache()
print("successfully created datasets")


#strategy = tf.distribute.MirroredStrategy()
#with strategy.scope():
def build_model(input_shape):
    model = Sequential([
        ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True, 
                   dropout=0.2, recurrent_dropout=0.2, input_shape=input_shape),
        LeakyReLU(),
        BatchNormalization(),
        ConvLSTM2D(filters=16, kernel_size=(5, 5), padding='same', return_sequences=True, 
                   dropout=0.2, recurrent_dropout=0.2),
        BatchNormalization(),
        LeakyReLU(),
        ConvLSTM2D(filters=1, kernel_size=(5, 5), padding='same', return_sequences=False, 
                   dropout=0.2, recurrent_dropout=0.2),
        BatchNormalization(),
        LeakyReLU(),
        Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same'),
        Conv2D(filters=1, kernel_size=(3, 3), activation='relu', padding='same'),
        #SpatialDropout2D(0.2),
        Reshape((lats, lons)), # reshape to the original grid
    ])
    return model

input_shape = (sequence_length, lats, lons, channels) # (sequence_length, lats, lons, channels)

with strategy.scope():
    model = build_model(input_shape)
    #adam_optimizer = Adam()
    #optimizer=LossScaleOptimizer(adam_optimizer)
    model.compile(optimizer='adam', loss=Huber(), metrics=['mae', RootMeanSquaredError()])

model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-6)
callbacks_list = [early_stopping, reduce_lr]

# set_global_policy('float32')
#save model checkpoint
checkpoint_path = "/p/scratch/share/sivaprasad1/visakh/model_checkpoint_test_1"
checkpoint_callback = ModelCheckpoint(checkpoint_path, save_best_only=True)

#model fitting

#history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=10, callbacks=[callbacks_list, checkpoint_callback], batch_size=16, verbose=1)
history = model.fit(train_dataset, validation_data=val_dataset, epochs=200, callbacks=[callbacks_list,tensorboard_callback,  checkpoint_callback], verbose=1)

# set_global_policy('float32')
#save the model 
model.save('convlstm_model_1')

# #save model weights
model.save_weights('convlstm_model_weights_1')

#model evaluation with test_data
loss, mae, rmse = model.evaluate(test_dataset, verbose=1)

print("Test loss:", loss)
print("Test mae:", mae)
print("Root Mean Squared Error:", rmse)


