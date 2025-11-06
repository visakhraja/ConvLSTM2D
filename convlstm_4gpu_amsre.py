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
    X_data=sliced_data[['amsre_smoothed', 'ocean_land_mask', 'day_of_sin', 'day_of_cos']]

    X_sequences = []
    Y_sequences = []
    for i in range(lookback_window, len(sliced_data['time']) - days_after):
        start_idx = i - lookback_window
        end_idx = i + days_after
        # create a sequence from the sliced_data
        X_seq_before = X_data.isel(time=slice(start_idx, i)).to_array().transpose('time', 'Latitude', 'Longitude', 'variable')
        # Create a sequence for days after the current day
        X_seq_after = X_data.isel(time=slice(i + 1, end_idx + 1)).to_array().transpose('time', 'Latitude', 'Longitude', 'variable')
        # Concatenate sequences from before and after the current day
        X_seq_np = np.concatenate((X_seq_before, X_seq_after), axis=0)
        Y_seq = sliced_data[target_variable].isel(time=i).values
        # Replace 'target_variable' with your actual target variable

        X_sequences.append(X_seq_np)
        Y_sequences.append(Y_seq)

    return np.array(X_sequences), np.array(Y_sequences)

# Define date ranges and parameters
train_start, train_end = '2009-01-07', '2011-12-28'
validation_start, validation_end = '2007-12-04', '2009-01-03'
test_start, test_end = '2007-01-07', '2007-11-30'
#prediction_start, prediction_end = '2007-01-07', '2023-12-22'
lookback_window = 3
days_after = 3
sequence_length = lookback_window + days_after
channels = 4
target_variable = 'AMSR_predicted'  

#loading the datasets
combined_dataset=xr.open_dataset("/p/scratch/share/sivaprasad1/niesel1/EU_DATA/combined_dataset_amsre_2002_2011_20250814.nc")

# amsr = xr.open_dataset("/p/scratch/share/sivaprasad1/niesel1/EU_DATA/AMSRE_A_25_mask_uncert_cor_2002_2011_smoothed_new.nc")
# ascat = xr.open_dataset('/p/scratch/share/sivaprasad1/niesel1/EU_DATA/predictions_2007_2023_new.nc')
# ocean = xr.open_dataset('/p/scratch/share/sivaprasad1/niesel1/EU_DATA/ocean_land_mask_2003_2011_new.nc')
# doy = xr.open_dataset('/p/scratch/share/sivaprasad1/niesel1/EU_DATA/dayofyear_2003_2011.nc')


# combined_dataset = xr.merge([amsr, ascat, ocean, doy])
# #create cyclic encoding of day of year
# combined_dataset['day_of_sin'] = np.sin(2*np.pi*combined_dataset['dayofyear']/365)
# combined_dataset['day_of_cos'] = np.cos(2*np.pi*combined_dataset['dayofyear']/365)


# land_mask = combined_dataset['ocean_land_mask'] == 1
# combined_dataset['amsre_smoothed'] = combined_dataset['amsre_smoothed'].where(land_mask, 0)
# combined_dataset['AMSR_predicted'] = combined_dataset['AMSR_predicted'].where(land_mask, 0)
# combined_dataset['day_of_sin'] = combined_dataset['day_of_sin'].where(land_mask, 0)
# combined_dataset['day_of_cos'] = combined_dataset['day_of_cos'].where(land_mask, 0)
# #combined_dataset['day_of_year'] = combined_dataset['day_of_year'].where(land_mask, 0)
# combined_dataset['amsre_smoothed'] = combined_dataset['amsre_smoothed'].fillna(0)
# combined_dataset['AMSR_predicted'] = combined_dataset['AMSR_predicted'].fillna(0)
# #fill all values above 1 with 0 in amsr2_smoothed
# combined_dataset['amsre_smoothed'] = combined_dataset['amsre_smoothed'].where(combined_dataset['amsre_smoothed'] < 1, 0)

# combined_dataset['day_of_sin'] = combined_dataset['day_of_sin'].fillna(0)
# combined_dataset['day_of_cos'] = combined_dataset['day_of_cos'].fillna(0)


# Get the latitude and longitude dimensions in integer
lats = int(combined_dataset['Latitude'].shape[0])
lons = int(combined_dataset['Longitude'].shape[0])

# Create sequences for training, validation, and test sets
X_train, Y_train = create_sequences(combined_dataset, train_start, train_end, lookback_window, days_after, target_variable)
X_val, Y_val = create_sequences(combined_dataset, validation_start, validation_end, lookback_window, days_after, target_variable)
X_test, Y_test = create_sequences(combined_dataset, test_start, test_end, lookback_window, days_after, target_variable)
#X_pred, _ = create_sequences(combined_dataset, prediction_start, prediction_end, lookback_window, days_after, target_variable)
print("successfully created sequences")

X_train = X_train.astype('float16')
Y_train = Y_train.astype('float16')
X_val = X_val.astype('float16')
Y_val = Y_val.astype('float16')
X_test = X_test.astype('float16')
Y_test = Y_test.astype('float16')

X_train = tf.convert_to_tensor(X_train, dtype=tf.float16)
Y_train = tf.convert_to_tensor(Y_train , dtype=tf.float16)
X_val = tf.convert_to_tensor(X_val, dtype=tf.float16)
Y_val = tf.convert_to_tensor(Y_val, dtype=tf.float16)
X_test = tf.convert_to_tensor(X_test, dtype=tf.float16)
Y_test = tf.convert_to_tensor(Y_test, dtype=tf.float16)

# #convert to tensorflow dataset
# train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(4, drop_remainder=True)#.prefetch(tf.data.AUTOTUNE)
# val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(4, drop_remainder=True)#.prefetch(tf.data.AUTOTUNE)
# test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(4, drop_remainder=True)#.prefetch(tf.data.AUTOTUNE)
print("successfully created datasets")


input_shape = (sequence_length, lats, lons, channels) # (sequence_length, lats, lons, channels)
#strategy = tf.distribute.MirroredStrategy()
#with strategy.scope():
def build_model(pretrained_model):
    pretrained_layers = pretrained_model.layers[:4]  # Extract the first four layers
    new_layers = [
        ConvLSTM2D(filters=64, kernel_size=(5, 5), padding='same', return_sequences=True, 
                   dropout=0.2, recurrent_dropout=0.2, name="new_conv_lstm_1"),
        BatchNormalization(name="new_batch_norm_1"),
        ConvLSTM2D(filters=16, kernel_size=(3, 3), padding='same', return_sequences=False, 
                   dropout=0.2, recurrent_dropout=0.2, name="new_conv_lstm_2"),
        BatchNormalization(name="new_batch_norm_2"),        
        Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same', name="new_conv2d_1"),
        #SpatialDropout2D(0.2, name="new_spatial_dropout"),
        Conv2D(filters=1, kernel_size=(3, 3), activation='relu', padding='same', name="new_conv2d_2"),
        Reshape((lats, lons), name="new_reshape"),  # reshape to the original grid
    ]
    model = Sequential(pretrained_layers + new_layers)
    return model

with strategy.scope():
    pretrained_model = tf.keras.models.load_model('/p/scratch/share/sivaprasad1/visakh/code/convlstm_model_20241106_rev')
    pretrained_model.trainable = False
    model = build_model(pretrained_model)
    #adam_optimizer = Adam()
    #optimizer=LossScaleOptimizer(adam_optimizer)
    model.compile(optimizer='adam', loss=Huber(), metrics=['mae', RootMeanSquaredError()])

model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-6)
callbacks_list = [early_stopping, reduce_lr]

# set_global_policy('float32')
#save model checkpoint
checkpoint_path = "/p/scratch/share/sivaprasad1/visakh/model_checkpoint_amsre_1"
checkpoint_callback = ModelCheckpoint(checkpoint_path, save_best_only=True)

# #model fitting

history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=200, callbacks=[callbacks_list, tensorboard_callback, checkpoint_callback], batch_size=32, verbose=1)
#history = model.fit(train_dataset, validation_data=val_dataset, epochs=50, callbacks=[callbacks_list,tensorboard_callback,  checkpoint_callback], verbose=1)

# # set_global_policy('float32')
# #save the model 
model.save('convlstm_model_amsre_20241106_rev')

# # #save model weights
model.save_weights('convlstm_model_weights_amsre_20241106')

#model evaluation with test_data
loss, mae, rmse = model.evaluate(X_test, Y_test, verbose=1)

print("Test loss:", loss)
print("Test mae:", mae)
print("Root Mean Squared Error:", rmse)


