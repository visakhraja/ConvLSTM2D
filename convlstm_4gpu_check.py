# Load necessary libraries
import tensorflow as tf
# Enable dynamic GPU memory allocation
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

import numpy as np
import pandas as pd
import xarray as xr
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import datetime

# Enable mixed precision
from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')

# Setup distributed strategy
strategy = tf.distribute.MirroredStrategy()
print(f'Number of devices: {strategy.num_replicas_in_sync}')

# Create log directory for TensorBoard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def create_sequences(data, start_date, end_date, lookback_window, days_after, target_variable):
    extended_start = pd.to_datetime(start_date) - pd.Timedelta(days=lookback_window)
    extended_end = pd.to_datetime(end_date) + pd.Timedelta(days=days_after)

    # Slice the data for extended period
    sliced_data = data.sel(time=slice(extended_start, extended_end))
    X_data = sliced_data[['ascat_smoothed', 'ocean_land_mask', 'day_of_sin', 'day_of_cos']]

    X_sequences = []
    Y_sequences = []
    for i in range(lookback_window, len(sliced_data['time']) - days_after):
        start_idx = i - lookback_window
        end_idx = i + days_after
        
        # Create sequence from before current day
        X_seq_before = X_data.isel(time=slice(start_idx, i)).to_array().transpose('time', 'Latitude', 'Longitude', 'variable')
        # Create sequence for days after current day
        X_seq_after = X_data.isel(time=slice(i + 1, end_idx + 1)).to_array().transpose('time', 'Latitude', 'Longitude', 'variable')
        # Concatenate sequences
        X_seq_np = np.concatenate((X_seq_before, X_seq_after), axis=0)
        Y_seq = sliced_data[target_variable].isel(time=i).values

        X_sequences.append(X_seq_np)
        Y_sequences.append(Y_seq)

    return np.array(X_sequences), np.array(Y_sequences)

# Define parameters
train_start, train_end = '2015-10-03', '2023-12-22'
validation_start, validation_end = '2013-10-03', '2015-10-02'
test_start, test_end = '2012-10-03', '2013-10-02'
lookback_window = 3
days_after = 3
sequence_length = lookback_window + days_after
channels = 4
target_variable = 'amsr2_smoothed'

# Load dataset
print("Loading dataset...")
combined_dataset = xr.open_dataset("/p/scratch/share/sivaprasad1/niesel1/EU_DATA/combined_data_AMSR2_ASCAT_full_20250727.nc")

# Clip values above 0.8
combined_dataset['amsr2_smoothed'] = combined_dataset['amsr2_smoothed'].where(combined_dataset['amsr2_smoothed'] <= 0.8, 0.8)
combined_dataset['ascat_smoothed'] = combined_dataset['ascat_smoothed'].where(combined_dataset['ascat_smoothed'] <= 0.8, 0.8)

# Get spatial dimensions
lats = int(combined_dataset['Latitude'].shape[0])
lons = int(combined_dataset['Longitude'].shape[0])

print("Creating sequences...")
# Create sequences for training, validation, and test sets
X_train, Y_train = create_sequences(combined_dataset, train_start, train_end, lookback_window, days_after, target_variable)
X_val, Y_val = create_sequences(combined_dataset, validation_start, validation_end, lookback_window, days_after, target_variable)
X_test, Y_test = create_sequences(combined_dataset, test_start, test_end, lookback_window, days_after, target_variable)
print("Successfully created sequences")

# Convert to float16 and tensors
print("Converting to tensors...")
X_train = tf.convert_to_tensor(X_train.astype('float16'), dtype=tf.float16)
Y_train = tf.convert_to_tensor(Y_train.astype('float16'), dtype=tf.float16)
X_val = tf.convert_to_tensor(X_val.astype('float16'), dtype=tf.float16)
Y_val = tf.convert_to_tensor(Y_val.astype('float16'), dtype=tf.float16)
X_test = tf.convert_to_tensor(X_test.astype('float16'), dtype=tf.float16)
Y_test = tf.convert_to_tensor(Y_test.astype('float16'), dtype=tf.float16)
print("Successfully converted to tensors")

# ===== TRANSFER LEARNING WITH FINE-TUNING =====

def setup_fine_tuning(pretrained_path, freeze_layers=3):
    """
    Load pretrained model and setup for fine-tuning
    
    Args:
        pretrained_path: Path to the pretrained model
        freeze_layers: Number of layers to freeze from the beginning (default: 3)
    """
    with strategy.scope():
        print(f"Loading pretrained model from: {pretrained_path}")
        model = tf.keras.models.load_model(pretrained_path)
        
        print("Original model architecture:")
        model.summary()
        
        print(f"\nSetting up fine-tuning (freezing first {freeze_layers} layers):")
        
        # Freeze the first few layers, keep the rest trainable
        for i, layer in enumerate(model.layers):
            if i < freeze_layers:
                layer.trainable = False
                print(f"  ❄️  Frozen: Layer {i} - {layer.name}")
            else:
                layer.trainable = True
                print(f"  🔥 Trainable: Layer {i} - {layer.name}")
        
        # Compile with lower learning rate for fine-tuning
        learning_rate = 1e-4  # Much lower than normal training
        print(f"\nCompiling model with learning rate: {learning_rate}")
        
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=Huber(), 
            metrics=['mae', RootMeanSquaredError()]
        )
        
        print("✅ Model ready for fine-tuning!")
        return model

# Load and setup the pretrained model
pretrained_model_path = '/p/scratch/share/sivaprasad1/visakh/code/convlstm_model_20241106_rev'

print("=" * 60)
print("SETTING UP TRANSFER LEARNING")
print("=" * 60)

model = setup_fine_tuning(pretrained_model_path, freeze_layers=3)

# ===== SETUP CALLBACKS =====

print("\nSetting up training callbacks...")

# Early stopping - more aggressive for fine-tuning
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,  # Stop if no improvement for 8 epochs
    min_delta=1e-5,  # Minimum change to qualify as improvement
    restore_best_weights=True,
    verbose=1
)

# Learning rate reduction
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,  # Cut learning rate in half
    patience=10,  # Wait 10 epochs before reducing
    min_lr=1e-6,
    verbose=1
)

# Model checkpoint
checkpoint_path = "/p/scratch/share/sivaprasad1/visakh/model_checkpoint_fine_tuned_20250727"
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

# TensorBoard
tensorboard_callback = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    write_graph=True
)

# Combine all callbacks
callbacks = [
    early_stopping,
    reduce_lr,
    checkpoint_callback,
    tensorboard_callback
]

print("✅ Callbacks configured")

# ===== TRAINING =====

print("\n" + "=" * 60)
print("STARTING FINE-TUNING")
print("=" * 60)

# Training parameters
EPOCHS = 100  # Start with fewer epochs for fine-tuning
BATCH_SIZE = 24  # Larger batch size for stability

print(f"Training parameters:")
print(f"  - Epochs: {EPOCHS}")
print(f"  - Batch size: {BATCH_SIZE}")
print(f"  - Training samples: {X_train.shape[0]}")
print(f"  - Validation samples: {X_val.shape[0]}")

# Start training
history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

print("🎉 Training completed!")

# ===== SAVE MODEL =====

print("\nSaving fine-tuned model...")
model.save('convlstm_model_fine_tuned_20250727')
model.save_weights('convlstm_model_weights_fine_tuned_20250727')
print("✅ Model saved")

# ===== EVALUATION =====

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

# Evaluate on test set
print("Evaluating on test set...")
test_loss, test_mae, test_rmse = model.evaluate(X_test, Y_test, verbose=1)

print(f"\n📊 Test Results:")
print(f"  - Loss: {test_loss:.6f}")
print(f"  - MAE: {test_mae:.6f}")
print(f"  - RMSE: {test_rmse:.6f}")

# Training summary
epochs_trained = len(history.history['loss'])
best_val_loss = min(history.history['val_loss'])
final_val_loss = history.history['val_loss'][-1]

print(f"\n📈 Training Summary:")
print(f"  - Epochs trained: {epochs_trained}/{EPOCHS}")
print(f"  - Best validation loss: {best_val_loss:.6f}")
print(f"  - Final validation loss: {final_val_loss:.6f}")

# ===== VISUALIZATION =====

print("\nGenerating training plots...")

plt.figure(figsize=(15, 5))

# Loss plot
plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Loss Curves', fontsize=12, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# MAE plot
plt.subplot(1, 3, 2)
plt.plot(history.history['mae'], label='Training MAE', linewidth=2)
plt.plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
plt.title('MAE Curves', fontsize=12, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.grid(True, alpha=0.3)

# Learning rate plot (if available)
plt.subplot(1, 3, 3)
if 'lr' in history.history:
    plt.plot(history.history['lr'], linewidth=2, color='red')
    plt.title('Learning Rate', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
else:
    plt.text(0.5, 0.5, 'Learning Rate\nHistory Not Available', 
             ha='center', va='center', transform=plt.gca().transAxes,
             fontsize=12)
    plt.title('Learning Rate', fontsize=12, fontweight='bold')

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fine_tuning_results_20250727.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ Training visualization saved")
print("\n🎉 Fine-tuning process completed successfully!")

# Print final summary
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"✅ Model fine-tuned successfully")
print(f"✅ Training completed in {epochs_trained} epochs")
print(f"✅ Best validation loss: {best_val_loss:.6f}")
print(f"✅ Test RMSE: {test_rmse:.6f}")
print(f"✅ Model saved as: convlstm_model_fine_tuned_20250727")
print("=" * 60)