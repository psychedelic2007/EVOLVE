import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Custom loss function to avoid log(0)
def custom_binary_crossentropy(y_true, y_pred):
    epsilon = 1e-7
    y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
    return tf.keras.losses.binary_crossentropy(y_true, y_pred)

# Function to recreate the model architecture
def create_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Dense(3, activation='tanh')(inputs)
    x = Dense(8, activation='tanh')(x)
    x = Dense(8, activation='tanh')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Loading the pre-trained model to get its weights
pretrained_model = load_model('<pre-trained model name>')

# Getting the input shape from the model's input attribute
input_shape = pretrained_model.input_shape[1:]
print(f"Inferred input shape: {input_shape}")

# Creating a new model with the same architecture
base_model = create_model(input_shape)

# Copying weights from pretrained model to the new model
base_model.set_weights(pretrained_model.get_weights())

# Freezing all layers except the last one
# You can change this if you want to un-freeze other layers by changing "base_model.layers[:-i]"
for layer in base_model.layers[:-2]:
    layer.trainable = False

# Compiling the model with gradient clipping
optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
base_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Function to load and preprocess data from a CSV file
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    X = data.drop('Target', axis=1)
    y = data['Target']
    
    # Handling NaN values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    return X_scaled, y.values.reshape(-1, 1)  # Reshaping y to match model output

# Custom training loop
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = base_model(x, training=True)
        loss = custom_binary_crossentropy(y, predictions)
    gradients = tape.gradient(loss, base_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, base_model.trainable_variables))
    return loss, predictions

# Directory containing new CSV files
new_data_dir = 'training_files'

# Training loop
num_epochs_per_file = 1000  # Adjust as needed

for epoch in range(num_epochs_per_file):
    print(f"Epoch {epoch + 1}/{num_epochs_per_file}")
    for filename in os.listdir(new_data_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(new_data_dir, filename)
            X, y = load_and_preprocess_data(file_path)
            
            # Converting to TensorFlow tensors
            X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
            y_tensor = tf.convert_to_tensor(y, dtype=tf.float32)
            
            # Train on this file
            loss, predictions = train_step(X_tensor, y_tensor)
            accuracy = tf.keras.metrics.binary_accuracy(y_tensor, predictions)
            mean_accuracy = tf.reduce_mean(accuracy).numpy()
            
            print(f"  File: {filename}, Loss: {np.mean(loss.numpy()):.4f}, Accuracy: {np.mean(mean_accuracy):.4f}")
            
            # Monitoring weights and gradients
            for layer in base_model.layers:
                if layer.trainable_weights:
                    print(f"  Layer: {layer.name}")
                    for w in layer.trainable_weights:
                        print(f"    Weight mean: {tf.reduce_mean(w).numpy():.4f}, std: {tf.math.reduce_std(w).numpy():.4f}")

# Saving the new model
base_model.save('transfer_learned_model.h5', save_format='h5')

# Function to make predictions on a new sequence
def predict_mutation(sequence):
    scaler = StandardScaler()
    sequence_scaled = scaler.fit_transform([sequence])
    prediction = base_model.predict(sequence_scaled)
    return prediction[0][0]
