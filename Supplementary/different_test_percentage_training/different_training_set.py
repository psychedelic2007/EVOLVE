import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

#Combining CSVs together
def combine_csv_files(data_folder, csv_files):
    df_list = []
    for csv_filename in csv_files:
        df = pd.read_csv(os.path.join(data_folder, csv_filename))
        df_list.append(df)
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df

# TRaining
def train_model(combined_df, neurons, activation_func, hidden_layers, learning_rate, epochs, test_size, model_save_path):
    combined_df = combined_df.dropna()  # dropping NaNs

    x = combined_df.drop('Target', axis=1)
    y = combined_df['Target']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size / 100, random_state=42)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(neurons, input_shape=(x_train.shape[1],), activation=activation_func))
    for _ in range(hidden_layers - 1):
        model.add(tf.keras.layers.Dense(neurons, activation=activation_func))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), verbose=0)

    model.save(model_save_path, save_format='h5')

    # AUROC
    y_pred = model.predict(x_test).ravel()
    auroc = roc_auc_score(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)

    return history.history['accuracy'], history.history['val_accuracy'], history.history['loss'], history.history['val_loss'], auroc, fpr, tpr, thresholds

# Input
data_folder = input("Enter the data folder path: ")
file_list = os.listdir(data_folder)
csv_files = [file for file in file_list if file.endswith(".csv")]

neurons = int(input("Enter the number of neurons in the hidden layers: "))
activation_func = input("Enter the activation function for hidden layers: ")
hidden_layers = int(input("Enter the number of hidden layers: "))
learning_rate = float(input("Enter the learning rate: "))
epochs = int(input("Enter the number of epochs: "))
test_size = float(input("Enter the test size percentage: "))

combined_df = combine_csv_files(data_folder, csv_files)

# Train
model_save_path = os.path.join(data_folder, "final_model_<training_size>.h5")
train_acc, val_acc, train_loss, val_loss, auroc, fpr, tpr, thresholds = train_model(
    combined_df, neurons, activation_func, hidden_layers, learning_rate, epochs, test_size, model_save_path
)

print(f"Final model trained with AUROC: {auroc}")

training_results_df = pd.DataFrame({
    'Epoch': np.arange(1, epochs + 1),
    'Training Accuracy': train_acc,
    'Validation Accuracy': val_acc,
    'Training Loss': train_loss,
    'Validation Loss': val_loss
})
training_results_df.to_csv(os.path.join(data_folder, "training_results_<training_size>.csv"), index=False)

roc_curve_df = pd.DataFrame({
    'FPR': fpr,
    'TPR': tpr,
    'Thresholds': thresholds
})
roc_curve_df.to_csv(os.path.join(data_folder, "roc_curve_data_<training_size>.csv"), index=False)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(training_results_df['Epoch'], training_results_df['Training Accuracy'], label='Training Accuracy')
plt.plot(training_results_df['Epoch'], training_results_df['Validation Accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Learning Curve - Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(training_results_df['Epoch'], training_results_df['Training Loss'], label='Training Loss')
plt.plot(training_results_df['Epoch'], training_results_df['Validation Loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Learning Curve - Loss')
plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUROC = {auroc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
