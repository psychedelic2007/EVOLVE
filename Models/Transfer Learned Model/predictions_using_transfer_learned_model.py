from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Custom loss function to avoid log(0)
def custom_binary_crossentropy(y_true, y_pred):
    epsilon = 1e-7
    y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
    return tf.keras.losses.binary_crossentropy(y_true, y_pred)

# Loading the model with custom objects
transfer_learned_model = load_model('transfer_learned_model.h5', custom_objects={'custom_binary_crossentropy': custom_binary_crossentropy}, compile=False)

# Function to preprocess and predict
def predict_mutation(sequences):
    # Note: Ensure the scaler is fitted on the training data and used here
    scaler = StandardScaler()
    sequences_scaled = scaler.fit_transform(sequences)
    predictions = transfer_learned_model.predict(sequences_scaled)
    return predictions

# Loading and preprocessing the data
df = pd.read_csv("test.csv")

# Assuming 'df' contains only features (no labels), preprocess it accordingly
X_test = df.values
predictions = predict_mutation(X_test)

print(predictions)

plt.figure(figsize=(12,4))
plt.plot(predictions1)
plt.tight_layout()
plt.show()

# To check the position having higher mutation than the cutoff
for i in range(len(predictions)):
    if (predictions1[i] >= 0.2):    #0.2 is the cutoff value. You can change this depending upon how much residue you want above teh cutoff. Suggested value should be varied between 0.2 to 0.6
        print("The position is: ", i+1)
