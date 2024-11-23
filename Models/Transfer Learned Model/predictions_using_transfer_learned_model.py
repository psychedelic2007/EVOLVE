from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Custom loss func
def custom_binary_crossentropy(y_true, y_pred):
    epsilon = 1e-7
    y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
    return tf.keras.losses.binary_crossentropy(y_true, y_pred)

# MOdle load
transfer_learned_model = load_model('transfer_learned_model.h5', custom_objects={'custom_binary_crossentropy': custom_binary_crossentropy}, compile=False)

# preprocess and predict
def predict_mutation(sequences):
    scaler = StandardScaler()
    sequences_scaled = scaler.fit_transform(sequences)
    predictions = transfer_learned_model.predict(sequences_scaled)
    return predictions

df = pd.read_csv("test.csv")

X_test = df.values
predictions = predict_mutation(X_test)

print(predictions)

plt.figure(figsize=(12,4))
plt.plot(predictions1)
plt.tight_layout()
plt.show()

for i in range(len(predictions)):
    if (predictions1[i] >= 0.2): 
        print("The position is: ", i+1)
