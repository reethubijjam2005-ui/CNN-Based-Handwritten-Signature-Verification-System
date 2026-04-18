import os
import kagglehub
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

path = kagglehub.dataset_download("ishanikathuria/handwritten-signature-datasets")
print("Dataset path:", path)

IMG_SIZE = 128

data = []
labels = []

# Dataset structure handling
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(".png") or file.endswith(".jpg"):
            img_path = os.path.join(root, file)

            try:
                img = cv2.imread(img_path)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img / 255.0

                data.append(img)

                # Labeling (based on folder name)
                if "forged" in root.lower():
                    labels.append(1)
                else:
                    labels.append(0)

            except:
                pass

data = np.array(data)
labels = np.array(labels)

print("Total images:", len(data))

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=32
)

model.save("signature_model.h5")

plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.legend()
plt.title("Accuracy")
plt.show()

loss, acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", acc)
