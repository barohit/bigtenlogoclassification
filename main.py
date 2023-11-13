import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
import os

# Step 1: Data Preprocessing
image_data = []
labels = []

for filename in os.listdir('./images'):
    if "wolverine" in filename:
        label = [1, 0, 0]
    elif "buckeyes" in filename:
        label = [0, 1, 0]
    elif "spartans" in filename:
        label = [0, 0, 1]

    img = cv2.imread(os.path.join('./images', filename))
    img = cv2.resize(img, (100, 100))
    img = img.flatten()

    image_data.append(img)
    labels.append(label)

X = np.array(image_data)
y = np.array(labels)

# Step 2: Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Model Building
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax'))  # 3 output classes

# Step 4: Compile the Model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32)


