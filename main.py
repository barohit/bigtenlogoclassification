import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
import os

image_data = []
labels = []

for filename in os.listdir(r'./train_images'):
    if "wolverine" in filename:
        label = [1, 0, 0]
    elif "buckeye" in filename:
        label = [0, 1, 0]
    elif "spartan" in filename:
        label = [0, 0, 1]

    img = cv2.imread(os.path.join(r'./train_images', filename))
    img = cv2.resize(img, (100, 100))
    img = img.reshape((100, 100, 3))

    image_data.append(img)
    labels.append(label)

X = np.array(image_data)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax'))  # 3 output classes

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32)

desired_class = ""
for filename in os.listdir('./test_images'):
    if "wolverine" in filename:
        label = [1, 0, 0]
        desired_class = "wolverine"
    elif "buckeye" in filename:
        label = [0, 1, 0]
        desired_class = "buckeye"
    elif "spartan" in filename:
        label = [0, 0, 1]
        desired_class = "spartan"
    img = cv2.imread(os.path.join('./test_images', filename))
    img = cv2.resize(img, (100, 100))
    img = img.reshape((1, 100, 100, 3))

    result = model.predict(img)
    print(result)
    if np.argmax(result) == np.argmax(img):
        print("correct prediction")
    else:
        print("incorrect prediction")



