import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Define Model Architecture
model = Sequential([
    # Block 1
    Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Block 2
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Block 3
    Conv2D(512, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Fully Connected Layer
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax') # 7 emotions
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 2. Setup Data Generators (Assuming you downloaded FER2013 and unzipped it to '/content/data/')
# NOTE: If you don't want to train from scratch, skip to the "Shortcut" below!
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=10, zoom_range=0.1, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    '/content/data/train', target_size=(48, 48), color_mode="grayscale", batch_size=64, class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    '/content/data/test', target_size=(48, 48), color_mode="grayscale", batch_size=64, class_mode='categorical')

# 3. Train the Model
model.fit(train_generator, epochs=30, validation_data=validation_generator)

# 4. Save and Download the Model
model.save('emotion_model.h5')
from google.colab import files
files.download('emotion_model.h5')