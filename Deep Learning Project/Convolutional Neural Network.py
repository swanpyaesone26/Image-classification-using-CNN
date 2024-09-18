import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the paths to your data
train_dir = r'C:\Users\DELL\Desktop\Deep Learning Project\1. Train Data\Male and Female face train dataset'
test_dir = r'C:\Users\DELL\Desktop\Deep Learning Project\2. Test Data'

# Create an instance of ImageDataGenerator for data rescaling
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Create the generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(32, 32),     # Resize images to 32x32
    batch_size=32,            # Number of images to be yielded from the generator per batch
    class_mode='binary'       # Use 'binary' for binary classification
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(32, 32),     # Resize images to 32x32
    batch_size=32,
    class_mode='binary'
)

# Define the model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))  # Output layer for binary classification

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Use binary crossentropy for binary classification
              metrics=['accuracy'])

# Train the model
history = model.fit(train_generator,
                    epochs=10,
                    validation_data=test_generator)

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

# Evaluate the model using the test generator
test_loss, test_acc = model.evaluate(test_generator, verbose=2)

print(test_acc)
model.save('gender_classification.h5')