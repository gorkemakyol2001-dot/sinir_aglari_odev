import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import layers, models, applications, optimizers


IMG_SIZE = 128
EPOCHS = 15
BATCH_SIZE = 16
MODEL_NAME = "lies_detector_final.keras"

try:
    from keras.src.legacy.preprocessing.image import ImageDataGenerator
except ImportError:
    from keras.preprocessing.image import ImageDataGenerator

def build_advanced_model():
    base = applications.MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
    base.trainable = True
    for layer in base.layers[:-30]: layer.trainable = False
    model = models.Sequential([
        base, layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    train_gen = ImageDataGenerator(rescale=1./255, rotation_range=15, horizontal_flip=True)
    test_gen = ImageDataGenerator(rescale=1./255)

    t_data = train_gen.flow_from_directory("processed_data/Train", target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='binary')
    v_data = test_gen.flow_from_directory("processed_data/Test", target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='binary')

    model = build_advanced_model()
    history = model.fit(t_data, validation_data=v_data, epochs=EPOCHS)
    model.save(MODEL_NAME)

   
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()
   
    plt.savefig("training_results.png") 
    print("[OK] Grafik 'training_results.png' olarak kaydedildi.")
    plt.show()
    plt.close()