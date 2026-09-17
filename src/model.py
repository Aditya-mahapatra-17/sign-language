import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import get_data_augmentation

def build_baseline_model(input_shape=(224, 224, 3), num_classes=29):
    """
    Builds a MobileNetV2-based transfer learning model for much higher accuracy.
    """
    data_augmentation = get_data_augmentation()
    
    # Load MobileNetV2 without the top classification layer
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    model = models.Sequential([
        # Data Augmentation layer
        data_augmentation,
        # MobileNetV2 expects pixel values in [-1, 1]
        layers.Rescaling(1./127.5, offset=-1, input_shape=input_shape),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def build_transfer_learning_model(input_shape=(224, 224, 3), num_classes=29):
    """
    Builds a Transfer Learning model using MobileNetV2 as the base.
    Optimized for real-time inference on CPU/Edge.
    """
    # Load MobileNetV2 without the top classification layers
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model to prevent destroying pre-trained weights during initial training
    base_model.trainable = False
    
    # Create the top classification head
    inputs = tf.keras.Input(shape=input_shape)
    
    # Preprocessing expected by MobileNetV2 (scales pixels to [-1, 1])
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    print("=== Phase 4 & 5: CNN Models ===")
    print("Building Baseline...")
    baseline = build_baseline_model()
    print("Building Transfer Learning (MobileNetV2)...")
    tl_model = build_transfer_learning_model()
    tl_model.summary()
    print("Models built successfully.")
