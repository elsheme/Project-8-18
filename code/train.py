import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from datasete import load_dataset, training_generator
from model import build_model
import numpy as np

BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-4

MODEL_CHECKPOINT_PATH = './best_egypt_9_classes_v2.keras'


def train_enhanced_model():
    X_train, X_val, X_test, Y_train, Y_val, Y_test = load_dataset()

    NUM_CLASSES = Y_train.shape[1]

    if len(X_train) == 0:
        print("FATAL ERROR: Training data is empty.")
        return

    print(f"Total training samples: {len(X_train)}")
    print(f"Number of classes detected: {NUM_CLASSES}")
    print("Building and compiling the enhanced 9-class model...")

    master_model = build_model()

    for layer in master_model.layers:
        if layer.name.startswith('mobilenetv2'):
            layer.trainable = False

    master_model.compile(
        optimizer=Adam(learning_rate=LR),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        ModelCheckpoint(MODEL_CHECKPOINT_PATH, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max'),
        EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, mode='min', min_lr=1e-7)
    ]

    print("Starting training (Phase 1: Frozen MobileNetV2)...")
    history = master_model.fit(
        training_generator(X_train, Y_train, BATCH_SIZE),
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=10,
        validation_data=(X_val, Y_val),
        callbacks=callbacks
    )

    print("\nStarting Fine-Tuning (Phase 2: Unfreezing lower layers)...")

    base_mobilenet = None
    for layer in master_model.layers:
        if isinstance(layer, tf.keras.Model) and layer.name.startswith('mobilenetv2'):
            base_mobilenet = layer
            break

    if base_mobilenet:
        print(f"Found base model: {base_mobilenet.name}. Preparing for fine-tuning...")
        base_mobilenet.trainable = True

        for layer in base_mobilenet.layers[:100]:
            layer.trainable = False
        for layer in base_mobilenet.layers[100:]:
            layer.trainable = True

        print(f"Unfroze layers from index 100 to {len(base_mobilenet.layers) - 1} for fine-tuning.")
    else:
        print("Warning: MobileNetV2 sub-model not found. Skipping fine-tuning setup.")

    master_model.compile(
        optimizer=Adam(learning_rate=LR / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history_fine_tune = master_model.fit(
        training_generator(X_train, Y_train, BATCH_SIZE),
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=EPOCHS,
        initial_epoch=history.epoch[-1] if history.epoch else 10,
        validation_data=(X_val, Y_val),
        callbacks=callbacks
    )

    print("\nFinal Evaluation...")
    loss, acc = master_model.evaluate(X_test, Y_test, verbose=0)
    print(f"Test Accuracy ({NUM_CLASSES} Classes): {acc * 100:.2f}%")


if __name__ == "__main__":
    train_enhanced_model()