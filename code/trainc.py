import os
import datasetc
from modelc import build_master_stacking_model, NUM_CLASSES_FINAL, NUM_CLASSES_A, NUM_CLASSES_B
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

INPUT_SHAPE = (128, 128, 3)
BATCH_SIZE = 32
STACKING_EPOCHS = 20
MASTER_LEARNING_RATE = 0.001
SAVE_PATH = '../saved_model/best_master_stacking_weights.weights.keras'

OLD_MODEL_A_WEIGHTS_PATH = '../saved_model/best_fine_tuned_weights.keras'
OLD_MODEL_B_WEIGHTS_PATH = '../saved_model/best_fine_tuned_weightse.keras'


def load_and_transfer_weights(master_model, weights_path, num_classes, branch_name):
    base_model_temp = MobileNetV2(
        input_shape=INPUT_SHAPE,
        include_top=False,
        weights=None
    )

    x = GlobalAveragePooling2D(name='gap')(base_model_temp.output)
    x = Dense(256, activation='relu', name='dense_256')(x)
    x = Dropout(0.5, name='dropout')(x)
    outputs = Dense(num_classes, activation='softmax', name='output')(x)

    temp_model = Model(inputs=base_model_temp.input, outputs=outputs)

    temp_model.load_weights(weights_path)

    for layer in temp_model.layers:
        if layer.get_weights():
            try:
                master_layer = master_model.get_layer(layer.name)
                master_layer.set_weights(layer.get_weights())
            except:
                pass

    old_kernel = temp_model.get_layer('dense_256').get_weights()[0]
    master_model.get_layer(f'dense_256_{branch_name}').set_weights([old_kernel])

    old_kernel = temp_model.get_layer('output').get_weights()[0]
    master_model.get_layer(f'pre_softmax_output_{branch_name}').set_weights([old_kernel])

    del temp_model


def load_weights_and_train_master_model():
    data = datasetc.load_combined_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test, class_names = data

    if y_train.shape[1] != NUM_CLASSES_FINAL:
        return

    master_model = build_master_stacking_model()

    load_and_transfer_weights(master_model, OLD_MODEL_A_WEIGHTS_PATH, NUM_CLASSES_A, 'A')
    load_and_transfer_weights(master_model, OLD_MODEL_B_WEIGHTS_PATH, NUM_CLASSES_B, 'B')

    for layer in master_model.layers:
        if 'stacking' not in layer.name and 'output_20_classes' not in layer.name:
            layer.trainable = False
        else:
            layer.trainable = True

    master_model.compile(
        optimizer=Adam(learning_rate=MASTER_LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True),
        ModelCheckpoint(filepath=SAVE_PATH, save_best_only=True, monitor='val_loss'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-7)
    ]

    train_gen = datasetc.training_generator(X_train, y_train, batch_size=BATCH_SIZE)

    master_model.fit(
        train_gen,
        steps_per_epoch=max(1, len(X_train) // BATCH_SIZE),
        epochs=STACKING_EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    master_model.load_weights(SAVE_PATH)
    master_model.evaluate(X_test, y_test)


if __name__ == "__main__":
    load_weights_and_train_master_model()
