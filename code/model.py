from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization


def build_model(num_classes=11):

    model = Sequential([

        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # ===== Flatten + Dense =====
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),

        # ===== Output Layer =====
        Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ===== Test the model =====
if __name__ == "__main__":
    num_classes = 11
    num_classes2 = 11
    model = build_model(num_classes=num_classes)
    model2 = build_model(num_classes=num_classes2)

    model.summary()
