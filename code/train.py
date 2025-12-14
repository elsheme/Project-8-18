import os
import dataset
import model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def train():
    print("Loading dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = dataset.load_dataset()

    num_classes = y_train.shape[1]
    print("Building model...")

    net, base_model = model.build_model(num_classes)

    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=4,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            patience=2,
            factor=0.5,
            min_lr=1e-6
        )
    ]

    print("Training head layers only...")
    net.fit(
        dataset.training_generator(X_train, y_train),
        steps_per_epoch=len(X_train) // 32,
        epochs=15,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )

    print("Fine-tuning last layers...")
    net = model.fine_tune_model(net, base_model, fine_tune_layers=15)

    net.fit(
        dataset.training_generator(X_train, y_train),
        steps_per_epoch=len(X_train) // 32,
        epochs=10,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )

    print("Evaluating on test set...")
    loss, acc = net.evaluate(X_test, y_test)
    print(f"Final Test Accuracy: {acc * 100:.2f}%")

    # Save model
    save_dir = os.path.join(os.path.dirname(__file__), "..", "Saved_model")
    os.makedirs(save_dir, exist_ok=True)
    net.save(os.path.join(save_dir, "trained_model.keras"))
    print("Model saved successfully.")


if __name__ == "__main__":
    train()
