import os
import dataset
import model
import tensorflow as tf
import numpy as np

def train_and_save():
    # 1. Load Dataset
    print("Loading dataset...")
    # X_train, X_val, X_test, y_train, y_val, y_test
    data = dataset.load_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test = data
    
    num_classes = y_train.shape[1]
    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # 2. Build Model
    print("Building model...")
    keras_model = model.build_model(num_classes=num_classes)
    
    # 3. Train Model
    BATCH_SIZE = 32
    EPOCHS = 20 
    
    print("Starting training...")
    # Use the generator for training data defined in dataset.py
    train_gen = dataset.training_generator(X_train, y_train, batch_size=BATCH_SIZE)
    
    history = keras_model.fit(
        train_gen,
        steps_per_epoch=max(1, len(X_train) // BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(X_val, y_val)
    )
    
    # 4. Save Model
    # User requested saving in "Saved-model" folder.
    # dataset.py uses BASE_PATH = "../Datasets/", so we are running from 'code' directory.
    # We will create "../Saved-model" to be consistent with the project structure.
    # Function to get the correct base path regardless of where the script is run
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, "..", "Saved_model")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")
        
    model_name = "trained_model.h5"
    model_path = os.path.join(save_dir, model_name)
    keras_model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # 5. Evaluate on test set
    print("Evaluating on test set...")
    loss, acc = keras_model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
    train_and_save()
