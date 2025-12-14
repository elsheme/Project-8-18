import os
import dataset
import model
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


def train_model_1():


    print("=" * 70)
    print(" Training Model 1")
    print("=" * 70)

    print("\n Loading dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = dataset.load_dataset()

    num_classes = y_train.shape[1]
    print(f" Dataset loaded!")
    print(f"   Classes: {num_classes}")
    print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    print("\n Building model...")
    net, base_model = model.build_model(num_classes)
    print(" Model built!")

    y_train_labels = np.argmax(y_train, axis=1)
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train_labels),
        y=y_train_labels
    )
    class_weight_dict = dict(enumerate(class_weights))

    print(f"\n Class weights:")
    for cls, weight in sorted(class_weight_dict.items()):
        print(f"   Class {cls}: {weight:.2f}")

    # ============================================
    # Phase 1: تدريب الـ head layers فقط
    # ============================================
    print("\n" + "=" * 70)
    print("PHASE 1: Training Head Layers (Base Frozen)")
    print("=" * 70)

    base_model.trainable = False

    net.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"Learning rate: 1e-3")
    print(f"Base model FROZEN")

    callbacks_phase1 = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            patience=3,
            factor=0.3,
            min_lr=1e-7,
            verbose=1
        )
    ]

    history_phase1 = net.fit(
        X_train, y_train,
        batch_size=32,
        epochs=12,
        validation_data=(X_val, y_val),
        callbacks=callbacks_phase1,
        class_weight=class_weight_dict,
        verbose=1
    )

    phase1_best = max(history_phase1.history['val_accuracy'])
    print(f"\nPhase 1 Best Val Acc: {phase1_best * 100:.2f}%")

    # ============================================
    # Phase 2: Fine-tuning آخر 80 طبقة
    # ============================================
    print("\n" + "=" * 70)
    print(" PHASE 2: Fine-tuning (Last 80 layers)")
    print("=" * 70)

    # فعّل آخر 80 طبقة
    base_model.trainable = True
    for layer in base_model.layers[:-80]:
        layer.trainable = False

    num_trainable = sum(1 for layer in base_model.layers if layer.trainable)
    print(f" Unfroze {num_trainable} layers")

    net.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f" Learning rate: 1e-4")

    callbacks_phase2 = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            patience=4,
            factor=0.3,
            min_lr=1e-7,
            verbose=1
        )
    ]

    history_phase2 = net.fit(
        X_train, y_train,
        batch_size=32,
        epochs=20,
        validation_data=(X_val, y_val),
        callbacks=callbacks_phase2,
        class_weight=class_weight_dict,
        verbose=1
    )

    phase2_best = max(history_phase2.history['val_accuracy'])
    print(f"\n Phase 2 Best Val Acc: {phase2_best * 100:.2f}%")

    # ============================================
    # التقييم والحفظ
    # ============================================
    print("\n" + "=" * 70)
    print(" Evaluating on test set...")
    print("=" * 70)

    loss, acc = net.evaluate(X_test, y_test, verbose=0)
    print(f"\n Final Test Accuracy: {acc * 100:.2f}%")

    # حفظ النموذج
    print("\n" + "=" * 70)
    print(" Saving model...")
    print("=" * 70)

    save_dir = os.path.join(os.path.dirname(__file__), "..", "Saved_model")
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, "model_1_best.keras")
    net.save(model_path)
    print(f" Model saved to: {model_path}")

    # ملخص
    print("\n" + "=" * 70)
    print(" FINAL SUMMARY")
    print("=" * 70)
    print(f"Phase 1 Best Val Acc: {phase1_best * 100:.2f}%")
    print(f"Phase 2 Best Val Acc: {phase2_best * 100:.2f}%")
    print(f"Test Accuracy: {acc * 100:.2f}%")
    print("=" * 70)

    if acc > 0.85:
        print("\n ممتاز جداً!")
    elif acc > 0.80:
        print("\n ممتاز!")
    elif acc > 0.75:
        print("\n✅ جيد جداً!")

    return net, (loss, acc)


if __name__ == "__main__":
    try:
        train_model_1()
        print("\n✅ Training completed successfully!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()