import os
import dataset
import model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

def train():
    print("Loading dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = dataset.load_dataset()

    num_classes = y_train.shape[1]
    print("Building model...")

    # بناء الموديل الأساسي
    net, base_model = model.build_model(num_classes)

    # إعدادات التوقف المبكر وتقليل معدل التعلم
    # زودنا الـ Patience شوية عشان نديله فرصة يتعلم
    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=6,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            patience=3,
            factor=0.2,
            min_lr=1e-7,
            verbose=1
        )
    ]

    print("=== Phase 1: Training head layers only ===")
    # المرحلة الأولى: تدريب سريع للطبقات الأخيرة فقط
    net.fit(
        dataset.training_generator(X_train, y_train),
        steps_per_epoch=len(X_train) // 32,
        epochs=12,  # كفاية 12 هنا لأننا هنكمل تحت
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )

    print("\n=== Phase 2: Fine-tuning last layers (Crucial for >90%) ===")
    # المرحلة الثانية: فك تجميد جزء من الموديل عشان نعلي الدقة
    # بنعمل Unfreeze لأخر 50 طبقة مثلاً عشان يحفظ التفاصيل
    base_model.trainable = True
    for layer in base_model.layers[:-50]:
        layer.trainable = False

    # لازم نعيد الـ Compile بمعدل تعلم واطي جداً عشان الدقة تزيد بنعومة
    net.compile(
        optimizer=Adam(learning_rate=1e-5), # معدل بطيء جداً للدقة العالية
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    net.fit(
        dataset.training_generator(X_train, y_train),
        steps_per_epoch=len(X_train) // 32,
        epochs=20, # سيبه ياخد وقته هنا، الـ EarlyStopping هيوقفه لو مفيش فايدة
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )

    print("\nEvaluating on test set...")
    loss, acc = net.evaluate(X_test, y_test)
    print(f"Final Test Accuracy: {acc * 100:.2f}%")

    save_dir = os.path.join(os.path.dirname(__file__), "..", "Saved_model")
    os.makedirs(save_dir, exist_ok=True)
    net.save(os.path.join(save_dir, "trained_model_high_acc.keras"))
    print("Model saved successfully.")

if __name__ == "__main__":
    train()