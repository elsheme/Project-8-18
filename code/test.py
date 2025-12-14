import os
import numpy as np
import tensorflow as tf
from datasete import load_dataset
from evaluate import evaluate_model, plot_confusion_matrix
from utils import load_model, plot_training_history, show_predictions


def main():
    """الدالة الرئيسية"""

    print("\n" + "=" * 70)
    print("Test and evaluate model")
    print("=" * 70)

    print("\n📥 Load data")
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = load_dataset()
    except Exception as e:
        print(f"error loading: {e}")
        return

    print("\n🤖 تحميل النموذج...")
    model = load_model('trained_model_high_acc')

    if model is None:
        print("❌ لم يتم العثور على النموذج المدرب!")
        print("   تأكد من تشغيل train.py أولاً")
        return

    class_names = [str(i) for i in range(y_test.shape[1])]
    print(f"✓ وجدت {len(class_names)} فئات")

    print("\n📊 تقييم النموذج على بيانات الاختبار...")
    print("-" * 70)

    try:
        accuracy, report, cm = evaluate_model(model, X_test, y_test, class_names)
        print("-" * 70)
    except Exception as e:
        print(f"Error ranking{e}")
        return

    print("\ndraw")
    try:
        plot_confusion_matrix(cm, class_names)
    except Exception as e:
        print(f"Error drawing: {e}")

    print("\nEx. on expected data")
    try:
        show_predictions(model, X_test, y_test, class_names, num=6)
    except Exception as e:
        print(f"error prediction :{e}")


    print(f"Accuracy:{accuracy * 100:.2f}%")
    print(f"Photos No{len(X_test)}")
    print(f"Group No{len(class_names)}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()