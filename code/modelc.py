import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout, Concatenate
from tensorflow.keras.applications import MobileNetV2

NUM_CLASSES_A = 11
NUM_CLASSES_B = 9
NUM_CLASSES_FINAL = NUM_CLASSES_A + NUM_CLASSES_B
INPUT_SHAPE = (128, 128, 3)


def create_old_model_branch(input_tensor, num_classes, branch_name):
    x = GlobalAveragePooling2D(name=f'gap_{branch_name}')(input_tensor)
    # FIX: Adding use_bias=False to match the expected weight count during transfer
    x = Dense(256, activation='relu', use_bias=False, name=f'dense_256_{branch_name}')(x)
    x = Dropout(0.5, name=f'dropout_{branch_name}')(x)
    # FIX: Adding use_bias=False to match the expected weight count during transfer
    logits = Dense(num_classes, activation=None, use_bias=False, name=f'pre_softmax_output_{branch_name}')(x)
    return logits


def build_master_stacking_model():
    img_input = Input(shape=INPUT_SHAPE, name='image_input')

    base_model = MobileNetV2(
        input_tensor=img_input,
        include_top=False,
        weights='imagenet'
    )

    logits_A = create_old_model_branch(base_model.output, NUM_CLASSES_A, 'A')

    logits_B = create_old_model_branch(base_model.output, NUM_CLASSES_B, 'B')

    merged_logits = Concatenate(name='merged_20_vector')([logits_A, logits_B])

    stack_x = Dense(32, activation='relu', name='stacking_dense')(merged_logits)

    final_output = Dense(NUM_CLASSES_FINAL, activation='softmax', name='output_20_classes')(stack_x)

    master_model = Model(inputs=img_input, outputs=final_output)

    return master_model


if __name__ == "__main__":
    model = build_master_stacking_model()
    model.summary()