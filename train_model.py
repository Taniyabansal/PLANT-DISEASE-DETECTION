import tensorflow as tf
from datasets import load_dataset

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 5
NUM_CLASSES = 38

print("Loading dataset...")

dataset = load_dataset(
    "geraldmc/plantvillage-full",
    revision="v0.1.0",
    streaming=True
)

train_stream = dataset["train"]

print("Dataset loaded!")

# Image + label ko TensorFlow format mein convert karna
def generator():
    for item in train_stream:
        image = item["image"].convert("RGB")
        image = image.resize((IMG_SIZE, IMG_SIZE))

        image = tf.keras.utils.img_to_array(image)
        image = image / 255.0

        label = int(item["class_idx"])

        yield image, label


output_signature = (
    tf.TensorSpec(
        shape=(IMG_SIZE, IMG_SIZE, 3),
        dtype=tf.float32
    ),
    tf.TensorSpec(
        shape=(),
        dtype=tf.int32
    )
)

train_dataset = tf.data.Dataset.from_generator(
    generator,
    output_signature=output_signature
)

train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(
    tf.data.AUTOTUNE
)

# MobileNetV2
print("Creating MobileNetV2 model...")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Starting training...")

model.fit(
    train_dataset,
    steps_per_epoch=500,
    epochs=EPOCHS
)

# Save model
model.save("model/plant_disease_model.keras")

print("================================")
print("MODEL TRAINING COMPLETE!")
print("Model saved successfully!")
print("================================")