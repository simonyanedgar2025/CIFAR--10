import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D, Resizing
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

label_names = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

train_images = train_images[..., np.newaxis]
test_images = test_images[..., np.newaxis]

train_images = np.concatenate([train_images, train_images, train_images], axis=-1)
test_images = np.concatenate([test_images, test_images, test_images], axis=-1)

train_images = train_images.astype("float32")
test_images = test_images.astype("float32")

train_labels_encoded = tf.keras.utils.to_categorical(train_labels, num_classes=10)
test_labels_encoded = tf.keras.utils.to_categorical(test_labels, num_classes=10)

plt.figure(figsize=(12, 12))

for index in range(25):
    plt.subplot(5, 5, index + 1)
    plt.imshow(train_images[index].astype("uint8"))
    plt.title(label_names[train_labels[index]])
    plt.axis("off")

plt.show()

feature_extractor = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

feature_extractor.trainable = False

image_input = Input(shape=(28, 28, 3))

resized_image = Resizing(224, 224)(image_input)

prepared_image = preprocess_input(resized_image)

features = feature_extractor(prepared_image, training=False)

pooled_features = GlobalAveragePooling2D()(features)

hidden_layer = Dense(256, activation="relu")(pooled_features)

regularized_layer = Dropout(0.3)(hidden_layer)

class_output = Dense(10, activation="softmax")(regularized_layer)

fashion_model = Model(inputs=image_input, outputs=class_output)

fashion_model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

fashion_model.summary()

training_result = fashion_model.fit(
    train_images,
    train_labels_encoded,
    validation_split=0.2,
    epochs=5,
    batch_size=64
)

test_loss, test_accuracy = fashion_model.evaluate(
    test_images,
    test_labels_encoded
)

print(f"\nTest accuracy: {test_accuracy:.4f}")

test_predictions = fashion_model.predict(test_images)

plt.figure(figsize=(15, 15))

for index in range(16):
    plt.subplot(4, 4, index + 1)

    plt.imshow(test_images[index].astype("uint8"))

    predicted_label = np.argmax(test_predictions[index])
    actual_label = test_labels[index]

    plt.title(
        f"Pred: {label_names[predicted_label]}\nTrue: {label_names[actual_label]}"
    )

    plt.axis("off")

plt.show()
