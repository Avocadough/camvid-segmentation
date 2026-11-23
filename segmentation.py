"""Train one encoder-decoder configuration on the CamVid road-scene dataset."""

import argparse
import csv
from pathlib import Path

import tensorflow as tf
from tensorflow import keras


def image_pairs(root, split):
    images = {p.stem: p for p in (root / split).glob("*.png")}
    masks = {p.stem[:-2]: p for p in (root / f"{split}_labels").glob("*_L.png")}
    if not images or images.keys() != masks.keys():
        missing = sorted(images.keys() - masks.keys())
        extra = sorted(masks.keys() - images.keys())
        raise ValueError(f"{split}: expected matching PNG files; missing masks={missing}, missing images={extra}")
    return [(str(images[key]), str(masks[key])) for key in sorted(images)]


def read_palette(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        colors = [tuple(int(row[key]) for key in ("r", "g", "b")) for row in csv.DictReader(stream)]
    if not colors or len(colors) != len(set(colors)) or any(not 0 <= c <= 255 for rgb in colors for c in rgb):
        raise ValueError("Expected distinct RGB colors in class_dict.csv")
    keys = [r * 65536 + g * 256 + b for r, g, b in colors]
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(tf.constant(keys), tf.range(len(keys))), -1
    )
    return table, len(colors)


def make_dataset(pairs, palette, size, batch_size, training=False):
    images, masks = zip(*pairs)
    data = tf.data.Dataset.from_tensor_slices((list(images), list(masks)))
    if training:
        data = data.shuffle(len(pairs), seed=42)

    def decode(image_path, mask_path):
        image = tf.io.decode_png(tf.io.read_file(image_path), channels=3)
        image = tf.image.resize(image, [size, size]) / 255.0
        mask = tf.io.decode_png(tf.io.read_file(mask_path), channels=3)
        mask = tf.cast(tf.image.resize(mask, [size, size], method="nearest"), tf.int32)
        r, g, b = tf.unstack(mask, axis=-1)
        labels = palette.lookup(r * 65536 + g * 256 + b)
        tf.debugging.assert_non_negative(labels, "Mask contains a color missing from class_dict.csv")
        return image, labels

    return data.map(decode, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(1)


def build_model(size, classes, width, depth, activation):
    if width < 1 or depth < 0 or size < 1 or size % (2 ** depth):
        raise ValueError("Width and size must be positive, and size must be divisible by 2**depth")
    layers = keras.layers

    def conv(inputs, filters):
        x = layers.Conv2D(filters, 3, padding="same")(inputs)
        x = layers.BatchNormalization()(x)
        if activation == "leaky_relu":
            return layers.LeakyReLU()(x)
        return layers.Activation(activation)(x)

    inputs = keras.Input((size, size, 3))
    x = conv(inputs, width)
    skips = []
    for level in range(depth):
        skips.append(x)
        x = conv(layers.MaxPooling2D()(x), width * 2 ** (level + 1))
    for level in reversed(range(depth)):
        x = layers.Concatenate()([layers.UpSampling2D()(x), skips[level]])
        x = conv(x, width * 2 ** level)
    outputs = layers.Conv2D(classes, 1, activation="softmax")(x)
    return keras.Model(inputs, outputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="Directory containing class_dict.csv and the dataset splits")
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--activation", choices=["relu", "leaky_relu", "elu", "gelu", "selu", "swish"], default="relu")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) < 1:
        parser.error("epochs and batch size must be positive")

    keras.utils.set_random_seed(42)
    train = image_pairs(args.data, "train")
    val = image_pairs(args.data, "val")
    overlap = {Path(p).stem for p, _ in train} & {Path(p).stem for p, _ in val}
    if overlap:
        raise ValueError(f"Frame IDs shared by train and val: {sorted(overlap)}")
    palette, classes = read_palette(args.data / "class_dict.csv")
    model = build_model(args.size, classes, args.width, args.depth, args.activation)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(
        make_dataset(train, palette, args.size, args.batch_size, training=True),
        validation_data=make_dataset(val, palette, args.size, args.batch_size),
        epochs=args.epochs,
    )
    print(f"Parameters: {model.count_params():,} | Best validation pixel accuracy: {max(history.history['val_accuracy']):.4f}")


if __name__ == "__main__":
    main()
