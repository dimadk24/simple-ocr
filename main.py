from typing import List, Dict

import numpy as np
import tensorflow as tf
from numpy import argmax, rint
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

alphabet = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
char_to_int: Dict[str, int] = dict((c, i) for i, c in enumerate(alphabet))
int_to_char: Dict[int, str] = dict((i, c) for i, c in enumerate(alphabet))


def encode_char(char: str) -> List[int]:
    integer_encoded = char_to_int[char]
    letter = [0 for _ in range(len(alphabet))]
    letter[integer_encoded] = 1
    return letter


def decode_char(array: List[int]) -> str:
    return int_to_char[argmax(array)]


img_width = 28
img_height = 28


def encode_single_sample(img_path):
    # 1. Read image
    img = tf.io.read_file(img_path)
    # 2. Decode and convert to grayscale
    img = tf.io.decode_png(img, channels=1)
    # 3. Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    # 4. Resize to the desired size
    img = tf.image.resize(img, [img_height, img_width])
    return img[0][0]


x_input = np.array([
    encode_single_sample('./data/training/letter/img_1.png'),
    encode_single_sample('./data/training/letter/img_2.png'),
    encode_single_sample('./data/training/letter/img_3.png'),
    encode_single_sample('./data/training/letter/img_4.png'),
    encode_single_sample('./data/training/letter/img_5.png'),
    encode_single_sample('./data/training/letter/img_6.png'),
])
y_input = np.array([
    encode_char('р'),
    encode_char('р'),
    encode_char('р'),
    encode_char('р'),
    encode_char('р'),
    encode_char('р'),
], dtype='float32')

model = Sequential()
model.add(Dense(units=32, activation="sigmoid", input_dim=x_input.shape[1], kernel_initializer='random_normal'))
model.add(Dense(units=len(alphabet), kernel_initializer='random_normal'))

model.compile(loss='mean_squared_error', optimizer='sgd')

model.summary()

model.fit(x_input, y_input, epochs=100, batch_size=32)

result = model.predict(np.array([
    encode_single_sample('./data/testing/letter/img.png'),
]))
print('letter')
print('int result:', result[0])
print('char result:', decode_char(result[0]))

x_input = np.array([
    encode_single_sample('./data/training/number/img_1.jpeg'),
    encode_single_sample('./data/training/number/img_2.png'),
    encode_single_sample('./data/training/number/img_3.png'),
])
y_input = np.array([
    18,
    18,
    18
], dtype='int')

model = Sequential()
model.add(Dense(units=32, activation="sigmoid", input_dim=x_input.shape[1], kernel_initializer='random_normal'))
model.add(Dense(units=1, kernel_initializer='random_normal'))

model.compile(loss='mean_squared_error', optimizer='sgd')

model.summary()

model.fit(x_input, y_input, epochs=100, batch_size=32)

result = model.predict(np.array([
    encode_single_sample('./data/testing/number/img.png'),
]))

print('number result', rint(result)[0][0])
