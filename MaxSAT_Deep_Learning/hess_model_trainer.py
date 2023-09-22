"""
MIT License

Copyright (c) 2023 Oscar Riveros

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import sys
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


def oracle(sat, cnf):
    loc = len(cnf)
    for cls in cnf:
        for lit in cls:
            if lit > 0 and sat[abs(lit) - 1] == 1 or lit < 0 and sat[abs(lit) - 1] == 0:
                loc -= 1
                break
    return loc


def hess(num_variables, cnf):
    sat = [0] * num_variables
    opt = sat[:]
    cur = np.inf
    while True:
        done = True
        glb = np.inf
        for i in range(num_variables):
            sat[i] = 1 - sat[i]
            loc = oracle(sat, cnf)
            if loc < glb:
                glb = loc
                if glb < cur:
                    done = False
                    cur = glb
                    # print(glb)
                    opt = sat[:]
                    if cur == 0:
                        return opt
            elif loc > glb:
                sat[i] = 1 - sat[i]
        if done:
            break
    return opt


def generate_random_cnf_file(num_variables, num_clauses):
    cnf = []
    for _ in range(num_clauses):
        cls = []
        # Randomly choose the number of literals in a cls (1-3).
        while len(cls) < np.random.randint(2, num_variables):
            var = np.random.randint(1, num_variables)  # Random var index
            if np.random.choice([True, False]):  # Randomly negate the var
                var = -var
            if not -var in cls and not var in cls:
                cls.append(var)
        cnf.append(cls)
    return cnf


def algorithm_learning(n, m, input_data, output_data, test_input_data, test_output_data):
    # TODO: Put a decent model here...
    preprocessing_layers = [
        tf.keras.layers.InputLayer(input_shape=(n, m, 1)),
    ]

    def conv_2d_pooling_layers(filters, number_colour_layers):
        return [
            tf.keras.layers.Conv2D(
                filters,
                number_colour_layers,
                padding='same',
                activation='linear'
            ),
            tf.keras.layers.MaxPooling2D()
        ]

    core_layers = \
        conv_2d_pooling_layers(8, 1) + \
        conv_2d_pooling_layers(16, 1) + \
        conv_2d_pooling_layers(32, 1) + \
        conv_2d_pooling_layers(64, 1)

    dense_layers = [
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='linear'),
        tf.keras.layers.Dense(n)
    ]

    # Build the CNN model
    model = tf.keras.Sequential(
        preprocessing_layers +
        core_layers +
        dense_layers
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint('./checkpoints/hess_model_{}_{}'.format(n, m),
                                                    monitor="val_binary_accuracy",
                                                    mode="max",
                                                    save_weights_only=False,
                                                    save_best_only=True,
                                                    verbose=1)
    callbacks = [checkpoint]

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['binary_accuracy'])

    # Train the model
    epochs = 40

    history = model.fit(input_data,
                        output_data,
                        validation_data=(test_input_data, test_output_data),
                        epochs=epochs,
                        batch_size=128,
                        callbacks=callbacks)

    # Restore the best model
    model = tf.keras.models.load_model(
        './checkpoints/hess_model_{}_{}'.format(n, m))

    print("Evaluate on test data")
    results = model.evaluate(test_input_data, test_output_data)
    print("test loss, test acc:", results)

    plt.clf()
    plt.plot(history.history['binary_accuracy'])
    plt.plot(history.history['val_binary_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('acc_history.png')

    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('loss_hystory.png')


def gen_dataset(n, m, num_samples):
    input_data, output_data = [], []

    while len(output_data) < num_samples:
        num_variables = np.random.randint(10, n)
        num_clauses = np.random.randint(10, m)

        cnf = generate_random_cnf_file(num_variables, num_clauses)
        opt = hess(num_variables, cnf)
        cnf_matrix = np.zeros(shape=(n, m))
        for i, cls in enumerate(cnf):
            for lit in cls:
                cnf_matrix[abs(lit) - 1][i] = -1 if lit < 0 else 1

        opt += (n - len(opt)) * [0]

        input_data.append(cnf_matrix)
        output_data.append(opt)

    return np.asarray(input_data), np.asarray(output_data)


if __name__ == '__main__':

    context_size = int(sys.argv[1])  # size of context
    num_samples = int(sys.argv[2])  # number of test samples

    n = context_size  # good results why? number of _variables
    m = context_size  # good results why? number of clauses

    print("Generating data")
    input_data, output_data = gen_dataset(n, m, num_samples // 2)
    print("Generating test data")
    test_input_data, test_output_data = gen_dataset(n, m, num_samples // 2)

    algorithm_learning(n, m, input_data, output_data,
                       test_input_data, test_output_data)
