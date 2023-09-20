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
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from PIL import Image

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
        while len(cls) < 3:  # Randomly choose the number of literals in a cls (1-3).
            var = np.random.randint(1, num_variables)  # Random var index
            if np.random.choice([True, False]):  # Randomly negate the var
                    var = -var
            if not -var in cls and not var in cls:
                cls.append(var)
        cnf.append(cls)
    return cnf

def algorithm_learning(limit_size, input_data, output_data, test_input_data, test_output_data):
    # TODO: Put a decent model here...

    # Build the CNN model
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(limit_size, limit_size, 1)))
    model.add(tf.keras.layers.Activation('linear'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Flatten())
    
    model.add(tf.keras.layers.Dense(limit_size))
    
    model.build(input_shape=(None, limit_size, limit_size, 1))
    model.summary()

    # Compile the model
    model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['binary_accuracy'])


    # Train the model
    epochs = 10
 
    history = model.fit(input_data, output_data, validation_data=(test_input_data, test_output_data), epochs=epochs, batch_size=128, shuffle=True)

    print("Evaluate on test data")
    results = model.evaluate(test_input_data, test_output_data)
    print("test loss, test acc:", results)

    # Save the weights
    model.save('./checkpoints/hess_model_{}'.format(limit_size))

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

def gen_dataset(limit_size, num_samples):
    input_data, output_data = [], []
    
    while len(output_data) < num_samples:    
        num_variables = np.random.randint(10, limit_size) 
        num_clauses = np.random.randint(10, limit_size) 

        cnf = generate_random_cnf_file(num_variables, num_clauses)
        opt = hess(num_variables, cnf)
        cnf_matrix = np.zeros(shape=(limit_size, limit_size))
        for i, cls in enumerate(cnf):
            for lit in cls:
                cnf_matrix[abs(lit) - 1][i] = 0.25 if lit < 0 else 0.75

        opt += (limit_size - len(opt)) * [0]

        input_data.append(cnf_matrix)
        output_data.append(opt)

    return np.asarray(input_data), np.asarray(output_data)


if __name__ == '__main__':

    num_samples = 1000
    limit_size = 150

    for _ in range(1):
        print("Generating data")
        input_data, output_data = gen_dataset(limit_size, num_samples // 2)
        print("Generating test data")
        test_input_data, test_output_data = gen_dataset(limit_size, num_samples // 2)

        algorithm_learning(limit_size, input_data, output_data, test_input_data, test_output_data)