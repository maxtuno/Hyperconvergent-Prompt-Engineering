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
                    print('o {}'.format(glb))
                    opt = sat[:]
                    if cur == 0:
                        return opt
            elif loc > glb:
                sat[i] = 1 - sat[i]
        if done:
            break
    return opt

def cnf_to_matrix(cnf, context_size):
    matrix = np.zeros(shape=(context_size, context_size))
    for i, cls in enumerate(cnf):
        for lit in cls:
            matrix[abs(lit) - 1][i] = -1 if lit < 0 else 1
    return np.asarray(matrix).flatten()

if __name__ == '__main__':
    context_size = int(sys.argv[2])

    # Restore the weights
    model = tf.keras.models.load_model('./checkpoints/hess_model_{}'.format(context_size))

    n, m, cnf = 0, 0, []
    with open(sys.argv[1], 'r') as cnf_file:
        lines = cnf_file.readlines()
        for line in filter(lambda x: not x.startswith('c'), lines):
            if line.startswith('p cnf'):
                n, m = list(map(int, line[6:].split(' ')))
            else:
                cnf.append(list(map(int, line.rstrip('\n')[:-2].split(' '))))

    cnf_matrix = cnf_to_matrix(cnf, context_size)
    
    prediction = model.predict(np.asarray([cnf_matrix]))[0]
    sat = prediction > 0.5
    model_suboptimal = oracle(sat, cnf)
    print('c HESS for MaxSAT model')
    print('s OPTIMAL (?) {}'.format(model_suboptimal))
    assignment = ''
    for i in range(n):
        if sat[i]:
            assignment += str(+(i + 1)) + ' '
        else:
            assignment += str(-(i + 1)) + ' '
    print('v ' + assignment + '0')

    model_assignment = assignment[:]

    print('c HESS for MaxSAT Algorithm')
    sat = hess(n, cnf)
    hess_suboptimal = oracle(sat, cnf)
    print('s OPTIMAL (?) {}'.format(hess_suboptimal))
    assignment = ''
    for i in range(n):
        if sat[i]:
            assignment += str(+(i + 1)) + ' '
        else:
            assignment += str(-(i + 1)) + ' '
    print('v ' + assignment + '0')

    hess_assignment = assignment[:]

    print('precision: ', (sum([x == y for x, y in zip(
        model_assignment.split(' '), hess_assignment.split(' '))]) - 1) / n)

    print('expected model suboptimal ', model_suboptimal)
    print('expected hess suboptimal ', hess_suboptimal)
