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

def generate_random_cnf_file(num_variables, num_clauses):
    with open('test.cnf', 'w') as file:
        print('p cnf {} {}'.format(num_variables, num_clauses), file=file)
        for _ in range(num_clauses):
            cls = []
            while len(cls) < np.random.randint(2, num_variables):
                var = np.random.randint(1, num_variables)  # Random var index
                if np.random.choice([True, False]):  # Randomly negate the var
                    var = -var
                if -var not in cls and var not in cls:
                    cls.append(var)
            print(' '.join(map(str, cls)) + ' 0', file=file)

if __name__ == '__main__':
    num_variables = int(sys.argv[1])
    num_clauses = int(sys.argv[2])
    generate_random_cnf_file(num_variables, num_clauses)