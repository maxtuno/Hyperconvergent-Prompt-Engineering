import sys
import numpy as np

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

"""
My Reformulation of SAT
CNF n vars m clauses matrix form (-1, 0, 1) -> (n + 1,m) with n + 1 column number of nonzero elements in row, i.e, number of literals in each clause, CNF is SATIAFIABLE if and only if exist a (-1, 1...) that is interior to the polyhedral H-form of the CNF.
"""

"""
The H form of CNF  (-1, 0, 1) elements, last column number nonzero elements per ROW

If S (-1, 1...) not satisfied the CNF, exist S * ROW where S=-ROW (-S in CNF) then sum(S * ROW) == minus the number of nonzero elements in the ROW.

The result follows from ensure SATISFIABILITY
"""

def cnf_to_matrix(cnf, n, m):
    matrix = np.zeros(shape=(n + 1, m))
    for i, cls in enumerate(cnf):
        for lit in cls:
            matrix[abs(lit) - 1][i] = -1 if lit < 0 else 1
        matrix[-1][i] = len(cls)
    return np.asarray(matrix).T

def punto_dentro_del_politopo_cnf(point, h_representation):
    return len(h_representation) - sum(sum(c * x for c, x in zip(row[:-1], point)) > -row[-1] for row in h_representation)

def hess_polyedra(num_variables, h_representation):
    sat = [-1] * num_variables
    opt = sat[:]
    glb = np.inf
    while True:
        done = True
        for i in range(num_variables):
            sat[i] = -sat[i]
            loc = punto_dentro_del_politopo_cnf(sat, h_representation)
            if loc < glb:
                glb = loc
                print(glb)
                done = False
                opt = sat[:]
                if glb == 0:
                    return opt
            elif loc > glb:
                sat[i] = -sat[i]
        if done:
            break
    return opt

def generate_random_cnf_file(num_variables, num_clauses):
    cnf = []
    for _ in range(num_clauses):
        cls = []
        while len(cls) < 3:
            var = np.random.randint(1, num_variables + 1)
            if np.random.choice([True, False]):
                var = -var
            if not -var in cls and not var in cls:
                cls.append(var)
        cnf.append(cls)
    return cnf

if __name__ == '__main__':

    n, m, cnf = 0, 0, []
    with open(sys.argv[1], 'r') as cnf_file:
        lines = cnf_file.readlines()
        for line in filter(lambda x: not x.startswith('c'), lines):
            if line.startswith('p cnf'):
                n, m = list(map(int, line[6:].split(' ')))
            else:
                cnf.append(list(map(int, line.rstrip('\n')[:-2].split(' '))))

    
    h_representation = cnf_to_matrix(cnf, n, m)
    print(h_representation)

    sub_optimal = hess_polyedra(n, h_representation)
    
    print(sub_optimal)
