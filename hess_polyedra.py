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

If S (-1, 1...) not satisfied the CNF, exist S * ROW where S=-ROW (-S in CNF) then sum(S * ROW)  == minus the number of nonzero elements in the ROW.

The result follows from ensure SATISFIABILITY
"""

import sys
import numpy as np

db = []

def cnf_to_h_form(cnf, n, m):
    matrix = np.zeros(shape=(n + 1, m), dtype=np.int8)
    for i, cls in enumerate(cnf):
        for lit in cls:
            matrix[abs(lit) - 1][i] = -1 if lit < 0 else 1
        matrix[-1][i] = -len(cls)
    return np.asarray(matrix).T


def inside_polytope(point, h_cnf):
    return np.sum(np.matmul(h_cnf[:, :-1], point) > h_cnf[:, -1])


def hess(num_variables, h_cnf):
    sat = [-1] * num_variables
    opt = sat[:]
    glb = np.inf
    while True:
        done = True
        for i in range(num_variables):
            for j in range(num_variables):
                for k in range(num_variables):
                    sat[i], sat[j] = sat[j], sat[i]
                    if sat in db:
                        sat[k] = -sat[k]
                    else:
                        db.append(sat[:])
                        loc = h_cnf.shape[0] - inside_polytope(sat, h_cnf)
                        if loc < glb:
                            glb = loc
                            print(glb)
                            done = False
                            opt = sat[:]
                            if glb == 0:
                                return opt
                        elif loc > glb:
                            sat[k] = -sat[k]
        if done:
            break
    return opt


if __name__ == '__main__':

    n, m, cnf = 0, 0, []
    with open(sys.argv[1], 'r') as cnf_file:
        lines = cnf_file.readlines()
        for line in filter(lambda x: not x.startswith('c'), lines):
            if line.startswith('p cnf'):
                n, m = list(map(int, line[6:].split(' ')))
            else:
                try:
                    cnf.append(list(map(int, line.rstrip('\n')[:-2].split(' '))))
                except Exception as ex:
                    print(ex)

    h_cnf = cnf_to_h_form(cnf, n, m)
    print(h_cnf)

    sub_optimal = hess(n, h_cnf)

    print(sub_optimal)
