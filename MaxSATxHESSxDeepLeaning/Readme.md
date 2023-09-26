# Algorithm Learning

## MaxSAT x HESS x Deep Learning

This project focuses on the learning of complex algorithms, specifically HESS (first-order), which is a black-box polynomial optimization algorithm, for various NP-Hard problems. In this project, MaxSAT serves as the foundational problem.

usage:

    1- ejecute hess_model_trainer.ipynb # this generate an model on the checkpoint folder, take note on CONTEXT_SIZE.

    2- python3 gen_easy_cnf.py <numver_of_variables <= limit_size>> <number_of_clauses <= limit_size>> # this generate a "test.cnf" file

    2- python3 hess_model_evaluation.py <cnf_file|test.cnf> <context_size> # this evaluate the model vs HESS algorithm.

