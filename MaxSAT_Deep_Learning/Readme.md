# MaxSAT x Deep Learning

First of all, this is a basic example, and in many ways, a "toy" one. To make it work better (I believe), it needs a more robust multi-output learning model and, of course, more computational resources in general. This is the foundational idea, and if someone wants to take up the project again, they are welcome to do so.

usage:

    1- python3 hess_model_trainer.py <limit_size> <num_of_samples> # this generate an model on the checkpoint folder

    2- python3 gen_easy_cnf.py <numver_of_variables <= limit_size>> <number_of_clauses <= limit_size>> # this generate a "test.cnf" file

    2- python3 hess_model_evaluation.py <cnf_file|test.cnf> <limit_size> # this evaluate the model vs HESS algorithm.


for my machine a limit_size of 150 and num_of_samples of 1000 is ok.
