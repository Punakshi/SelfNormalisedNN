# Reproducing Figure 1 from the paper

The data for the plot was created by running the following commands preeceded by python.
Each argument has its own meaning and significance which can be refered from the codebase.

    ./run.py -g 0 -d 08 -a selu -l 1e-5 -e 2000 --dataset mnist
    ./run.py -g 1 -d 16 -a selu -l 1e-5 -e 2000 --dataset mnist
    ./run.py -g 2 -d 32 -a selu -l 1e-5 -e 2000 --dataset mnist
    ./run.py -g 3 -d 08 -a relu --batchnorm -l 1e-5 -e 2000 --dataset mnist
    ./run.py -g 0 -d 16 -a relu --batchnorm -l 1e-5 -e 2000 --dataset mnist
    ./run.py -g 1 -d 32 -a relu --batchnorm -l 1e-5 -e 2000 --dataset mnist

    ./run.py -g 0 -d 08 -a selu -l 1e-5 -e 2000 --dataset cifar10
    ./run.py -g 1 -d 16 -a selu -l 1e-5 -e 2000 --dataset cifar10
    ./run.py -g 2 -d 32 -a selu -l 1e-5 -e 2000 --dataset cifar10
    ./run.py -g 3 -d 08 -a relu --batchnorm -l 1e-5 -e 2000 --dataset cifar10
    ./run.py -g 0 -d 16 -a relu --batchnorm -l 1e-5 -e 2000 --dataset cifar10
    ./run.py -g 1 -d 32 -a relu --batchnorm -l 1e-5 -e 2000 --dataset cifar10

The plots where then created using `create_plots.ipynb`.
