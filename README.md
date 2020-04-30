# HMM MNIST

Note: This code was written many years ago for an old version of tensorflow (before tensorflow 1.0). Hence, recommended to use this as reference code rather than to run directly.

MNIST digit recognition using an HMM ensemble and a Neural Network. The forward likelihood for a digit image is computed from an ensemble of HMMs. The forward log-likelihoods are then analyzed by an FC network to determine the label of the digit. The complete system is trained end-to-end using backpropagation.

The basic inspiration for this work is the idea that the distance between a "representative" of a digit and any instance of any digit can indicate whether the instance is in the same category as the representative. This can be extended to thinking in terms of edit distance. From there, we can extend this to HMMs (pair-HMMs are used to make probabilistic edit-distance computations). Each HMM models columns in a digit using its states which emit multivariate Gaussian vectors. If an HMM is fit to a digit "8", say, then the expectation is that the states in the HMM will learn column vectors in their Gaussian means that represent the column-elements generally found in "8"s. This is generalized to use an ensemble of HMMs to determine forward likelihoods, and then process these likelihoods with a feed-forward neural network.

1. Training the HMM_NN network:

	```python run_script.py```

will provide options on how to run the script.

An example run command is:

	python run_script.py --alg forward --num_hmms 30 --num_states 14 --frac_vertical 0.50 --batch_size 100 --num_epochs 10

This run saves the session in a file in the current path. The file may be later passed to tensorflow using the --init_file option. 

To obtain just training accuracy for the whole dataset, the script has to be run with zero epochs, however a saved session needs to be passed to the script. e.g.:

	python run_script.py --alg forward --num_hmms 30 --num_states 14 --frac_vertical 0.50 --batch_size 100 --num_epochs 0 --init_file ./<TF session file>

The init-file seems to be tensorflow version dependent. The version from which the provided init-file is produced is 0.12.0-rc0.

2. Evaluating the HMM_NN

For this, please run python run_script_eval.py. This script expects a saved tensorflow session to be provided as input. The command to be used is for example:

	python run_script_eval.py --alg forward --num_hmms 30 --num_states 14 --frac_vertical 0.50 --batch_size 100 --num_epochs 0 --init_file ./<TF session file>

3. Plotting the HMM state mean values

	```python plotHMMs.py --alg forward --num_hmms 30 --num_states 14 --frac_vertical 0.50 --batch_size 100 --num_epochs 0 --init_file ./<TF session file>```


