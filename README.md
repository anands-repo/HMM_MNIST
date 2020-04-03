#HMM MNIST

MNIST digit recognition using an HMM ensemble and a Neural Network. The forward likelihood for a digit image is computed from an ensemble of HMMs. The forward log-likelihoods are then analyzed by an FC network to determine the label of the digit. The complete system is trained end-to-end using backpropagation.

1. Training the HMM_NN network:

	python run_script.py 

will provide options on how to run the script.

An example run command is:
	python run_script.py --alg forward --num_hmms 30 --num_states 14 --frac_vertical 0.50 --batch_size 100 --num_epochs 10

This run saves the session in a file in the current path. The file may be later passed to tensorflow using the --init_file option. 

To obtain just training accuracy for the whole dataset, the script has to be run with zero epochs, however a saved session needs to be passed to the script. e.g.:

	python run_script.py --alg forward --num_hmms 30 --num_states 14 --frac_vertical 0.50 --batch_size 100 --num_epochs 0 --init_file ./<TF session file>

The init-file seems to be tensorflow version dependent. The version from which the provided init-file is produced is 0.12.0-rc0.

2. Evaluating the HMM_NN

For this, please run python run_script_eval.py. This script expects a saved tensorflow session to be provided as input. The command to be used is for example:
	python run_script_eval.py --alg forward --num_hmms 30 --num_states 14 --frac_vertical 0.50 --batch_size 100 --num_epochs 0 --init_file ./Session_fourth_set14_30_forward_7-30

3. Plotting the HMM state mean values
	python plotHMMs.py --alg forward --num_hmms 30 --num_states 14 --frac_vertical 0.50 --batch_size 100 --num_epochs 0 --init_file ./Session_fourth_set14_30_forward_7-30


