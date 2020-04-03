from optparse import OptionParser
import numpy as np
import sys
import HMM
import tensorflow.contrib.learn.python.learn.datasets.mnist as mnist

def one_hot(array):
    onehot_array = np.zeros((array.shape[0], 10), dtype=np.float64);
    onehot_array[np.arange(array.shape[0]), array] = 1;
    return onehot_array;

options = OptionParser();

#Add option to collect variable here
options.add_option("--alg", action="store", type="string", dest="alg");
options.add_option("--num_hmms", action="store", type="int", dest="num_hmms");
options.add_option("--num_states", action="store", type="int", dest="num_states");
options.add_option("--frac_vertical", action="store", type="float", dest="frac_vertical");
options.add_option("--batch_size", action="store", type="int", dest="batch_size");
options.add_option("--num_epochs", action="store", type="int", dest="num_epochs");
options.add_option("--init_file", action="store", type="string", dest="init_file");
options.add_option("--saver_prefix", action="store", type="string", dest="saver_prefix", default="Session");

(opts, args) = options.parse_args();

#Extract variable here
alg           = opts.alg;
num_hmms      = opts.num_hmms;
num_states    = opts.num_states;
frac_vertical = opts.frac_vertical;
batch_size    = opts.batch_size;
num_epochs    = opts.num_epochs;
init_file     = opts.init_file;
saver_prefix  = opts.saver_prefix;

#If an option is compulsory, add to must
must = [];
must.append(alg);
must.append(num_hmms);
must.append(num_states);
must.append(frac_vertical);
must.append(batch_size);
must.append(num_epochs);

#Check whether options passed correctly
correct = True;

for m in must:
    if m is None:
        correct = False;
        break;

#Modify message here
if correct is False:
    sys.exit("Usage : python " + sys.argv[0] + " --alg <HMM algorithm> --num_hmms <Number of HMMs> --num_states <Number of states in each HMM> --frac_vertical <Number of HMMs to look at column information> --batch_size <mini-batch size> --num_epochs <Number of epochs to do training> [--init_file <optional file to initialize from> --saver_prefix [prefix of file in which session is saved]]");

#Prepare datasets
datasets         = mnist.load_mnist();
training_vectors = datasets[0].images;
training_labels  = datasets[0].labels;
training_vectors = training_vectors.reshape((training_vectors.shape[0], 28, 28));
training_vectors = np.array([v.T for v in training_vectors], dtype=np.float64);
training_labels  = one_hot(training_labels);
test_vectors     = datasets[2].images;
test_labels      = datasets[2].labels;
test_vectors     = test_vectors.reshape((test_vectors.shape[0], 28, 28));
test_vectors     = np.array([v.T for v in test_vectors], dtype=np.float64);
test_labels      = one_hot(test_labels);

Net = HMM.HMM_NN(num_dims=28,num_states=num_states,num_hmms=num_hmms);
#Train
#Net.run_batch(training_vectors, training_labels, hmm_alg=alg, frac_vertical=frac_vertical, batch_size=batch_size, num_epochs=num_epochs, init_file=init_file, saver_prefix=saver_prefix, test_vectors=training_vectors, test_labels=training_labels);
#Test
Net.run_batch(training_vectors, training_labels, hmm_alg=alg, frac_vertical=frac_vertical, batch_size=batch_size, num_epochs=0, init_file=init_file, saver_prefix=saver_prefix, test_vectors=test_vectors, test_labels=test_labels);
