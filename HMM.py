import tensorflow as tf
import math
import numpy as np
#from tensorflow.examples.tutorials.mnist import input_data as mnist
import tensorflow.contrib.learn.python.learn.datasets.mnist as mnist

class state_np:
    def __init__(self, state, session):
        self.mu  = session.run(state.mu);
        self.cov = session.run(state.cov);

class hmm_np:
    def __init__(self, hmm, session):
        self.T0 = session.run(hmm.T0);
        self.T  = session.run(hmm.T);

        self.states = [state_np(state, session) for state in hmm.states];
        print "Completed initialization";

    def most_probable_mus(self, traverse_once=False):
        state_id = np.argmax(self.T0);
        counter  = 1;
        retmus   = [];
        count_lim = 14 if traverse_once is True else 28;

        while(counter < count_lim):
            if traverse_once is False:
            	retmus.append(self.states[state_id].mu);
            	state_id = np.argmax(self.T[state_id]);
            	counter += 1;
            	print "counter = " + str(counter);

            if traverse_once is True:
            	retmus.append(self.states[state_id].mu);
            	state_id2 = np.argmax(self.T[state_id]);
            	self.T[state_id][state_id2] = -1;
            	state_id = state_id2;
            	counter += 1;
            	print "counter = " + str(counter);

        return np.array(retmus);

class State:
    def __init__(self, dim=28):
        self.dim = dim; 

        #The mean vector
        mu       = np.random.uniform(0, 1, dim);
        self.mu  = tf.Variable(mu, dtype=tf.float64);

        #The covariance vector - initialize to diagonal matrix, with non-zero diagonals
        sigma    = np.diag(np.random.uniform(0.1,0.9,dim));
        self.sig = tf.Variable(sigma, dtype=tf.float64);
        self.cov = tf.matmul(self.sig, self.sig);

        #Define the distribution
        self.dist = tf.contrib.distributions.MultivariateNormalFull(self.mu, self.cov);

    #Return density function evaluated for a sequence of symbols
    def prob(self, x):
        return self.dist.pdf(x);

class HMM:
    def __init__(self, num_states=4, num_dims=28):
        self.num_states = num_states;

        #Meta-parameters. Parameters from which transition probabilities are obtained
        self.Z  = tf.Variable(tf.truncated_normal([num_states, num_states], stddev=1, dtype=tf.float64));
        self.Z0 = tf.Variable(tf.truncated_normal([1, num_states], stddev=1, dtype = tf.float64));

        #Create state sequence
        self.states = [State(num_dims) for i in range(num_states)];

        #Small initial bias
        self.small_bias    = 10 ** (-300);

    #Return the density function evaluated for each state across every member of a sequence
    def prob(self, x):
        #Initialize probabilities
        prob = [];

        for s in self.states:
            prob.append(s.prob(x));

        return tf.transpose(tf.pack(prob));

    #Obtain probabilities for a batch
    def prob_batch(self, x, batch_size=100, seq_length=28):
        prob = [];

        for s in self.states:
            prob.append(s.prob(x));

        pre = tf.pack(prob);
        pre = tf.reshape(prob, shape=[self.num_states, batch_size * seq_length]);
        pre = tf.transpose(pre);
        val = tf.reshape(pre, shape=[batch_size, seq_length, self.num_states]);

        return val;

    #Obtain the viterbi score
    def run_viterbi(self, obs_list, length=28):
        #Convert meta-parameters to HMM parameters
        self.T            = tf.exp(self.Z) / tf.reduce_sum(tf.exp(self.Z), 1) + self.small_bias;
        self.T0           = tf.exp(self.Z0) / tf.reduce_sum(tf.exp(self.Z0))  + self.small_bias;

        #Observation probability for each position for each state
        obs_prob_list     = self.prob(obs_list) + self.small_bias;

        #Initialize the forward recursion
        viterbi           = tf.log(self.T0) + tf.log(obs_prob_list[0]);

        ##The forward recursion
        for t in range(length-1):
            viterbi = tf.reduce_max(tf.reshape(viterbi, shape=[self.num_states, 1]) + tf.log(self.T), reduction_indices=[0])  + tf.log(obs_prob_list[t+1]);

        #termination
        prob_o = tf.reduce_max(viterbi);

        return prob_o;

    #Return tensor representing probability of sequence
    def run_forward(self, obs_list, length=28):
        #Convert meta-parameters to HMM parameters
        self.T            = tf.exp(self.Z) / (tf.reshape(tf.reduce_sum(tf.exp(self.Z), 1), (self.num_states, 1))) + self.small_bias;
        self.T0           = tf.exp(self.Z0) / tf.reduce_sum(tf.exp(self.Z0)) + self.small_bias;

        #Observation probability for each position for each state
        obs_prob_list     = self.prob(obs_list) + self.small_bias;

        #Initialize the forward recursion
        forward_pre = self.T0 * obs_prob_list[0];
        redux       = tf.reduce_sum(forward_pre);
        beta        = tf.cond(redux > tf.constant(0, dtype=tf.float64), lambda : redux, lambda : tf.constant(1.0, dtype=tf.float64));
        forward     = forward_pre / beta;
        beta_all    = tf.log(beta + self.small_bias);

        ##The forward recursion
        for t in range(length-1):
            forward_pre = tf.matmul(forward, self.T) * obs_prob_list[t+1];
            redux       = tf.reduce_sum(forward_pre);
            beta        = tf.cond(redux > tf.constant(0, dtype=tf.float64), lambda : redux, lambda : tf.constant(1.0, dtype=tf.float64));
            forward     = forward_pre / beta;
            beta_all    = tf.log(beta + self.small_bias) + beta_all;

        #termination
        prob_o = beta_all;

        return prob_o;

    #Obtain the viterbi score in batches
    def run_viterbi_batch(self, obs_list, batch_size=500, length=28):
        #Convert meta-parameters to HMM parameters
        self.T         = tf.exp(self.Z) / tf.reduce_sum(tf.exp(self.Z), 1) + self.small_bias;
        self.T0        = tf.exp(self.Z0) / tf.reduce_sum(tf.exp(self.Z0))  + self.small_bias;

        #Observation probability for each position for each state
        obs_prob_list  = self.prob_batch(obs_list, batch_size, length) + self.small_bias;

        #Initialize the forward recursion
        viterbi        = tf.reshape(
                             tf.reshape(tf.log(self.T0), shape=[1,self.num_states]) + 
                             tf.reshape(tf.log(obs_prob_list[:,0,:]), shape=[batch_size,self.num_states]), 
                         shape=[batch_size,1,self.num_states]);

        #Obtain replicates of transition probability matrix
        Tr             = tf.log(tf.pack([tf.transpose(self.T) for i in range(batch_size)]));

        #The viterbi recursion
        for t in range(length-1):
            viterbi = tf.reduce_max(viterbi + Tr, reduction_indices=[2]);# + tf.log(tf.reshape(obs_prob_list[:,t+1,:], shape=[batch_size,1,self.num_states]));
            viterbi = tf.reshape(viterbi, shape=[batch_size,1,self.num_states]);
            obs     = tf.reshape(obs_prob_list[:,t+1,:], shape=[batch_size,1,self.num_states]);
            viterbi = viterbi + tf.log(obs);

        #Termination
        viterbi = tf.reshape(viterbi, shape=[batch_size, self.num_states]);
        prob_o  = tf.reduce_max(viterbi, reduction_indices=[1]);

        return tf.reshape(prob_o, shape=[1,batch_size]);

    #Return tensor representing probability of a batch of sequences
    def run_forward_batch(self, obs_list, batch_size=100, length=28):
        #Convert meta-parameters to HMM parameters
        self.T         = tf.exp(self.Z) / tf.reduce_sum(tf.exp(self.Z), 1) + self.small_bias;
        self.T0        = tf.exp(self.Z0) / tf.reduce_sum(tf.exp(self.Z0))  + self.small_bias;

        #Observation probability for each position for each state
        obs_prob_list  = self.prob_batch(obs_list, batch_size=batch_size, seq_length=length) + self.small_bias;

        #Initialize the forward recursion
        forward_pre = tf.reshape(self.T0, shape=[1,self.num_states]) * tf.reshape(obs_prob_list[:,0,:], shape=[batch_size,self.num_states]);
        redux       = tf.reduce_sum(forward_pre, reduction_indices=[1]);
        beta        = tf.reshape(redux + self.small_bias, shape=[batch_size,1]);
        forward     = forward_pre / beta;
        beta_all    = tf.log(beta);

        ##The forward recursion
        for t in range(length-1):
            forward_pre = tf.matmul(forward, self.T) * tf.reshape(obs_prob_list[:,t+1,:], shape=[batch_size, self.num_states]);
            redux       = tf.reduce_sum(forward_pre, reduction_indices=[1]);
            beta        = tf.reshape(redux + self.small_bias, shape=[batch_size,1]);
            forward     = forward_pre / beta;
            beta_all    = tf.log(beta) + beta_all;

        #termination
        prob_o = beta_all;

        return prob_o;

class HMM_NN():
    def __init__(self, num_dims=28, num_states=4, num_hmms=5, learning_rate=1e-4):
        self.num_dims   = num_dims;
        self.num_states = num_states;
        self.num_hmms   = num_hmms;

        #Initialize all HMMs
        self.HMMs = []
        for i in range(num_hmms):
            self.HMMs.append(HMM(num_dims=num_dims, num_states=num_states));

        #Initialize the softmax layer
        self.W = self.weight_variable([num_hmms, 10]);
        self.b = self.bias_variable([10]);

        #Learning rate
        self.learning_rate = learning_rate;

    #Initialization of the softmax layer
    def weight_variable(self, shape):
            initial = tf.truncated_normal(shape, stddev=1, dtype = tf.float64)
            return tf.Variable(initial)

    #Initialization of the softmax layer
    def bias_variable(self, shape):
            initial = tf.constant(0.1, shape = shape, dtype = tf.float64)
            return tf.Variable(initial)

    #Train the HMM_NN layer
    def run(self, training_vectors, training_labels, hmm_alg="forward", num_epochs=5, init_file = None):
        #Input placeholders
        x   = tf.placeholder(tf.float64, shape = training_vectors.shape[1:]);
        y_  = tf.placeholder(tf.float64, shape = [1,training_labels.shape[1]]);

        hmm_prob  = None;

        #Collect the HMM computation tensors
        if hmm_alg == "forward":
            hmm_prob  = tf.reshape(tf.pack([hmm.run_forward(x,training_vectors.shape[1]) for hmm in self.HMMs]), shape=[1, self.num_hmms]);
        else:
            hmm_prob  = tf.reshape(tf.pack([hmm.run_viterbi(x,training_vectors.shape[1]) for hmm in self.HMMs]), shape=[1, self.num_hmms]);

        norm_prob = hmm_prob;

        #Setup the softmax layer output tensor
        y        = tf.matmul(norm_prob, self.W) + self.b;

        #Tensors for prediction and training
        cross_entropy      = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
        train_step         = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy           = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        acc_sum = 0;

        if init_file is not None:
            loader = tf.train.Saver();
            loader.restore(sess, init_file);

        #Stochastic gradient descend
        for epoch_num in range(num_epochs):
            for i, (vector, label) in enumerate(zip(training_vectors, training_labels)):
                #Feed dict
                feed_dict     = {x:vector, y_:label.reshape((1,label.shape[0]))};

                ##Predict accuracy
                acc_sum += sess.run(accuracy, feed_dict=feed_dict);

                ##Sample one or two probabilities
                prob_val = sess.run(norm_prob, feed_dict=feed_dict);

                ##Train
                sess.run(train_step, feed_dict=feed_dict);

                if (i % 100 == 0):
                    print "Accuracy for batch " + str(i/100) + " = " + str(acc_sum * 1.0 / 100);
                    acc_sum = 0;

    def load_session(self, training_vectors, training_labels, hmm_alg="forward", frac_vertical=0.5, batch_size=50, init_file=None):
        #Input placeholders
        x1  = tf.placeholder(tf.float64, shape = [batch_size] + list(training_vectors.shape[1:]));
        x2  = tf.placeholder(tf.float64, shape = [batch_size] + list(training_vectors.shape[1:]));
        y_  = tf.placeholder(tf.float64, shape = [batch_size,training_labels.shape[1]]);

        only_eval = True;
        num_epochs = 0;

        hmm_prob  = None;

        #Collect the forward computation tensors
        if hmm_alg == "forward":
            hmm_prob  = tf.pack([hmm.run_forward_batch(x1,batch_size=batch_size,length=training_vectors.shape[1]) if (i < frac_vertical * self.num_hmms) else
                                 hmm.run_forward_batch(x2,batch_size=batch_size,length=training_vectors.shape[1])
                                 for i, hmm in enumerate(self.HMMs)]);
        else:
            hmm_prob  = tf.pack([hmm.run_viterbi_batch(x1,batch_size=batch_size,length=training_vectors.shape[1]) if (i < frac_vertical * self.num_hmms) else
                                 hmm.run_viterbi_batch(x2,batch_size=batch_size,length=training_vectors.shape[1])
                                 for i, hmm in enumerate(self.HMMs)]);

        norm_prob = tf.transpose(tf.reshape(hmm_prob, shape=[self.num_hmms, batch_size]));

        #Obtain transposed copy of training_vectors
        if (not only_eval) and (num_epochs > 0):
            training_vectors_transpose = np.array([v.T for v in training_vectors], dtype=np.float64);

        #Setup the softmax layer output tensor
        y         = tf.matmul(norm_prob, self.W) + self.b;

        #Tensors for prediction and training
        cross_entropy      = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
        train_step         = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy           = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
        y_softmax          = tf.nn.softmax(y);

        sess = tf.Session()
        sess.run(tf.initialize_all_variables())

        if init_file is not None:
            print "Loading session from " + init_file;
            loader = tf.train.Saver();
            loader.restore(sess, init_file);

        return sess;

    #Train the HMM_NN layer on a batch
    def run_batch(self, training_vectors, training_labels, hmm_alg="forward", frac_vertical=0.5, batch_size=50, num_epochs=4, init_file = None, saver_prefix = "Session", test_vectors = None, test_labels = None, only_eval=False):
        #Input placeholders
        x1  = tf.placeholder(tf.float64, shape = [batch_size] + list(training_vectors.shape[1:]));
        x2  = tf.placeholder(tf.float64, shape = [batch_size] + list(training_vectors.shape[1:]));
        y_  = tf.placeholder(tf.float64, shape = [batch_size,training_labels.shape[1]]);

        hmm_prob  = None;

        #Collect the forward computation tensors
        if hmm_alg == "forward":
            hmm_prob  = tf.pack([hmm.run_forward_batch(x1,batch_size=batch_size,length=training_vectors.shape[1]) if (i < frac_vertical * self.num_hmms) else
                                 hmm.run_forward_batch(x2,batch_size=batch_size,length=training_vectors.shape[1])
                                 for i, hmm in enumerate(self.HMMs)]);
        else:
            hmm_prob  = tf.pack([hmm.run_viterbi_batch(x1,batch_size=batch_size,length=training_vectors.shape[1]) if (i < frac_vertical * self.num_hmms) else
                                 hmm.run_viterbi_batch(x2,batch_size=batch_size,length=training_vectors.shape[1])
                                 for i, hmm in enumerate(self.HMMs)]);

        norm_prob = tf.transpose(tf.reshape(hmm_prob, shape=[self.num_hmms, batch_size]));

        #Obtain transposed copy of training_vectors
        if (not only_eval) and (num_epochs > 0):
            training_vectors_transpose = np.array([v.T for v in training_vectors], dtype=np.float64);

        #Setup the softmax layer output tensor
        y         = tf.matmul(norm_prob, self.W) + self.b;

        #Tensors for prediction and training
        cross_entropy      = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
        train_step         = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy           = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
        y_softmax          = tf.nn.softmax(y);

        sess = tf.Session()
        sess.run(tf.initialize_all_variables())

        if init_file is not None:
            print "Loading session from " + init_file;
            loader = tf.train.Saver();
            loader.restore(sess, init_file);

        if (not only_eval) and (num_epochs > 0):
            num_batches = training_vectors.shape[0] / batch_size;

            if (num_batches * batch_size != training_vectors.shape[0]):
                sys.exit("Unsupported batch_size!");

            #Mini-batch gradient descend
            for it in range(num_epochs):
                acc_sum = 0;

                for i in range(num_batches):
                    batch_start = i * batch_size;
                    batch_end   = (i + 1) * batch_size;

                    #Feed dict
                    feed_dict   = {x1:training_vectors[batch_start:batch_end],x2:training_vectors_transpose[batch_start:batch_end],y_:training_labels[batch_start:batch_end]};

                    ##Predict accuracy
                    acc         = sess.run(accuracy, feed_dict=feed_dict);
                    acc_sum    += acc;

                    ##Train
                    sess.run(train_step, feed_dict=feed_dict);

                    print "\tAccuracy for batch " + str(i) + " = " + str(acc);

                print "Epoch (num_epoch=%d, num_states=%d, num_hmms=%d, alg=%s, frac_vertical=%f) training accuracy = %f" %(it, self.num_states, self.num_hmms, hmm_alg, frac_vertical ,acc_sum / num_batches);
                print;
                acc_sum = 0;

            saver      = tf.train.Saver();
            sess_name  = saver_prefix + str(self.num_states) + "_" + str(self.num_hmms) + "_" + hmm_alg + "_" + str(int(frac_vertical * self.num_states));
            saved_name = saver.save(sess, sess_name, global_step=num_epochs);
            print "Session saved in " + saved_name;

        if test_vectors is not None:
            test_vectors_transpose = np.array([v.T for v in test_vectors], dtype=np.float64);

            num_batches = test_vectors.shape[0] / batch_size;

            acc_sum = 0;

            if num_batches * batch_size != test_vectors.shape[0]:
                sys.exit("Batch size not supported!");

            classification_confusion = np.array([[0.0 for i in range(10)] for j in range(10)]);

            for i in range(num_batches):
                batch_start = i * batch_size;
                batch_end   = (i + 1) * batch_size;

                #Feed dict
                feed_dict   = {x1:test_vectors[batch_start:batch_end],x2:test_vectors_transpose[batch_start:batch_end],y_:test_labels[batch_start:batch_end]};

                ##Predict accuracy
                labs, acc   = sess.run((y_softmax, accuracy), feed_dict=feed_dict);
                acc_sum    += acc;

                for lab, pred_lab in zip(test_labels[batch_start:batch_end], labs):
                    plab = np.argmax(pred_lab);
                    classification_confusion[np.argmax(lab)][plab] += 1;
                
            print "Test accuracy = " + str(acc_sum/num_batches);
            print classification_confusion.astype(np.int32);
