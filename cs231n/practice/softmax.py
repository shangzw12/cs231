import numpy as np
import  matplotlib.pyplot as plt
#hyparameter
N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in xrange(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j
# lets visualize the data:

#plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
#plt.show()

# some hyperparameters
step_size = 1e-0
reg = 1e-3 # regularization strength
#init Weight
W = 0.01 * np.random.randn(D, K)
b = np.zeros((1, K))
for i in range(200):
    # scores
    scores = X.dot( W ) + b
    exp_scores = np.exp( scores )
    #probability
    probs  = exp_scores / np.sum(exp_scores, axis = 1, keepdims = True)
    #correct_log probs
    num_examples = X.shape[0]
    correct_logprobs = -np.log(probs[range(num_examples) , y])
    #compute loss
    data_loss = np.sum(correct_logprobs) / num_examples
    #regularization
    reg_loss  =  0.5 * reg * np.sum(W*W) #sum all elements
    #whole loss
    loss = data_loss + reg_loss
    if i % 10 == 0:
        print "iteration %d: loss %f" % (i, loss)

    dscores = probs
    dscores[range(num_examples), y ] -= 1
    dscores /= num_examples

    dW = X.T.dot(dscores)
    db = np.sum (dscores, axis = 0, keepdims = True)
    dW += reg * W
    #update the parameter
    W += -step_size * dW
    b += -step_size * db