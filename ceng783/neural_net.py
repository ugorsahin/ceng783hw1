import numpy as np
import matplotlib.pyplot as plt
import ceng783.linear_classification as lc


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    """

    """
    # Unpack variables from the params dictionary

    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    wrap   = self.wrapit(X)
    layer1 = self.linear(wrap,W1,b1)
    layer1 = self.relu(layer1)
    layer2 = self.linear(layer1,W2,b2)
    if y is None: return layer2["data"]
    
    layer2        = self.softmax(layer2)
    loss,scores   = self.xentropy(y,layer2,reg)

    dW2,db2 = self.grad(layer2,W=W2,y=scores,reg=reg,X=layer1)
    dW1,db1 = self.grad(layer1,X=X,y=scores,W1=W1,reg=reg,W2=W2)
    
    grads = {"W1":dW1.squeeze(),"W2":dW2.squeeze(),"b1":db1.squeeze(),"b2":db2.squeeze()}

    return loss, grads

  def train(self, X, y, X_val, y_val,learning_rate=1e-3, learning_rate_decay=0.95,reg=1e-5, num_iters=100,batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train // batch_size, 1)
    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in range(num_iters):
      rand_vals = np.random.choice(num_train,batch_size)
      X_batch,y_batch = X[rand_vals],y[rand_vals]

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      pass
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      self.params["W1"] += -1 * grads["W1"] * learning_rate 
      self.params["W2"] += -1 * grads["W2"] * learning_rate 
      self.params["b1"] += -1 * grads["b1"] * learning_rate 
      self.params["b2"] += -1 * grads["b2"] * learning_rate 
      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      pass
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay
    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']

    wrap   = self.wrapit(X)
    layer1 = self.linear(wrap,W1,b1)
    layer1 = self.relu(layer1)
    layer2 = self.linear(layer1,W2,b2)
    layer2 = self.softmax(layer2)
    scores = np.argmax(layer2["data"],axis=1)
    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    pass
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return scores

  def xentropy(self, y, wrapper,reg):
    scores = wrapper["data"]
    logs = -np.log(scores[np.arange(y.shape[0]),y])
    loss = np.sum(logs) / y.shape[0]
    scores[np.arange(y.shape[0]),y] -=1
    scores /= y.shape[0]

    loss += reg*np.sum(self.params["W1"]**2)/2 + reg*np.sum(self.params["W2"]**2)/2
    return loss,scores

  def wrapit(self,data):
    return {"data":data}

  def linear(self,wrapper,W,b=None):
    scores = {}
    X = wrapper["data"]
    scores["data"] = X.dot(W) + b
    return scores

  def relu(self,wrapper):
    scores  = {"grad_back":"relu"}
    wrapper["data"][wrapper["data"]<0] = 0
    scores["data"] = wrapper["data"]
    return scores

  def softmax(self,wrapper):
    scores = {"grad_back":"softmax"}
    exps = np.exp(wrapper["data"] - np.expand_dims(np.argmax(wrapper["data"],axis=1),axis=1))
    sums  = np.sum(exps,axis=1,keepdims=True)
    scores["data"] = np.divide(exps,sums)
    return scores

  def grad(self,wrapper,**arg):
    if wrapper["grad_back"] == "relu":
      X,y   = arg["X"],arg["y"]
      W1,W2 = arg["W1"], arg["W2"]
      reg   = arg["reg"]

      inter = y.dot(W2.T)
      inter[wrapper["data"]<=0] = 0

      dW = X.T.dot(inter) + W1 * reg
      db = np.sum(inter, axis=0, keepdims=True)

      return dW,db

    elif wrapper["grad_back"] == "softmax":
      X,y = arg["X"]["data"],arg["y"]
      W,reg = arg["W"],arg["reg"]
      
      inter = wrapper["data"]
      dW = X.T.dot(inter) + W*reg
      db = np.sum(inter, axis=0, keepdims=True)
    
      return dW,db