import numpy as np
import matplotlib.pyplot as plt


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
    self.params['W1'] = std * np.random.randn(input_size, hidden_size).astype(np.float64)
    self.params['b1'] = np.zeros(hidden_size).astype(np.float64)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size).astype(np.float64)
    self.params['b2'] = np.zeros(output_size).astype(np.float64)

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
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


    loss,scores = self.sqerror(y,layer2,reg)
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
    train_err_history = []
    val_err_history = []
    
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']

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

      W1 -= grads["W1"] * learning_rate
      W2 -= grads["W2"] * learning_rate
      b1 -= grads["b1"] * learning_rate
      b2 -= grads["b2"] * learning_rate
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
        print ('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val error and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check error
        # print(self.predict(X_batch))
        # print(y_batch)
        train_err = np.sum(np.square(self.predict(X_batch) - y_batch), axis=1).mean()
        val_err = np.sum(np.square(self.predict(X_val) - y_val), axis=1).mean()
        train_err_history.append(train_err)
        val_err_history.append(val_err)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_err_history': train_err_history,
      'val_err_history': val_err_history,
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
    y_pred = self.loss(X)
    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    pass
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred
  
  def wrapit(self,data):
    return {"data":data}

  def sqerror(self,y,wrapper,reg):
    inter = wrapper["data"] - y
    loss  = np.sum(inter**2 /(2*y.shape[0]))
    # loss += reg * (np.sum(self.params["W1"]**2) + np.sum(self.params["W2"]**2)) / 2 # OVERFLOW PROBLEM
    loss += reg * np.sum(self.params["W1"]**2) / 2  + reg *np.sum(self.params["W2"]**2) / 2
    return loss, inter

  def linear(self,wrapper,W,b=None):
    scores = {"grad_back":"linear"}
    X = wrapper["data"]
    scores["data"] = X.dot(W) + b
    return scores

  def relu(self,wrapper):
    scores  = {"grad_back":"relu"}
    wrapper["data"][wrapper["data"]<0] = 0
    scores["data"] = wrapper["data"]
    return scores

  def grad(self,wrapper,**arg):
    if wrapper["grad_back"] == "relu":
      W1,W2 = arg["W1"], arg["W2"]
      reg   = arg["reg"]
      X,y   = arg["X"],arg["y"]

      inter = y.dot(W2.T) / y.shape[0]
      inter[wrapper["data"]<=0] = 0

      dW = X.T.dot(inter) + W1*reg
      db = np.sum(inter, axis=0, keepdims=True)
    
    elif wrapper["grad_back"] == "linear":
      X = arg["X"]["data"]
      y = arg["y"]
      W = arg["W"]
      reg = arg["reg"]
      # print(wrapper["data"].shape)
      # print(X.shape)
      # print(y.shape)
      inter = X.T.dot(y) / y.shape[0] 
      # print(inter.shape)
      # print(W.shape)
      dW = inter + W*reg
      db = np.sum(y, axis=0, keepdims=True) / y.shape[0] 

    return dW,db