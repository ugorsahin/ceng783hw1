import numpy as np
import matplotlib.pyplot as plt


class LinearClassifier(object):
  """
  A linear classifier with hinge loss or softmax loss.
  """

  def __init__(self, input_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W: Layer weights; has shape (C, D)  -- bias is included here.   

    Inputs:
    - input_size: The dimension D of the input data.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W'] = std * np.random.randn(output_size, input_size) # This was is easier; we get rid of taking the transpose in multiplications    
  
  def naive_hinge_loss(self, X, y=None, W=None, reg=0.0, delta=1):
    """
    Compute the loss and gradients.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.
    - loss_type: 'hinge' for hinge loss, 'xentropy' for cross-entropy loss.

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
    if W is None: W = self.params['W']
    N, D = X.shape
    
    scores = None
    loss = 0.0
    
    dW = np.zeros(W.shape)

    num_classes = W.shape[0]
    num_train = X.shape[0]

    for i in range(num_train):
      scores   = W.dot(X[i])   # W . X_i  (bias is included in W & X
      correct_ = scores[y[i]] # score for the correct labelf
      multip   = 0
      
      for class_no in range(num_classes):
        if class_no == y[i]: continue

        margin = scores[class_no] - correct_ + delta 
        
        if margin > 0:
          multip +=1
          loss += margin
          dW[class_no] += X[i]

      dW[y[i]] -= multip * X[i]

    dW /= num_train
    dW += reg * W
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)

    #############################################################################
    # TODO: Compute the gradient of the loss function and store it dW.          #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # pass
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################  

    return loss, dW

  def vectorized_hinge_loss(self, X, y=None, W=None, reg=0.0, delta=1):
    """
    Compute the loss and gradients.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.
    - delta: margin in hinge-loss definition.

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
    if W is None: W = self.params['W'] # has shape C x D
    N, D = X.shape

    num_train = X.shape[0]
    dW = np.zeros(W.shape)
    
    # Compute the forward pass
    rangeVal  = np.arange(num_train)

    scores    = X.dot(W.T)
    correct_  = np.expand_dims(scores[rangeVal,y],axis=1)
    margin_   = np.maximum(0,scores - correct_ + 1)

    margin_[rangeVal, y] = 0
    
    loss = np.sum(margin_) / num_train
    loss += reg * np.sum(W*W) / 2
    
    if y is None:
      return scores

    intermediate = np.ones(margin_.shape)
    intermediate[margin_ <= 0] = 0

    intermediate[rangeVal, y] = -1 * np.sum(intermediate, axis=1)

    dW = X.T.dot(intermediate)
    dW = dW / num_train + reg*W.T
    dW = dW.T

    return loss, dW
    
  def vectorized_xentropy_loss(self, X, y=None, W=None, reg=0.0, b=None):
    """
    Compute the loss and gradients.

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
    if W is None: W = self.params['W'] # has shape C x D
    N, D = X.shape
    # Compute the forward pass
    scores = self.softmax(X,W)
    if b: scores += b
    if y is None: return scores

    rangeVal = np.arange(y.shape[0])
    logs = -np.log(scores[np.arange(y.shape[0]),y])
    loss =  np.sum(logs) / y.shape[0]

    inter = scores
    inter[rangeVal,y] -=1
    inter /= y.shape[0]
    dW = X.T.dot(inter).T + W*reg

    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    # loss = None
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W. Store the result  	#
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss. So that your results match ours, multiply the            #
    # regularization loss by 0.5                                                #
    #############################################################################
    pass
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    # grad = None
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights.#
    # Store the results in the `grad' variable, which should be a matrix that   #
    # has the same size as 'W' 							#
    #############################################################################
    pass
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, dW   
    
  def softmax(self,X,W=None):
    if W is None : W = self.params["W"]
    scores = X.dot(W.T)
    exps = np.exp(scores - np.expand_dims(np.argmax(scores,axis=1),axis=1))
    # exps = np.exp(scores)
    sums  = np.sum(exps,axis=1,keepdims=True)
    nrml = np.divide(exps,sums)
    return nrml

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, delta=1, num_iters=100,
            batch_size=200, loss='hinge', verbose=False):
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
    - loss: hinge or xentropy
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train // batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    # np.random.shuffle(X)
    # np.random.shuffle(y)
    # datas = iter(BatchMaker(X,y,batch_size))
    W_ = self.params["W"]

    it = 0
    for it in range(num_iters):
      rand_vals = np.random.choice(num_train,batch_size,replace=False)
      X_batch,y_batch = X[rand_vals],y[rand_vals]
      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      # pass 
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      if loss == 'hinge':
      	loss_val, grad = self.vectorized_hinge_loss(X_batch, y=y_batch, reg=reg, delta=delta)
      elif loss == 'xentropy':
      	loss_val, grad = self.vectorized_xentropy_loss(X_batch, y=y_batch, reg=reg)
      else:
      	print("Unknown loss function ('%s')" % (loss))
      
      W_ -=  grad * learning_rate
      loss_history.append(loss_val)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grad variable defined above.                            #
      #########################################################################
      
      # pass
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print ('\riteration %d / %d: loss %f' % (it, num_iters, loss_val),end="")

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch, loss) == y_batch).mean()
        val_acc = (self.predict(X_val, loss) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay
    print ('\riteration %d / %d: loss %f' % (it, num_iters, loss_val),end="")
    print("\n")
    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X, loss='hinge'):
    founds = X.dot(self.params["W"].transpose())
    y_pred = np.argmax(founds,axis=1)

    """
    Use the trained weights of the model to predict labels for
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

    # y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    # Hint: Look up numpy's argmax function				      #
    ###########################################################################
    pass 
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred
