# Implementation of Artificial Neural Network in R
# 4 units

# Importing training data
train_data = read.csv("train_data.csv", check.names = FALSE)

# Check structure and view the data
str(train_data)
View(train_data)

# ------ Training and validation data prepration ------ #

# Training data without labels
x_train= train_data[-1]
x_train = as.matrix(x_train)
dim(x_train)

# Training data labels
y_train = train_data[,1]
y_train = as.numeric(y_train)
# Find reason
y_train[y_train == 0] = 10

# Validation data
v1 = train_data[1:100,]
v2 = train_data[501:600,]
v3 = train_data[1001:1100,]
v4 = train_data[1501:1600,]
v5 = train_data[2001:2100,]
v6 = train_data[2501:2600,]
v7 = train_data[3001:3100,]
v8 = train_data[3501:3600,]
v9 = train_data[4001:4100,]
v10 = train_data[4501:4600,]

val_data = rbind(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10)
dim(val_data)

# Validation data without labels
x_val= val_data[-1]
x_val = as.matrix(x_val)
dim(x_val)

# Validation data labels
y_val = val_data[,1]
y_val = as.numeric(y_val)
y_val[y_val == 0] = 10

# ------ Parameter intialization ------ #

# Total number of input layers
input_layer_size = dim(x_train)[2]

# Total number of output layers
output_layer_size = length(unique(y_train))

# Number of neurons present in the hidden layer
hidden_layer_size = 4

# Genreating normally distributed random numbers which will be used as weights. 
# Size will be depend upon input layer size, bias and hidden layer size
# So, we will define a vector of the size 
# ((input_layer_size + 1)* hidden_layer_size) + ((hidden_layer_size + 1) * output_layer_size)

n = ((input_layer_size + 1) * hidden_layer_size) + 
  ((hidden_layer_size + 1) * output_layer_size)

# Setting seed so it will genreate same weigths in every iteration
set.seed(1)
wt = runif(n)

# ------ Activation function -> sigmoid function ------ #

sigmoid <- function(x) {1 / (1 + exp(-x))}

# -------------------------------------------------------

# ------ Cost function ------ #

cost_fun = function(wt, input_layer_size, hidden_layer_size, output_layer_size, x, y, lambda) {
  # Defining the weight matrix for the neural network. The first layer will have 
  # inputs from all the input nodes and bias. The second layer will have inputs only 
  # from hidden units and bias. Here weight1 will represent weights for layer 1 and
  # weight2 will represent weights for layer 2.
  k = (input_layer_size + 1) * hidden_layer_size
  w1 = matrix(wt[1:k], hidden_layer_size, input_layer_size + 1)
  w2 = matrix(wt[(k + 1):length(wt)],output_layer_size, hidden_layer_size + 1)
  
  # Total number of rows in data
  n = dim(x)[1]
  
  # Adding bias to each example
  b1 = cbind(rep(1,n), x)
  # Multiplying weigths and the inputs (layer - 1)
  o1 = b1 %*% t(w1)
  # Applying activation function on the calculated raw output
  o1 = sigmoid(o1)
  
  # Adding bias unit in the input for the next layer
  b2 = cbind(rep(1,n), o1)
  # Multiplying weigths and the inputs (layer - 2)
  o2 = b2 %*% t(w2)
  
  # Applying activation function on the calculated raw output
  final_output = sigmoid(o2)
  
  diag_matrix = diag(output_layer_size)
  
  # Calulate cost and regularization of cost function
  cost1 = (-1/n) * sum (log(final_output) * t(diag_matrix[,y]) + 
                          log(1-final_output) * t((1- diag_matrix[,y]))) 
  cost = cost1 + lambda/(2*n) * (sum(w1[, -c(1)]^2) 
                                 + sum(w2[, -c(1)]^2))
  cost
}

# -------------------------------------------------------

# t1 = cost_fun(wt, input_layer_size, hidden_layer_size, output_layer_size, x_train, y_train, 1)
# 19.87897

# ------ Derivative of sigmoid function ------ #

siggrad =  function(z) {
  temp <- 1 / (1 + exp(-z))
  temp * (1 - temp)
}

# -------------------------------------------------------

# ------ Partial derivatives of the cost fucntion with respect to w1 and w2 ------ #

p_grad = function(wt, input_layer_size, hidden_layer_size, output_layer_size,
                  x, y, lambda) {
  
  # Intializing weights for layer1 and layer 2
  k <- (input_layer_size + 1) * hidden_layer_size
  w1 <- matrix(wt[1:k], hidden_layer_size, input_layer_size + 1)
  w2 <- matrix(wt[(k + 1):length(wt)], output_layer_size, hidden_layer_size + 1)
  
  # Total number of rows in data
  n = dim(x)[1]
  
  # Adding bias to each example
  b1 = cbind(rep(1,n), x)
  # Multiplying weigths and the inputs (layer - 1)
  o1 = b1 %*% t(w1)
  # Applying activation function on the calculated raw output
  o1_1 = sigmoid(o1)
  
  # Adding bias unit in the input for the next layer
  b2 = cbind(rep(1,n), o1_1)
  # Multiplying weigths and the inputs (layer - 2)
  o2 = b2 %*% t(w2)
  
  # Applying activation function on the calculated raw output
  final_output = sigmoid(o2)
  
  diag_matrix = diag(output_layer_size)
  
  # ----- Backpropogation ----- #
  
  # Matrix of actual output
  y_matrix = diag_matrix[y,]
  
  # Calculating error in output layer by subtracting actual output and 
  # predicted output
  d3 = final_output - y_matrix
  
  # Calculating error in second layer
  d2 <- (d3 %*% w2[, -c(1)]) * siggrad(o1)
  
  # Calculating delta values for each layer
  delta1 = t(d2) %*% b1
  delta2 = t(d3) %*% b2
  
  # ----- Gradient regularization ----- #
  
  # Calculating regularization terms
  reg1 = lambda/n * w1
  reg1[, 1] = 0
  reg2 = lambda/n * w2
  reg2[,1] = 0
  
  # Adding calculated regularization terms into delta matrices
  w1_grad = 1/n * delta1 + reg1
  w2_grad = 1/n * delta2 + reg2
  
  # Calculated gradient
  grad = c(as.vector(w1_grad), as.vector(w2_grad))
  grad
}

#kk = p_grad(wt, input_layer_size, hidden_layer_size, output_layer_size,
#            x_train, y_train, 1)

# -------------------------------------------------------

# ------ Predict function ------ #

predict <- function(w1, w2, x) {
  
  # Number of rows present in the test data
  n <- dim(x)[1]
  
  # Adding bias 
  p1_1 = cbind(rep(1,n),x)
  # Multiplying inputs with calculated weights
  p1_2 = p1_1 %*% t(w1)
  # Applying sigmoid function on the output
  p1 = sigmoid(p1_2)
  
  # Adding bias 
  p2_1 = cbind(rep(1,n),p1)
  # Multiplying inputs with calculated weights
  p2_2 = p2_1 %*% t(w2)
  # Applying sigmoid function on the output
  p2 = sigmoid(p2_2)
  # Returning the columns having maximum predicted ouput
  max.col((p2))
}

# ------- Model execution ------- #

# Intialzing value to lambda
lambda = 1
# Optimizing cost function and calculating optimum weights for 
# neural network

# Options for optimization - "Nelder-Mead", "BFGS", "CG", "L-BFGS-B", "SANN",
# "Brent"
opt_out <- optim(wt,
                 function(p) cost_fun(p, 
                                      input_layer_size,
                                      hidden_layer_size,
                                      output_layer_size,
                                      x_train, 
                                      y_train, 
                                      lambda),
                 function(p) p_grad(p, 
                                    input_layer_size,
                                    hidden_layer_size,
                                    output_layer_size,
                                    x_train, 
                                    y_train,
                                    lambda),
                 method = "L-BFGS-B",
                 control = list(maxit = 500))
# Storing optimized weights in new variable which will be used 
# for prediction
new_weight <- opt_out[[1]]

# Updating weight matrices which are calculated from training model
k = (input_layer_size + 1) * hidden_layer_size
w1 = matrix(new_weight[1:k], hidden_layer_size, input_layer_size + 1)
w2 = matrix(new_weight[(k + 1):length(new_weight)], output_layer_size, hidden_layer_size + 1)

# -------- Predictions on cross-validation set -------- #

# For lambda = 1
# Call to prediction function
preds = predict(w1, w2, x_val)
# Confusion matrix and percentage accuracy
table(preds, y_val)
sum(preds == y_val) * 100 / dim(x_val)[1]
# Accuracy = 92.6%

# Selecting value of lambda
acc = 0
for (i in seq(1,20,1)){
  lambda = i
  opt_out <- optim(wt,
                   function(p) cost_fun(p, input_layer_size,
                                        hidden_layer_size,
                                        output_layer_size,
                                        x_train, y_train, lambda),
                   function(p) p_grad(p, input_layer_size,
                                      hidden_layer_size,
                                      output_layer_size,
                                      x_train, y_train, lambda),
                   method = "L-BFGS-B", control = list(maxit = 500))
  # Storing optimized weights in new variable which will be used 
  # for prediction
  new_weight <- opt_out[[1]]
  
  # Updating weight matrices which are calculated from training model
  k = (input_layer_size + 1) * hidden_layer_size
  w1 = matrix(new_weight[1:k], hidden_layer_size, input_layer_size + 1)
  w2 = matrix(new_weight[(k + 1):length(new_weight)], output_layer_size, hidden_layer_size + 1)
  preds = predict(w1, w2, x_val)
  acc[i] = sum(preds == y_val) * 100 / dim(x_val)[1]
}
acc_LBFGS_B_4 = acc
write.csv(acc_LBFGS_B_4, "acc_LBFGS_B_4.csv")

# Calculating accuracy for lambda = 1
lambda = 1
opt_out <- optim(wt,
                 function(p) cost_fun(p, input_layer_size,
                                      hidden_layer_size,
                                      output_layer_size,
                                      x_train, y_train, lambda),
                 function(p) p_grad(p, input_layer_size,
                                    hidden_layer_size,
                                    output_layer_size,
                                    x_train, y_train, lambda),
                 method = "L-BFGS-B", control = list(maxit = 500))
# Storing optimized weights in new variable which will be used 
# for prediction
new_weight <- opt_out[[1]]

# Updating weight matrices which are calculated from training model
k = (input_layer_size + 1) * hidden_layer_size
w1 = matrix(new_weight[1:k], hidden_layer_size, input_layer_size + 1)
w2 = matrix(new_weight[(k + 1):length(new_weight)], output_layer_size, hidden_layer_size + 1)
preds = predict(w1, w2, x_val)
sum(preds == y_val) * 100 / dim(x_val)[1]
# Accuracy = 92.6%

# ------ Testing model on test data ------ #

test_data = read.csv("test_data.csv", check.names = FALSE)

# Training data
x_test= test_data[-1]
x_test = as.matrix(x_test)
dim(x_test)

# Trainind data labels
y_test = test_data[,1]
y_test = as.numeric(y_test)
y_test[y_test == 0] = 10

preds1 <- predict(w1, w2, x_test)
y_test[y_test == 10] = 0
preds1[preds1 == 10] = 0
table(preds1, y_test)
sum(preds1 == y_test) * 100 / dim(x_test)[1]
# Accuracy = 75.32%

# ---------------------------------------------------------------------------

# ----- Confidence Interval  ----- # 

miss_label = length(y_test) - sum(preds1 == y_test)
miss_label 
# 1234

m = miss_label/length(y_test)
m
#  0.2468

v = length(y_test)*m*(1-m)
v
# 929.4488

v1 = sqrt(v)
v1
# 30.48686

s = v1/n
s
# 0.00955701

# Now we need to find 95% confidence interval
u = m + 1.96*s
l = m - 1.96*s
u
# 0.2655317
l
# 0.2280683

