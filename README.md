# Artificial Neural Network implementation for classifying handwritten digits

The neural network in R is implemented having one hidden layer with two, three and four hidden units for classifying handwritten digits. The data for training and test were provided in different files. For training the neural network, we used first 500 rows from every input file. We have implemented neural network using Forward propagation as well as Backpropagation. The output from the Forward propagation is used in a Cost function for calculating the cost and then Backpropagation used for finding gradient. After implementing this, we have used an advanced optimizer “L-BFGS-B” for optimizing the weights which then further used for predicting validation as well test data sets.

The detailed discussions about the project can be found at the below link
(https://github.com/ajaskulkarni/Artificial-Neural-Network-implementation-for-classifying-handwritten-digits/blob/master/Report.pdf) 


## References
1) Machine Learning online course, Coursera, Andrew Ng
2) Building a Neural Network from scratch in R, Bath Machine Learning Meetup, Owen
Jones
3) R for Deep Learning (I): Build fully connected Neural Network from scratch, Peng Zhao
4) A visual and interactive guide to the basics of Neural Networks, J Alammar
5) https://www.rdocumentation.org/packages/stats/versions/3.4.1/topics/optim
6) Machine Learning, Tom Mitchell
