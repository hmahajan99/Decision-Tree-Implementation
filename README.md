# Decision Tree Implementation
A python 3 implementation of decision tree commonly used in machine learning classification problems. Currently, only discrete datasets can be learned.
(The algorithm treats continuous valued features as discrete valued ones)
## Features
You can fit the classifier over the training data(using either gain ratio or gini index as metric), make predictions and get the score(mean accuracy) for testing data as well.
**The machine learning decision tree model after fitting the training data can be exported into a PDF.**
On **comparison of inbuilt sklearn's decision tree** with our model on the same training data, the results were similar.