# Loan Analysis & Prediction

## How to run 
```
$ python3 run.py
```

## Jupyter notebook
```
$ jupyter notebook
```
http://localhost:8888/notebooks/prediction.ipynb


## Short Questions & Answers
Your models have been implemented and customers are now using them in production.
1. Imagine the credit risk use case above. You have two models: a logistic regression
model with an F1-score of 0.60 and a neural network with an F1-score of 0.63. Which
model would you recommend for the bank and why?

> I would recommend a logistic regression model for this use case.
  Because this is the binary classification problem, so the decision boundary can be linear or non-linear.
  A logistic regression is useful if we are working with a dataset where the classes are more or less linearly separable.
  And it is useful for structured data set and relatively very small data set sizes.
  It can be used for predicting the probability of an event like this use case.
  Outputs have a nice probabilistic interpretation, and the algorithm can be regularized to avoid overfitting.
  Meanwhile, a neural network is hard to interpret the output such as "black box".
  Instead, neural network is good for unstructured data sets like image, audio, and text.
  Neural network is useful for image processing and natural language processing and so on.


2. A customer wants to know which features matter for their dataset. They have several
models created in the DataRobot platform such as random forest, linear regression, and
neural network regressor. They also are pretty good at coding in Python so wouldn't
mind using another library or function. How do you suggest this customer gets feature
importance for their data?

> In terms of feature selection, we can suggest a sequential feature selection algorithm such as SBS(Sequential Backward Selection)
  to suggest which features can improve the performance of model.
  Or using a random forest, we can measure feature importance.
