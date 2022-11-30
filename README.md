# Naive_Bayes_Classifier_ML
we compute the Naive Bayes to train the model and compute the
accuracy for all the features in the q3.csv data. In the q3.py, the priors are calculated where each
feature is taken and classified as True or False for boolean values and the priors are calculated.
Then using the priors the posterior are calculated, for which likelihoods are taken. The fit method
creates a template for the model to train according to the data based on the features and values. If
the feature values are integers then the likelihood parameters mean and standard deviation are
calculated. In the predict function, the likelihoods are multiplied with the prior which is known
as the posteriors and appended to the probabilities list where all the predictions of each row in
the data except for the “ is spam column ”. After getting the predictions, the model is trained and
predictions are stored in a list. Then the classification errors are calculated where predictions list
and y_test which is obtained from “q3b.csv ” file, these two columns are compared and the
values which are different the error is incremented and divided by the total number of columns

After calculating and training the model with the Naive Bayes , the following output is obtained:
The accuracy for all the features is 88% and error is found to be 12%.

Here, randomly three columns: “in html”,”has my name”,”has sig” which are taken from the
q3b.csv data to test the accuracy of the model. For the subset chosen, the values are again
predicted are stored in the list. This list is again predicted with and the accuracy is calculated.
The accuracy calculated for the above subset is 61.6%

