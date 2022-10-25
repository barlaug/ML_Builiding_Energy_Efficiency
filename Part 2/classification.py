# In this script, the heating load is classified and the result is evaluated statistically
# We compare a baseline, logistic regression model and an (ANN/CT/KNN/NB)

# 1: create baseline, logistic regression and method 2.
# For LR, use lambda as a complexity-controlling parameter.
# For meth2, figure out model complexity by trial and error
# For baseline: Compute the largest class on the training data, and predict
# everything in the test data as belonging to that class. 
# -> corresponding to log regr with bias term and no features

# 2: Use two level cross validation to creata table similar to report,
# comparing LR, baseline and meth2
# Error measure is the error rate = num_misclassified / len(testset)


# 3: Compare the three models pair-wise. Choose either:
# McNemeras test (section 11.3), or
# the one from section 11.4.
# Include p-vals and conf.intervals for all pairwise tests


# 4: Train logistic regression model using suitable lambda. 