
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, cross_val_score

#import metadata and inspect the dataframe


tracks = pd.read_csv("datasets/fma-rock-vs-hiphop.csv")
echonest_metrics = pd.read_json("datasets/echonest-metrics.json")
echo_tracks = echonest_metrics.merge(tracks[['genre_top','track_id']], on = 'track_id')
echo_tracks.info()

# Create a correlation matrix to avoid feature redundancy
corr_metrics = echo_tracks.corr()
corr_metrics.style.background_gradient()

# Normalize the data to perform features reduction
features = echo_tracks.drop(['genre_top','track_id'], axis = 1)
labels = echo_tracks.genre_top
scaler = StandardScaler()
scaled_train_features = scaler.fit_transform(features)
pca = PCA().fit(scaled_train_features)
exp_variance = pca.explained_variance_ratio_
print(exp_variance)

# plot the explained variance using a barplot to see the elbow-effect or 90% of the variance
fig, ax = plt.subplots()
ax.bar(range(0,8), exp_variance)
ax.set_xlabel('Principal Component #')



cum_exp_variance = np.cumsum(exp_variance)

# Plot the cumulative explained variance and draw a dashed line at 0.90.
fig, ax = plt.subplots()
ax.plot(cum_exp_variance)
ax.axhline(y=0.9, linestyle='--')
n_components = 6

# Perform PCA with the important components
pca = PCA(n_components, random_state=10)
pca.fit(scaled_train_features)
pca_projection = pca.transform(scaled_train_features)

# Comparing between decision tree and logistic regression machine learning algorithms
train_features, test_features, train_labels, test_labels = train_test_split(pca_projection,labels, test_size = None, random_state = 10)
tree = DecisionTreeClassifier(random_state = 10)
tree.fit(train_features,train_labels)
pred_labels_tree = tree.predict(test_features)

#logistic regression
logreg = LogisticRegression(random_state = 10)
logreg.fit(train_features, train_labels)
pred_labels_logit = logreg.predict(test_features)

# Create the classification report for both models
class_rep_tree = classification_report(test_labels,pred_labels_tree)
class_rep_log = classification_report(test_labels, pred_labels_logit)

print("Decision Tree: \n", class_rep_tree)
print("Logistic Regression: \n", class_rep_log)

# As classification of rock songs does better than hip-hop songs. This could possibly due to large number of data points for rock classification.
# To check whether the algorithms work with both classifications, it is desired to sample the same data points for each genre.

hop_only = echo_tracks.loc[echo_tracks['genre_top']=='Hip-Hop']
rock_only = echo_tracks.loc[echo_tracks['genre_top']=='Rock']
rock_only = rock_only.sample(n = 910, random_state=10)
rock_hop_bal = pd.concat([rock_only,hop_only])
print(rock_hop_bal.shape)

# perform features reduction and reapply machine learning algorithms on new dataframe to reexamine the result
features = rock_hop_bal.drop(['genre_top', 'track_id'], axis=1) 
labels = rock_hop_bal['genre_top']
pca_projection = pca.fit_transform(scaler.fit_transform(features))
train_features, test_features, train_labels, test_labels = train_test_split(pca_projection,labels, random_state=10)

tree = DecisionTreeClassifier(random_state = 10)
tree.fit(train_features,train_labels)
pred_labels_tree = tree.predict(test_features)

logreg = LogisticRegression(random_state=10)
logreg.fit(train_features,train_labels)
pred_labels_logit = logreg.predict(test_features)

# Compare the models
print("Decision Tree: \n", classification_report(test_labels,pred_labels_tree))
print("Logistic Regression: \n", classification_report(test_labels, pred_labels_logit))

# we will use K-fold cross-validation to systematically evaluate the models

kf = KFold(n_splits=10, random_state = 10)
tree = DecisionTreeClassifier(random_state=10)
logreg = LogisticRegression(random_state=10)

tree_score = cross_val_score(tree,pca_projection,labels,cv=kf)
logit_score = cross_val_score(logreg,pca_projection,labels,cv=kf)

# Compare each fold to see which model does better
print("Decision Tree:", tree_score, "Logistic Regression:", logit_score)


