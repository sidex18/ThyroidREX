# %% 
import pandas as pd
import numpy as np

excel_file_path = 'features_with_labels.xlsx' # Address to your excel file
df= pd.read_excel(excel_file_path)

# %%
from sklearn.model_selection import train_test_split

# Splitting the data into training and testing sets
X = df.iloc[:, 2:]
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# -------------------------------------------------

# %%
# Normalizing the data
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def scale_data(X_train_continuous, X_test_continuous):

    # fit on training data column
    scale = StandardScaler().fit(X_train_continuous)

    names = X_train_continuous.columns

    # transform the training data column
    X_train_norm1 = scale.transform(X_train_continuous)
    X_train_norm = pd.DataFrame(X_train_norm1, columns=names) 

    # transform the testing data column
    X_test_norm1 = scale.transform(X_test_continuous)
    X_test_norm = pd.DataFrame(X_test_norm1, columns=names) 

    return X_train_norm, X_test_norm
  
X_train_norm, X_test_norm = scale_data(X_train, X_test)  

# -------------------------------------------------

# %%
# Removing highly correlated features using Spearman
import scipy.stats

def remove_highly_correlated_features(X_train_norm, X_test_norm, threshold=0.90):
    # Calculate Spearman's correlation matrix
    correlation_matrix = X_train_norm.corr(method='spearman').abs()

    # Create a mask to identify highly correlated features
    mask = (correlation_matrix >= threshold) & (correlation_matrix < 1.0)

    # Identify the pairs of highly correlated features
    correlated_features = set()
    for feature in mask.columns:
        correlated_features.update(set(mask.index[mask[feature]].tolist()))

    # Remove one feature from each correlated pair
    features_to_remove = set()
    for feature in correlated_features:
        if feature not in features_to_remove:
            correlated_pair = mask.index[mask[feature]].tolist()
            features_to_remove.update(set(correlated_pair))

    # Drop the identified features from the dataframe
    X_train_filtered = X_train_norm.drop(columns=features_to_remove)
    X_test_filtered = X_test_norm.drop(columns=features_to_remove)

    return X_train_filtered, X_test_filtered

X_train_filtered, X_test_filtered = remove_highly_correlated_features(X_train_norm, X_test_norm, threshold=0.90)

# -------------------------------------------------

# %%
from imblearn.over_sampling import SMOTE

# Synthetic Minority Oversampling Technique (SMOTE)

# Initialize SMOTE
smote = SMOTE(k_neighbors=5, sampling_strategy='auto',  random_state=42)

# Perform SMOTE on X_train_filtered
X_train_filtered, y_train = smote.fit_resample(X_train_filtered, y_train)

# Now X_train_filtered_resampled and y_train_resampled contain the resampled data

# -------------------------------------------------

# %%
# Feature Selection using Recursive Feature Elimination (RFE) with the XGBoost classifier as the estimator
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier
import pandas as pd
import matplotlib.pyplot as plt

# Define your classifier
classifier = XGBClassifier()

# Initialize RFE
rfe = RFE(estimator=classifier, n_features_to_select=10)  # Adjust number of features as needed

# Fit RFE
rfe.fit(X_train_filtered, y_train.values.ravel())

# Get the selected features
selected_features = X_train_filtered.columns[rfe.support_]

# Create filtered datasets
X_train_selected = X_train_filtered[selected_features]
X_test_selected = X_test_filtered[selected_features]

print("Selected features using RFE:", selected_features)

# Get feature importances from the classifier used in RFE
feature_importances = pd.Series(
    rfe.estimator_.feature_importances_, 
    index=X_train_selected.columns
).sort_values(ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
feature_importances.head(20).plot(kind='bar')
plt.title("Top Feature Importances")
plt.ylabel("Importance Score")
plt.xlabel("Features")
plt.show()

# -------------------------------------------------


# %%
# import pymrmr
# import pandas as pd
# from sklearn.feature_selection import mutual_info_classif
# import matplotlib.pyplot as plt

# # Combine features and target
# data = pd.concat([X_train_filtered, y_train], axis=1)

# # Rename the target column
# # data.columns = [*X_train_filtered.columns, 'target']

# # Handle missing values
# data.fillna(data.mean(), inplace=True)  # Impute missing values

# # Ensure all data is numeric
# if not all(data.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
#     data = pd.get_dummies(data, drop_first=True)

# # Perform mRMR feature selection
# selected_features_mrmr = pymrmr.mRMR(data, 'MIQ', 11)  # Select top 10 features
# print("Selected features using mRMR:", selected_features_mrmr)


# # Use the selected features to filter the datasets
# # Remove 'Label' from selected_features_mrmr if it exists
# if 'Label' in selected_features_mrmr:
#     selected_features_mrmr.remove('Label')
# X_train_selected_mrmr = X_train_filtered[selected_features_mrmr]
# X_test_selected_mrmr = X_test_filtered[selected_features_mrmr]

# print("X_train_selected_mrmr shape:", X_train_selected_mrmr.shape)
# print("X_test_selected_mrmr shape:", X_test_selected_mrmr.shape)

# # Calculate mutual information for all features
# mutual_info = mutual_info_classif(X_train_filtered, y_train.values.ravel(), random_state=42)

# # Create a DataFrame of feature importance
# feature_importances = pd.DataFrame({
#     'Feature': X_train_filtered.columns,
#     'Mutual Information': mutual_info
# }).sort_values(by='Mutual Information', ascending=False)

# # Highlight selected features in the top list
# feature_importances['Selected'] = feature_importances['Feature'].isin(selected_features_mrmr)

# # Visualization
# plt.figure(figsize=(12, 6))
# selected = feature_importances[feature_importances['Selected'] == True]
# plt.barh(selected['Feature'], selected['Mutual Information'], color='teal', label='Selected')
# plt.barh(
#     feature_importances[~feature_importances['Selected']]['Feature'],
#     feature_importances[~feature_importances['Selected']]['Mutual Information'],
#     color='lightgrey',
#     label='Not Selected'
# )
# plt.title('Feature Importance (Mutual Information)')
# plt.xlabel('Mutual Information Score')
# plt.ylabel('Features')
# plt.legend()
# plt.show()

# -------------------------------------------------


# %%
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_curve, auc, classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from scipy import interp

# Assuming you have the data ready:
# X_train_selected, y_train, X_test_selected, y_test

# Binarize the labels for ROC curve plotting
y_train_binarized = label_binarize(y_train, classes=np.unique(y_train))
y_test_binarized = label_binarize(y_test, classes=np.unique(y_train))

# param_grid_xgb = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [3, 4, 5],
#     'learning_rate': [0.1, 0.01, 0.001],
#     'min_child_weight': [1, 3, 5],
#     'gamma': [0, 0.1, 0.3],
#     'subsample': [0.6, 0.8, 1.0],
#     'colsample_bytree': [0.6, 0.8, 1.0],
#     'reg_alpha': [0, 0.1, 1, 10],
#     'reg_lambda': [0, 0.1, 1, 10]
# }

param_grid_xgb = {
    'n_estimators': [300],
    'max_depth': [5],
    'learning_rate': [0.001],
    'min_child_weight': [5],
    'gamma': [0.3],
    'subsample': [1.0],
    'colsample_bytree': [1.0],
    'reg_alpha': [1],
    'reg_lambda': [1]
}

# Initialize the XGBoost classifier
xgb = XGBClassifier(random_state=42)

# Configure GridSearchCV
grid_search_xgb = GridSearchCV(estimator=xgb, param_grid=param_grid_xgb, cv=5, scoring='accuracy', verbose=1)

# Execute the grid search
grid_search_xgb.fit(X_train_selected, y_train.values.ravel())

# Print the best parameters and best score
print("Best parameters:", grid_search_xgb.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search_xgb.best_score_))

# Use the best model found by GridSearchCV
best_xgb = grid_search_xgb.best_estimator_

# Predictions and probabilities on the test set
y_pred_test = best_xgb.predict(X_test_selected)
y_pred_proba_test = best_xgb.predict_proba(X_test_selected)

# Calculate ROC AUC for each class
n_classes = y_train_binarized.shape[1]
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_proba_test[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), y_pred_proba_test.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curves for each class
plt.figure(figsize=(8, 6))
colors = cycle(['blue', 'red', 'green', 'purple'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for XGBoost - Multi-Class')
plt.legend(loc="lower right")
plt.show()

# Evaluation Metrics on test
print("Accuracy on Test Set: {:.2f}".format(accuracy_score(y_test, y_pred_test)))
print("Classification Report:")
print(classification_report(y_test, y_pred_test))

# Confusion matrix for the test set
conf_matrix_test = confusion_matrix(y_test, y_pred_test)
print("Confusion Matrix (Test Set):\n", conf_matrix_test)

# %%
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import accuracy_score, roc_curve, auc, classification_report, confusion_matrix
# from sklearn.preprocessing import label_binarize
# import numpy as np
# import matplotlib.pyplot as plt
# from itertools import cycle
# from scipy import interp

# # Assuming you have the data ready:
# # X_train_selected, y_train, X_test_selected, y_test

# # Binarize the labels for ROC curve plotting
# y_train_binarized = label_binarize(y_train, classes=np.unique(y_train))
# y_test_binarized = label_binarize(y_test, classes=np.unique(y_train))

# # Parameter grid for Random Forest
# param_grid_rf = {
#     'n_estimators': [100, 200, 300],  # Number of trees in the forest
#     'max_depth': [5, 10, 15],  # Maximum depth of each tree
#     'min_samples_split': [2, 5, 10],  # Minimum samples required to split a node
#     'min_samples_leaf': [1, 2, 4],  # Minimum samples required at a leaf node
#     'max_features': ['auto', 'sqrt', 'log2'],  # Number of features to consider at each split
#     'bootstrap': [True, False]  # Whether bootstrap samples are used when building trees
# }


# # Initialize the Random Forest classifier
# rf = RandomForestClassifier(random_state=42)

# # Configure GridSearchCV
# grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, scoring='accuracy', verbose=1)

# # Execute the grid search
# grid_search_rf.fit(X_train_selected, y_train.values.ravel())

# # Print the best parameters and best score
# print("Best parameters:", grid_search_rf.best_params_)
# print("Best cross-validation score: {:.2f}".format(grid_search_rf.best_score_))

# # Use the best model found by GridSearchCV
# best_rf = grid_search_rf.best_estimator_

# # Predictions and probabilities on the test set
# y_pred_test = best_rf.predict(X_test_selected)
# y_pred_proba_test = best_rf.predict_proba(X_test_selected)

# # Calculate ROC AUC for each class
# n_classes = y_train_binarized.shape[1]
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# for i in range(n_classes):
#     fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_proba_test[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])

# # Micro-average ROC curve and ROC area
# fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), y_pred_proba_test.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# # Plot ROC curves for each class
# plt.figure(figsize=(8, 6))
# colors = cycle(['blue', 'red', 'green', 'purple'])
# for i, color in zip(range(n_classes), colors):
#     plt.plot(fpr[i], tpr[i], color=color, lw=2,
#              label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

# plt.plot([0, 1], [0, 1], 'k--', lw=2)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic for Random Forest - Multi-Class')
# plt.legend(loc="lower right")
# plt.show()

# # Evaluation Metrics on test
# print("Accuracy on Test Set: {:.2f}".format(accuracy_score(y_test, y_pred_test)))
# print("Classification Report:")
# print(classification_report(y_test, y_pred_test))

# # Confusion matrix for the test set
# conf_matrix_test = confusion_matrix(y_test, y_pred_test)
# print("Confusion Matrix (Test Set):\n", conf_matrix_test)

# -------------------------------------------------

# %%
# from sklearn.svm import SVC
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import accuracy_score, roc_curve, auc, classification_report, confusion_matrix
# from sklearn.preprocessing import label_binarize
# import numpy as np
# import matplotlib.pyplot as plt
# from itertools import cycle
# from scipy import interp

# # Assuming you have the data ready:
# # X_train_selected, y_train, X_test_selected, y_test

# # Binarize the labels for ROC curve plotting
# y_train_binarized = label_binarize(y_train, classes=np.unique(y_train))
# y_test_binarized = label_binarize(y_test, classes=np.unique(y_train))

# # Parameter grid for SVM
# param_grid_svm = {
#     'C': [0.1, 1, 10],  # Regularization parameter
#     'kernel': ['linear', 'rbf'],  # Type of kernel
#     'gamma': ['scale', 'auto'],  # Kernel coefficient
#     'degree': [3, 4, 5],  # Degree for the polynomial kernel
#     'class_weight': ['balanced', None]  # Handling class imbalance
# }

# # Initialize the SVM classifier
# svm = SVC(probability=True, random_state=42)

# # Configure GridSearchCV
# grid_search_svm = GridSearchCV(estimator=svm, param_grid=param_grid_svm, cv=5, scoring='accuracy', verbose=1)

# # Execute the grid search
# grid_search_svm.fit(X_train_selected, y_train.values.ravel())

# # Print the best parameters and best score
# print("Best parameters:", grid_search_svm.best_params_)
# print("Best cross-validation score: {:.2f}".format(grid_search_svm.best_score_))

# # Use the best model found by GridSearchCV
# best_svm = grid_search_svm.best_estimator_

# # Predictions and probabilities on the test set
# y_pred_test = best_svm.predict(X_test_selected)
# y_pred_proba_test = best_svm.predict_proba(X_test_selected)

# # Calculate ROC AUC for each class
# n_classes = y_train_binarized.shape[1]
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# for i in range(n_classes):
#     fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_proba_test[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])

# # Micro-average ROC curve and ROC area
# fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), y_pred_proba_test.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# # Plot ROC curves for each class
# plt.figure(figsize=(8, 6))
# colors = cycle(['blue', 'red', 'green', 'purple'])
# for i, color in zip(range(n_classes), colors):
#     plt.plot(fpr[i], tpr[i], color=color, lw=2,
#              label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

# plt.plot([0, 1], [0, 1], 'k--', lw=2)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic for SVM - Multi-Class')
# plt.legend(loc="lower right")
# plt.show()

# # Evaluation Metrics on test
# print("Accuracy on Test Set: {:.2f}".format(accuracy_score(y_test, y_pred_test)))
# print("Classification Report:")
# print(classification_report(y_test, y_pred_test))

# # Confusion matrix for the test set
# conf_matrix_test = confusion_matrix(y_test, y_pred_test)
# print("Confusion Matrix (Test Set):\n", conf_matrix_test)

# %%
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import accuracy_score, roc_curve, auc, classification_report, confusion_matrix
# from sklearn.preprocessing import label_binarize
# import numpy as np
# import matplotlib.pyplot as plt
# from itertools import cycle
# from scipy import interp

# # Assuming you have the data ready:
# # X_train_selected, y_train, X_test_selected, y_test

# # Binarize the labels for ROC curve plotting
# y_train_binarized = label_binarize(y_train, classes=np.unique(y_train))
# y_test_binarized = label_binarize(y_test, classes=np.unique(y_train))

# # Parameter grid for Decision Tree
# param_grid_dt = {
#     'criterion': ['gini', 'entropy'],  # The function to measure the quality of a split
#     'max_depth': [None, 5, 10, 15],  # The maximum depth of the tree
#     'min_samples_split': [2, 5, 10],  # The minimum number of samples required to split an internal node
#     'min_samples_leaf': [1, 2, 4],  # The minimum number of samples required to be at a leaf node
#     'max_features': [None, 'sqrt', 'log2'],  # The number of features to consider when looking for the best split
#     'class_weight': [None, 'balanced']  # Whether to adjust weights for imbalanced classes
# }

# # Initialize the Decision Tree classifier
# dt = DecisionTreeClassifier(random_state=42)

# # Configure GridSearchCV
# grid_search_dt = GridSearchCV(estimator=dt, param_grid=param_grid_dt, cv=5, scoring='accuracy', verbose=1)

# # Execute the grid search
# grid_search_dt.fit(X_train_selected, y_train.values.ravel())

# # Print the best parameters and best score
# print("Best parameters:", grid_search_dt.best_params_)
# print("Best cross-validation score: {:.2f}".format(grid_search_dt.best_score_))

# # Use the best model found by GridSearchCV
# best_dt = grid_search_dt.best_estimator_

# # Predictions and probabilities on the test set
# y_pred_test = best_dt.predict(X_test_selected)
# y_pred_proba_test = best_dt.predict_proba(X_test_selected)

# # Calculate ROC AUC for each class
# n_classes = y_train_binarized.shape[1]
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# for i in range(n_classes):
#     fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_proba_test[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])

# # Micro-average ROC curve and ROC area
# fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), y_pred_proba_test.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# # Plot ROC curves for each class
# plt.figure(figsize=(8, 6))
# colors = cycle(['blue', 'red', 'green', 'purple'])
# for i, color in zip(range(n_classes), colors):
#     plt.plot(fpr[i], tpr[i], color=color, lw=2,
#              label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

# plt.plot([0, 1], [0, 1], 'k--', lw=2)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic for Decision Tree - Multi-Class')
# plt.legend(loc="lower right")
# plt.show()

# # Evaluation Metrics on test
# print("Accuracy on Test Set: {:.2f}".format(accuracy_score(y_test, y_pred_test)))
# print("Classification Report:")
# print(classification_report(y_test, y_pred_test))

# # Confusion matrix for the test set
# conf_matrix_test = confusion_matrix(y_test, y_pred_test)
# print("Confusion Matrix (Test Set):\n", conf_matrix_test)

# %%
# from sklearn.neural_network import MLPClassifier
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import accuracy_score, roc_curve, auc, classification_report, confusion_matrix
# from sklearn.preprocessing import label_binarize
# import numpy as np
# import matplotlib.pyplot as plt
# from itertools import cycle
# from scipy import interp

# # Assuming you have the data ready:
# # X_train_selected, y_train, X_test_selected, y_test

# # Binarize the labels for ROC curve plotting
# y_train_binarized = label_binarize(y_train, classes=np.unique(y_train))
# y_test_binarized = label_binarize(y_test, classes=np.unique(y_train))

# # Parameter grid for MLP
# param_grid_mlp = {
#     'hidden_layer_sizes': [(50,), (100,), (150,)],  # Size of hidden layers (number of neurons)
#     'activation': ['relu', 'tanh'],  # Activation function for the hidden layers
#     'solver': ['adam', 'sgd'],  # Optimization algorithm to use
#     'learning_rate': ['constant', 'adaptive'],  # Learning rate schedule for weight updates
#     'max_iter': [200, 300, 500],  # Maximum number of iterations
#     'alpha': [0.0001, 0.001, 0.01],  # L2 penalty (regularization term)
#     'batch_size': ['auto', 32, 64]  # Size of minibatches for stochastic optimizers
# }

# # Initialize the MLP classifier
# mlp = MLPClassifier(random_state=42)

# # Configure GridSearchCV
# grid_search_mlp = GridSearchCV(estimator=mlp, param_grid=param_grid_mlp, cv=5, scoring='accuracy', verbose=1)

# # Execute the grid search
# grid_search_mlp.fit(X_train_selected, y_train.values.ravel())

# # Print the best parameters and best score
# print("Best parameters:", grid_search_mlp.best_params_)
# print("Best cross-validation score: {:.2f}".format(grid_search_mlp.best_score_))

# # Use the best model found by GridSearchCV
# best_mlp = grid_search_mlp.best_estimator_

# # Predictions and probabilities on the test set
# y_pred_test = best_mlp.predict(X_test_selected)
# y_pred_proba_test = best_mlp.predict_proba(X_test_selected)

# # Calculate ROC AUC for each class
# n_classes = y_train_binarized.shape[1]
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# for i in range(n_classes):
#     fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_proba_test[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])

# # Micro-average ROC curve and ROC area
# fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), y_pred_proba_test.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# # Plot ROC curves for each class
# plt.figure(figsize=(8, 6))
# colors = cycle(['blue', 'red', 'green', 'purple'])
# for i, color in zip(range(n_classes), colors):
#     plt.plot(fpr[i], tpr[i], color=color, lw=2,
#              label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

# plt.plot([0, 1], [0, 1], 'k--', lw=2)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic for MLP - Multi-Class')
# plt.legend(loc="lower right")
# plt.show()

# # Evaluation Metrics on test
# print("Accuracy on Test Set: {:.2f}".format(accuracy_score(y_test, y_pred_test)))
# print("Classification Report:")
# print(classification_report(y_test, y_pred_test))

# # Confusion matrix for the test set
# conf_matrix_test = confusion_matrix(y_test, y_pred_test)
# print("Confusion Matrix (Test Set):\n", conf_matrix_test)

# # %%
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import accuracy_score, roc_curve, auc, classification_report, confusion_matrix
# from sklearn.preprocessing import label_binarize
# import numpy as np
# import matplotlib.pyplot as plt
# from itertools import cycle
# from scipy import interp

# # Assuming you have the data ready:
# # X_train_selected, y_train, X_test_selected, y_test

# # Binarize the labels for ROC curve plotting
# y_train_binarized = label_binarize(y_train, classes=np.unique(y_train))
# y_test_binarized = label_binarize(y_test, classes=np.unique(y_train))

# # Parameter grid for KNN
# param_grid_knn = {
#     'n_neighbors': [3, 5, 7, 10],  # Number of neighbors to consider
#     'weights': ['uniform', 'distance'],  # Weight function used in prediction
#     'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # Algorithm used to compute the nearest neighbors
#     'leaf_size': [10, 20, 30],  # Leaf size passed to the tree algorithms
#     'p': [1, 2],  # Power parameter for the Minkowski distance metric (1 = Manhattan, 2 = Euclidean)
#     'metric': ['minkowski', 'euclidean', 'manhattan'],  # Distance metric to use
# }

# # Initialize the KNN classifier
# knn = KNeighborsClassifier()

# # Configure GridSearchCV
# grid_search_knn = GridSearchCV(estimator=knn, param_grid=param_grid_knn, cv=5, scoring='accuracy', verbose=1)

# # Execute the grid search
# grid_search_knn.fit(X_train_selected, y_train.values.ravel())

# # Print the best parameters and best score
# print("Best parameters:", grid_search_knn.best_params_)
# print("Best cross-validation score: {:.2f}".format(grid_search_knn.best_score_))

# # Use the best model found by GridSearchCV
# best_knn = grid_search_knn.best_estimator_

# # Predictions and probabilities on the test set
# y_pred_test = best_knn.predict(X_test_selected)
# y_pred_proba_test = best_knn.predict_proba(X_test_selected)

# # Calculate ROC AUC for each class
# n_classes = y_train_binarized.shape[1]
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# for i in range(n_classes):
#     fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_proba_test[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])

# # Micro-average ROC curve and ROC area
# fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), y_pred_proba_test.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# # Plot ROC curves for each class
# plt.figure(figsize=(8, 6))
# colors = cycle(['blue', 'red', 'green', 'purple'])
# for i, color in zip(range(n_classes), colors):
#     plt.plot(fpr[i], tpr[i], color=color, lw=2,
#              label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

# plt.plot([0, 1], [0, 1], 'k--', lw=2)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic for KNN - Multi-Class')
# plt.legend(loc="lower right")
# plt.show()

# # Evaluation Metrics on test
# print("Accuracy on Test Set: {:.2f}".format(accuracy_score(y_test, y_pred_test)))
# print("Classification Report:")
# print(classification_report(y_test, y_pred_test))

# # Confusion matrix for the test set
# conf_matrix_test = confusion_matrix(y_test, y_pred_test)
# print("Confusion Matrix (Test Set):\n", conf_matrix_test)

# # %%
# # Binary
# from xgboost import XGBClassifier
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import accuracy_score, roc_curve, auc, classification_report, confusion_matrix
# import numpy as np
# import matplotlib.pyplot as plt

# # Assuming you have the data ready:
# # X_train_selected, y_train, X_test_selected, y_test

# # Hyperparameter grid (simplified)
# param_grid_xgb = {
#     'n_estimators': [300],
#     'max_depth': [5],
#     'learning_rate': [0.001],
#     'min_child_weight': [5],
#     'gamma': [0.3],
#     'subsample': [1.0],
#     'colsample_bytree': [1.0],
#     'reg_alpha': [1],
#     'reg_lambda': [1]
# }

# # Initialize the XGBoost classifier
# xgb = XGBClassifier(random_state=42)

# # Configure GridSearchCV
# grid_search_xgb = GridSearchCV(estimator=xgb, param_grid=param_grid_xgb, cv=5, scoring='accuracy', verbose=1)

# # Execute the grid search
# grid_search_xgb.fit(X_train_selected, y_train.values.ravel())

# # Print the best parameters and best score
# print("Best parameters:", grid_search_xgb.best_params_)
# print("Best cross-validation score: {:.2f}".format(grid_search_xgb.best_score_))

# # Use the best model found by GridSearchCV
# best_xgb = grid_search_xgb.best_estimator_

# # Predictions and probabilities on the test set
# y_pred_test = best_xgb.predict(X_test_selected)
# y_pred_proba_test = best_xgb.predict_proba(X_test_selected)[:, 1]

# # Calculate ROC AUC
# fpr, tpr, _ = roc_curve(y_test, y_pred_proba_test)
# roc_auc = auc(fpr, tpr)

# # Plot ROC curve
# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
# plt.plot([0, 1], [0, 1], 'k--', lw=2)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic for XGBoost - Binary Classification')
# plt.legend(loc="lower right")
# plt.show()

# # Evaluation Metrics on test
# print("Accuracy on Test Set: {:.2f}".format(accuracy_score(y_test, y_pred_test)))
# print("Classification Report:")
# print(classification_report(y_test, y_pred_test))

# # Confusion matrix for the test set
# conf_matrix_test = confusion_matrix(y_test, y_pred_test)
# print("Confusion Matrix (Test Set):\n", conf_matrix_test)

# %%
# # Binary
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import accuracy_score, roc_curve, auc, classification_report, confusion_matrix
# import matplotlib.pyplot as plt

# # Assuming you have the data ready:
# # X_train_selected, y_train, X_test_selected, y_test

# # Parameter grid for Random Forest
# param_grid_rf = {
#     'n_estimators': [100],  # Number of trees in the forest
#     'max_depth': [5],  # Maximum depth of each tree
#     'min_samples_split': [2],  # Minimum samples required to split a node
#     'min_samples_leaf': [1],  # Minimum samples required at a leaf node
#     'max_features': ['auto'],  # Number of features to consider at each split
#     'bootstrap': [True]  # Whether bootstrap samples are used when building trees
# }

# # Initialize the Random Forest classifier
# rf = RandomForestClassifier(random_state=42)

# # Configure GridSearchCV
# grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, scoring='accuracy', verbose=1)

# # Execute the grid search
# grid_search_rf.fit(X_train_selected, y_train.values.ravel())

# # Print the best parameters and best score
# print("Best parameters:", grid_search_rf.best_params_)
# print("Best cross-validation score: {:.2f}".format(grid_search_rf.best_score_))

# # Use the best model found by GridSearchCV
# best_rf = grid_search_rf.best_estimator_

# # Predictions and probabilities on the test set
# y_pred_test = best_rf.predict(X_test_selected)
# y_pred_proba_test = best_rf.predict_proba(X_test_selected)[:, 1]

# # Calculate ROC AUC
# fpr, tpr, _ = roc_curve(y_test, y_pred_proba_test)
# roc_auc = auc(fpr, tpr)

# # Plot ROC curve
# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
# plt.plot([0, 1], [0, 1], 'k--', lw=2)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic for Random Forest - Binary Classification')
# plt.legend(loc="lower right")
# plt.show()

# # Evaluation Metrics on test
# print("Accuracy on Test Set: {:.2f}".format(accuracy_score(y_test, y_pred_test)))
# print("Classification Report:")
# print(classification_report(y_test, y_pred_test))

# # Confusion matrix for the test set
# conf_matrix_test = confusion_matrix(y_test, y_pred_test)
# print("Confusion Matrix (Test Set):\n", conf_matrix_test)

# %%
