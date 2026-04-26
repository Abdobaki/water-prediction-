"""
Classification of Water Samples (Aluminium)
A machine learning project using Python and scikit-learn.

This script classifies water samples based on their quality using 
measurements like Aluminium Value and Depth.
"""

# ============================================================
# 1. Check library versions
# ============================================================
import sys
print('Python: {}'.format(sys.version))

import scipy
print('scipy: {}'.format(scipy.__version__))

import numpy
print('numpy: {}'.format(numpy.__version__))

# Use non-interactive backend (works without a display/GUI)
import matplotlib
matplotlib.use('Agg')
print('matplotlib: {}'.format(matplotlib.__version__))

import pandas
print('pandas: {}'.format(pandas.__version__))

import sklearn
print('sklearn: {}'.format(sklearn.__version__))


# ============================================================
# 2. Import libraries
# ============================================================
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, BaggingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron

# ============================================================
# 3. Load dataset
# ============================================================
# Load Aluminium dataset with semicolon separator
dataset = pandas.read_csv("Aluminium.csv", sep=';', encoding='latin-1')

# Data Cleaning & Preprocessing
# 1. Sample the dataset for performance (Aluminium has >130k rows)
dataset = dataset.sample(n=5000, random_state=1)

# 2. Drop irrelevant columns
dataset = dataset[['Value', 'Depth', 'Data.Quality']]

# 3. Handle missing values
dataset = dataset.dropna()

# 4. Ensure numeric types for features
dataset['Value'] = pandas.to_numeric(dataset['Value'], errors='coerce')
dataset['Depth'] = pandas.to_numeric(dataset['Depth'], errors='coerce')
dataset = dataset.dropna() # Drop rows where conversion failed


# ============================================================
# 4. Summarize the dataset
# ============================================================

# 4.1 Dimensions of Dataset (rows, columns)
print("\n--- Dataset Shape ---")
print(dataset.shape)

# 4.2 Peek at the Data
print("\n--- First 21 Rows ---")
print(dataset.head(21))

# 4.3 Statistical Summary
print("\n--- Statistical Summary ---")
print(dataset.describe())

# 4.4 Class Distribution
print("\n--- Class Distribution ---")
print(dataset.groupby('Data.Quality').size())


# ============================================================
# 5. Data Visualization (saved as PNG images)
# ============================================================

# 5.1 Univariate Plots - Box and Whisker
dataset.plot(kind='box', subplots=True, layout=(1, 2), sharex=False, sharey=False)
pyplot.suptitle("Box and Whisker Plots")
pyplot.tight_layout()
pyplot.savefig("plot_boxplots.png", dpi=150)
pyplot.close()
print("\nSaved: plot_boxplots.png")

# 5.2 Univariate Plots - Histograms
dataset.hist()
pyplot.suptitle("Histograms")
pyplot.tight_layout()
pyplot.savefig("plot_histograms.png", dpi=150)
pyplot.close()
print("Saved: plot_histograms.png")

# 5.3 Multivariate Plots - Scatter Plot Matrix
scatter_matrix(dataset)
pyplot.suptitle("Scatter Plot Matrix")
pyplot.tight_layout()
pyplot.savefig("plot_scatter_matrix.png", dpi=150)
pyplot.close()
print("Saved: plot_scatter_matrix.png")


# ============================================================
# 6. Evaluate Algorithms
# ============================================================

# 6.1 Split-out validation dataset: 80% train, 20% test
X = dataset[['Value', 'Depth']].values
Y = dataset['Data.Quality'].values
X_train, X_validation, Y_train, Y_validation = train_test_split(
    X, Y, test_size=0.2, random_state=1
)

# 6.2 Build Models - Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(C=1, solver='lbfgs', max_iter=4000, random_state=1)))
models.append(('LDA', LinearDiscriminantAnalysis( tol=0.0001 )))
models.append(('KNN', KNeighborsClassifier(n_neighbors=3,weights='distance')))
models.append(('CART', DecisionTreeClassifier(max_depth=10, min_samples_split=5, min_samples_leaf=2)))
models.append(('NB', GaussianNB(var_smoothing=1)))
models.append(('SVM', SVC(gamma='auto',C=9, kernel='rbf',random_state=1)))
models.append(('RF', RandomForestClassifier(n_estimators=100,max_features='sqrt',)))
models.append(('ADA', AdaBoostClassifier(
    n_estimators=100,
    learning_rate=1.0,
    random_state=1
)))

models.append(('GBM', GradientBoostingClassifier(subsample=0.8)))
#  models.append(('GPC', GaussianProcessClassifier(   kernel=1.0 ,  n_restarts_optimizer=5,    max_iter_predict=100,  random_state=1)))
models.append(('MLP', MLPClassifier(
    alpha=0.01,
    max_iter=300,
    random_state=1
)))
models.append(('ET', ExtraTreesClassifier(n_estimators=100,criterion='gini',verbose=0, warm_start=False,random_state=1)))
models.append(('BAG', BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=5),n_estimators=100, random_state=1)))
models.append(('PERC', Perceptron(alpha=0.0001, tol=0.5, random_state=1)))
models.append(('SGD', SGDClassifier(
    loss='log_loss',             
    penalty='elasticnet', 
    max_iter=1000, tol=0.4, random_state=1)))


# Evaluate each model using 10-fold Stratified Cross Validation
print("\n--- Model Comparison ---")
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# 6.3 Compare Algorithms visually
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.ylabel('Accuracy')
pyplot.savefig("plot_algorithm_comparison.png", dpi=150)
pyplot.close()
print("\nSaved: plot_algorithm_comparison.png")


# ============================================================
# 7. Make Predictions
# ============================================================

# Using SVM (the best performing model)
print("\n--- SVM Predictions on Validation Set ---")
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)


# ============================================================
# 8. Evaluate Predictions
# ============================================================

# Accuracy
print("\nAccuracy:", accuracy_score(Y_validation, predictions))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(Y_validation, predictions))

# Detailed Confusion Matrix with labels (dynamic based on classes)
labels = sorted(dataset['Data.Quality'].unique())
cmtx = pandas.DataFrame(
    confusion_matrix(Y_validation, predictions, labels=labels),
    index=['true:' + str(l) for l in labels],
    columns=['pred:' + str(l) for l in labels]
)
print("\nDetailed Confusion Matrix:")
print(cmtx)

# Classification Report
print("\nClassification Report:")
print(classification_report(Y_validation, predictions))
