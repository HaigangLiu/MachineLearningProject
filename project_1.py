from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn
from sklearn import neighbors
from sklearn.linear_model import Ridge, LassoCV
from sklearn.feature_selection import SelectFromModel

#let's do 10 fold cv
CROSS_VALIDATION = 10; THRESHOLD_ELIMINATE = 0.1
os.chdir("/Users/haigangliu/Desktop/machine_learning")

with open("training.txt") as f:
    training_set_full_size = pd.read_table(f, sep = "\t", header = 0)
with open("predicting.txt") as f:
    predicting_set = pd.read_table(f, sep = "\t", header = 0)

#randomize the order of records;
def shuffler(traing_set_to_shuffle):
    df = pd.DataFrame(traing_set_to_shuffle)
    df_shuffled =  df.iloc[np.random.permutation(len(df))]
    return df_shuffled

#set aside some data to avoid bleeding.
def splitter(training_set, ratio = 0.8):
    df_shuffled = shuffler(training_set)
    t_size = int(0.9*df_shuffled.shape[0])
    training_set = df_shuffled.iloc[:t_size,:]
    private_set = df_shuffled.iloc[t_size+1:, :]
    return (training_set, private_set)

training_set, private_set = splitter(training_set_full_size)

#split the labels and indepedent variables
def x_and_y_splitter(dataset):
    df = pd.DataFrame(dataset)
    y = df.iloc[:,-1]
    x = df.iloc[:,0:(df.shape[1]-1)]
    return (x, y)

def variable_selector(dataset, threshold = THRESHOLD_ELIMINATE):
    data, labels =x_and_y_splitter(dataset)
    lasso_object = LassoCV(max_iter=10000)
    model_selector = SelectFromModel(lasso_object, threshold)
    model_selector.fit(data,labels)
    return (model_selector.transform(data), labels)

def converter_machine(dataset, threshold = THRESHOLD_ELIMINATE):
    data, labels =x_and_y_splitter(dataset)
    lasso_object = LassoCV(max_iter=10000)
    model_selector = SelectFromModel(lasso_object, threshold)
    model_selector.fit(data,labels)
    return model_selector


transformed_data, labels= variable_selector(training_set)
frame = [pd.DataFrame(transformed_data), pd.DataFrame(np.array(labels))]
reduced_data_set = pd.concat(frame, axis = 1)

#the knn regressor
def knn(dataset, k):
    data, labels = x_and_y_splitter(dataset)
    knn_object = neighbors.KNeighborsRegressor(k)
    return knn_object.fit(data,labels)

## to do k fold cross-validation, we need to cut data in k parts
def data_set_chopper(dataset, cross_validation_counts = CROSS_VALIDATION):
    size_of_section = len(dataset)//cross_validation_counts
    i = 0;
    data_sections = []
    for i in range(cross_validation_counts):
        data_sections.append(dataset.iloc[i*size_of_section:(i+1)*size_of_section,:])
    return data_sections

#squared loss function is used here to evaluate models
def evaluation_handler(true_values, predicted_values):
    try:
        true_values_np = np.array(true_values)
        predicted_values_np = np.array(predicted_values)
        loss_function = np.sum(true_values_np - predicted_values_np)**2
        return loss_function
    except TypeError:
        print "check if there is string in your list"

# a function of k, in order to choose the best k
def loss_function_of_knn(dataset, k):
    sections = data_set_chopper(dataset,cross_validation_counts = CROSS_VALIDATION)
    error_list = []
    # used a cross_validation algorithm
    for i in range(CROSS_VALIDATION):
        #four parts to train and one part to test
        list_to_train = sections[:i] + sections[i+1:]
        training_set_of_cv = pd.concat(list_to_train)
        y_hat = knn(training_set_of_cv, k).predict(x_and_y_splitter(sections[i])[0])
        error = evaluation_handler(x_and_y_splitter(sections[i])[1],y_hat)
        error_list.append(error)
    return np.mean(error_list)

#plot a graph to find the best k
x_lim = np.arange(1, 10)
y_lim = []
for k in x_lim:
    y_lim.append(loss_function_of_knn(reduced_data_set,k))

print "the sum of squared residuals are %d" %(min(y_lim))
plt.plot(x_lim,y_lim)

def ridge_regressor(dataset):
    x_data, y_data = x_and_y_splitter(dataset)
    clf = Ridge(alpha = 1.0)
    clf.fit(x_data, y_data)
    Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
      normalize=False, random_state=None, solver='auto', tol=0.0001)
    return clf

def loss_function_of_ridge(dataset, CROSS_VALIDATION = 10):
    sections = data_set_chopper(dataset,cross_validation_counts = CROSS_VALIDATION)
    error_list = []
    # used a cross_validation algorithm
    for i in range(CROSS_VALIDATION):
        #four parts to train and one part to test
        list_to_train = sections[:i] + sections[i+1:]
        training_set_of_cv = pd.concat(list_to_train)
        y_hat = ridge_regressor(training_set_of_cv).predict(x_and_y_splitter(sections[i])[0])
        error = evaluation_handler(x_and_y_splitter(sections[i])[1],y_hat)
        error_list.append(error)
    return np.mean(error_list)

loss_of_ridge = loss_function_of_ridge(reduced_data_set)

##comparison use the private one
converter = converter_machine(training_set)

y_hat_knn = knn(reduced_data_set, k = 3).predict(converter.transform(x_and_y_splitter(private_set)[0]))
error_knn = evaluation_handler(x_and_y_splitter(private_set)[1],y_hat_knn)

y_hat_ridge = ridge_regressor(reduced_data_set).predict(converter.transform(x_and_y_splitter(private_set)[0]))
error_ridge = evaluation_handler(x_and_y_splitter(private_set)[1],y_hat_ridge)

print "the error of knn is %d" %error_knn
print "the error of ridge is %d" %error_ridge
plt.show()
