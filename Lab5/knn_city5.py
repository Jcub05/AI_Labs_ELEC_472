import pandas as pd
import numpy as np
import random
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

random.seed(42)

# Import the pre-extracted features
dataset = pd.read_csv('full_dataset_features.csv')

# Visualize the dataset (display first 5 rows)
print(dataset.head(5))

# Find how many classes the dataset has
unique_classes = dataset['activity'].unique()
print("Unique classes:", unique_classes)

# Find how many subjects there are
unique_subjects = dataset['subject_id'].unique()
print("Unique subjects:", unique_subjects)

# Get the total number of records
total_records = dataset.shape[0]
print("Total records:", total_records)

# Set number of folds for training-testing
total_fold = 10

# Initialize variables to store final performance
final_accuracy = 0
final_precision = 0
final_recall = 0
final_f1_score = 0
total_time = 0

indices = np.arange(0, len(dataset))
random.shuffle(indices)

for fold in range(0, total_fold):
    # select the test indices
    split_size = len(indices)//total_fold
    test_mask = np.zeros(len(indices), dtype=bool)
    test_mask[split_size*fold:split_size*(fold+1)] = True
    test_indices = indices[test_mask]
    # select train indices, which are total indices - test indices
    train_indices = indices[~test_mask]

    # select x_train, y_train and x_test, y_test from the dataset
    x_train = dataset.iloc[train_indices, :18]
    y_train = dataset.iloc[train_indices]['activity']
    x_test = dataset.iloc[test_indices, :18]
    y_test = dataset.iloc[test_indices]['activity']
    
    # TODO: build your model here
    model = KNeighborsClassifier(n_neighbors=5, metric='manhattan')

    # Start timing the training block
    start_time=time.time()
    # TODO: train your model using x_train and y_train
    model.fit(x_train, y_train)
    
    # end timing the training block and display elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Training took {elapsed_time:.2f} seconds\n', );

    #  start of evaluation 

    # initialization
    tr_acc = 0
    te_acc = 0
    te_precision = 0
    te_recall    = 0
    te_f1_score  = 0
    
    # TODO: calculate and display the training and testing accuracy for the fold
    y_pred = model.predict(x_test)
    y_pred_train = model.predict(x_train)
    tr_acc = accuracy_score(y_train, y_pred_train)
    te_acc = accuracy_score(y_test, y_pred)

    print(f'Classifier output fold {fold} - test acc: {te_acc*100:.2f}% - train acc: {tr_acc*100:.2f}%')
    
    # TODO: Calculate average precision, recall, and f1 score for the test set.
    te_precision = precision_score(y_test, y_pred, average='weighted')
    te_recall = recall_score(y_test, y_pred, average='weighted')
    te_f1_score = f1_score(y_test, y_pred, average='weighted')

    print(f'Classifier output fold {fold} - test precision: {te_precision*100:.2f}% - test recall: {te_recall*100:.2f}% - test f1-score: {te_f1_score*100:.2f}%')
    

    # TODO: save results from each fold to calculate final results averaged over all folds in the end
    final_accuracy  += te_acc
    final_precision += te_precision
    final_recall    += te_recall
    final_f1_score  += te_f1_score
    total_time += elapsed_time
    
    
# Calculate final average performance after 10-fold cross validation
final_accuracy  /= total_fold
final_precision /= total_fold
final_recall    /= total_fold
final_f1_score  /= total_fold

# Final output of cross validation results
print(f'\nClassifier final output Test: Accuracy:  {final_accuracy*100:.2f}% Precision: {final_precision*100:.2f}% Recall:    {final_recall*100:.2f}% F1-Score:  {final_f1_score*100:.2f}% Total Training Time: {total_time:.2f} seconds')
