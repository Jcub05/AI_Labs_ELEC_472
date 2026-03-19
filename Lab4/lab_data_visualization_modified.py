



import pandas as pd 
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from matplotlib import pyplot as plt
import numpy as np


if __name__=='__main__':

    raw_data=pd.read_csv('raw_accelerometer_dataset.csv', delimiter=',')

    # Display column names
    print(raw_data.columns)

    # Fetch unique activity classes
    classes = raw_data['Class'].unique()
    print(classes)

    # Visualize the x-axis of the accelerometer data
    fig, axes = plt.subplots(len(classes), 1, figsize=(4, 2 * len(classes)))
    fig.suptitle('Accelerometer X-axis', fontsize=12)

    for i, task in enumerate(classes):
        # Fetch rows particular class
        selected_task = raw_data[raw_data['Class'] == task]
        
        # Select the axis
        signal = selected_task.iloc[:, 1].values 
        
        # Plot a small segment
        axes[i].plot(signal[:5000])
        axes[i].set_ylabel(task, fontsize=12)
        
    axes[-1].set_xlabel('time', fontsize=12)
    plt.tight_layout()
    plt.savefig('figure_X.pdf')
    plt.close()

    # Y axis same thing
    fig, axes = plt.subplots(len(classes), 1, figsize=(4, 2 * len(classes)))
    fig.suptitle('Accelerometer Y-axis', fontsize=12)

    for i, task in enumerate(classes):

        selected_task = raw_data[raw_data['Class'] == task]
        

        signal = selected_task.iloc[:, 2].values # Y- axis
        

        axes[i].plot(signal[:5000])
        axes[i].set_ylabel(task, fontsize=12)
        
    axes[-1].set_xlabel('time', fontsize=12)
    plt.tight_layout()
    plt.savefig('figure_Y.pdf')
    plt.close()

    # Z axis same thing
    fig, axes = plt.subplots(len(classes), 1, figsize=(4, 2 * len(classes)))
    fig.suptitle('Accelerometer Z-axis', fontsize=12)

    for i, task in enumerate(classes):
        selected_task = raw_data[raw_data['Class'] == task]
        

        signal = selected_task.iloc[:, 3].values  # Z-axis
        
        axes[i].plot(signal[:5000])
        axes[i].set_ylabel(task, fontsize=12)
        
    axes[-1].set_xlabel('time', fontsize=12)
    plt.tight_layout()
    plt.savefig('figure_Z.pdf')
    plt.close()
