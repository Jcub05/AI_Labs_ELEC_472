
import pandas as pd
import numpy as np
from scipy.stats import skew, entropy

def extract_features(filename="raw_accelerometer_dataset.csv", 
                     segment_size = 250):

    # Read raw data
    raw_data = pd.read_csv(filename)

    # Initialize parameters
    feature_set = pd.DataFrame()
    event = 0
    classes = raw_data['Class'].unique()

    for task, class_name in enumerate(classes):
        # Filter data for current class/activity
        class_data = raw_data[raw_data['Class'] == class_name].iloc[:, 1:4]
        
        # Create group IDs
        num_samples = len(class_data)
        group_id = np.repeat(np.arange(event, event + np.ceil(num_samples/segment_size)), 
                            segment_size)[:num_samples]
        event += len(np.unique(group_id))
        
        class_data['group_id'] = group_id
        
        # Initialize feature containers
        features = []
        
        # Calculate features per segment
        for g_id, group in class_data.groupby('group_id'):
            data = group.iloc[:, :3].values        
            feature_row = [
                # Maximum features
                *np.max(data, axis=0),  
                # Minimum features
                *np.min(data, axis=0),  
                # Mean features
                *np.mean(data, axis=0),  
                # Standard Deviation features
                *np.std(data, axis=0),  
                # Skewness features
                *[skew(data[:, i]) for i in range(3)],  
                # Entropy features
                *[entropy(np.histogram(data[:, i], bins=10)[0] + 1e-10) for i in range(3)]  
            ]
            
            features.append(feature_row)
        
        columns = [
            'max_x', 'max_y', 'max_z',
            'min_x', 'min_y', 'min_z',
            'mean_x', 'mean_y', 'mean_z',
            'std_x', 'std_y', 'std_z',
            'skew_x', 'skew_y', 'skew_z',
            'entropy_x', 'entropy_y', 'entropy_z'
        ]
        
        class_features = pd.DataFrame(features, columns=columns)
        class_features['activity'] = task
        
        feature_set = pd.concat([feature_set, class_features], ignore_index=True)

    # Save results
    feature_set.to_csv(f'features_{segment_size}.csv', index=False)
    print(f'Window size: {segment_size}')
    print(f'Feature set shape: {feature_set.shape}')
    print(f'Columns: {list(feature_set.columns)}')
    return feature_set



if __name__=='__main__':

    # Extract features with window size 250
    print('Extracting features with window size 250...')
    features_250 = extract_features(filename="raw_accelerometer_dataset.csv", 
                                    segment_size=250)
    print()
    
    # Extract features with window size 500
    print('Extracting features with window size 500...')
    features_500 = extract_features(filename="raw_accelerometer_dataset.csv", 
                                    segment_size=500)