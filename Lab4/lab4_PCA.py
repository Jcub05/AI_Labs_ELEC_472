# If you do not have pandas installed, please install it using `pip install pandas`
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
dataset_full=pd.read_csv('dataset.csv', delimiter=';')
# We want to take a subset of the dataset (the entire dataset is too large).
x = np.array(dataset_full.values[155750:156250, 6:18], dtype=float)
x_names=list(dataset_full.columns)[6:18]
print('feature shape: ', x.shape)
print('feature names: ', x_names)
# We can plot the 12 dimensions.
plt.figure()
t=[i for i in range(x.shape[0])]
for k in range(len(x_names)):
    plt.plot(t, x[:, k], label=x_names[k])
plt.legend(loc='upper right')
plt.title('Feature visualization')
plt.savefig('fig1.pdf')

# Calculate the mean and standard deviation:
m_x = np.mean(x, axis=0)
s_x = np.std(x, axis=0)
# Perform normalization
x_bar = (x - m_x) / s_x
plt.figure()
t=[i for i in range(x_bar.shape[0])]
for k in range(len(x_names)):
    plt.plot(t, x_bar[:, k], label=x_names[k])
plt.legend(loc='upper right')
plt.title('Normalized feature visualization')
plt.savefig('fig2.pdf')

def pca(x):
    # Step 1: Mean-center the data
    mean_x = np.mean(x, axis=0)
    x_centered = x - mean_x
    
    # Step 2: Compute the cov matrix
    cov_matrix = np.cov(x_centered.T)
    
    # Step 3: Perform eigen decomposition (use np.linalg.eigh)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Step 4: Sort the eigenvalues AND eigenvectors in decreasing order
    # (Hint: use np.argsort on eigenvalues)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Step 5: Select the sorted eigenvectors as principal components (pcs)
    pcs = eigenvectors
    
    # Step 6: Compute the variance explained by each principal component (explained)
    explained = eigenvalues / np.sum(eigenvalues)
    
    # Step 7: Project the data onto principal component axes to obtain scores
    scores = np.dot(x_centered, pcs)
    
    return pcs, scores, explained

# Plot PCA results
pcs, scores, explained = pca(x_bar)

# Figure 3: Variance explained by each principal component
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.bar(range(1, len(explained) + 1), explained)
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.title('Variance Explained by Each Principal Component')
plt.xticks(range(1, len(explained) + 1))

# Figure 3b: Cumulative variance explained
plt.subplot(1, 2, 2)
cumulative_explained = np.cumsum(explained)
plt.plot(range(1, len(cumulative_explained) + 1), cumulative_explained, marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Variance Explained')
plt.title('Cumulative Variance Explained')
plt.grid(True)
plt.xticks(range(1, len(cumulative_explained) + 1))
plt.tight_layout()
plt.savefig('fig3_variance_explained.pdf')

# Figure 4: Scatter plot of PC1 vs PC2 (first two principal components)
plt.figure(figsize=(10, 8))
t = range(scores.shape[0])
plt.scatter(scores[:, 0], scores[:, 1], c=t, cmap='viridis', alpha=0.6, s=50)
plt.xlabel(f'PC1 ({explained[0]:.2%} variance)')
plt.ylabel(f'PC2 ({explained[1]:.2%} variance)')
plt.title('PCA Transformed Data: PC1 vs PC2')
plt.colorbar(label='Sample Index')
plt.grid(True, alpha=0.3)
plt.savefig('fig4_PCA_scatter.pdf')

# Print variance explained summary
print('\nPCA Analysis Summary:')
print(f'Total features: {x_bar.shape[1]}')
print(f'Variance explained by first 3 PCs: {np.sum(explained[:3]):.2%}')
print(f'Variance explained by first 5 PCs: {np.sum(explained[:5]):.2%}')
print(f'\nVariance explained by each PC:')
for i in range(min(5, len(explained))):
    print(f'  PC{i+1}: {explained[i]:.2%}')

# Min number of PCs to explain 98% var
cumulative_explained = np.cumsum(explained)
min_pcs_98 = np.argmax(cumulative_explained >= 0.98) + 1
print(min_pcs_98)
print(f'{cumulative_explained[min_pcs_98-1]:.4f}')