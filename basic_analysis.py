# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()

# Convert to a pandas DataFrame
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['species'] = iris.target

# Add species names for easier understanding
data['species'] = data['species'].map({i: name for i, name in enumerate(iris.target_names)})

# Display the first few rows
print("Dataset Preview:")
print(data.head())

# 1. Data Exploration
print("\nSummary of Dataset:")
print(data.info())

print("\nMissing Values Check:")
print(data.isnull().sum())

print("\nStatistical Summary:")
print(data.describe())

print("\nSpecies Distribution:")
print(data['species'].value_counts())

# 2. Basic Data Analysis
# Example: Mean of each feature by species
mean_features_by_species = data.groupby('species').mean()
print("\nMean Features by Species:")
print(mean_features_by_species)

# 3. Visualizations
# Bar chart: Distribution of species
data['species'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Distribution of Species')
plt.xlabel('Species')
plt.ylabel('Frequency')
plt.show()

# Histogram: Distribution of sepal length
data[iris.feature_names[0]].plot(kind='hist', bins=10, color='orange')
plt.title('Histogram of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.show()

# Scatter plot: Sepal length vs Sepal width
plt.scatter(data[iris.feature_names[0]], data[iris.feature_names[1]], c=data['species'].factorize()[0], cmap='viridis')
plt.title('Sepal Length vs Sepal Width')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.colorbar(ticks=[0, 1, 2], label='Species')
plt.show()

# 4. Pairwise comparison using scatter plots (sepal and petal)
pd.plotting.scatter_matrix(data.iloc[:, :4], figsize=(10, 10), alpha=0.7, diagonal='hist', color='purple')
plt.suptitle('Pairwise Comparison of Iris Features')
plt.show()

# Pairplot (optional, requires seaborn)
import seaborn as sns
sns.pairplot(data, hue='species', diag_kind='hist')
plt.show()

# 4. Findings and Observations
print("\nObservations:")
print("- The dataset has 150 entries with no missing values.")
print("- There are 3 species: Setosa, Versicolor, and Virginica, each with 50 samples.")
print("- Features such as petal length and petal width seem to have distinct patterns for each species.")
print("- Scatter plots indicate clear separations between Setosa and the other species, making it a good candidate for classification.")

