import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.cross_decomposition import CCA
independent_vars = ['Soil Nutrient', 'Irrigation Frequency', 'Pesticide Application', 'Sunlight Exposure']
dependent_vars = ['Crop Yield', 'Plant Height', 'Fruit Weight']
# Read data from CSV
df = pd.read_csv('data.csv')
# Separate independent and dependent variables
independent_data = df[independent_vars]
dependent_data = df[dependent_vars]
# Perform canonical correlation analysis
cca = CCA(n_components=1)
cca.fit(independent_data, dependent_data)
canonical_corr = cca.transform(independent_data)
# Visualize the results
plt.figure(figsize=(8, 6))
plt.scatter(canonical_corr, dependent_data['Crop Yield'], label='Crop Yield')
plt.scatter(canonical_corr, dependent_data['Plant Height'], label='Plant Height')
plt.scatter(canonical_corr, dependent_data['Fruit Weight'], label='Fruit Weight')
plt.xlabel('Canonical Correlation')
plt.ylabel('Dependent Variables')
plt.legend()
plt.show()
