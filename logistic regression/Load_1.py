import pandas as pd

# Load dataset
data = pd.read_csv("student_data.csv")

# View dataset
print(data)

# View first 5 records
print("\nFirst 5 records:")
print(data.head())

# View dataset information
print("\nDataset information:")
print(data.info())

# View dataset dimensions
print("\nDataset shape:")
print(data.shape)