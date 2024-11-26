import pandas as pd

# Load the raw dataset
dataset = pd.read_csv('doggie_dataset2.csv')

# Example preprocessing: Clean data, remove duplicates, etc.
processed_dataset = dataset.drop_duplicates()

# Save the processed dataset
processed_dataset.to_csv('processed_dataset.csv', index=False)

print("Data preparation completed successfully.")
