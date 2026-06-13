import pandas as pd

# Example dataset information
dataset_info = {
    "Dataset": ["COCO", "Oxford-102 Flowers"],
    "Classes": [80, 102],
    "Images": [118000, 8189]
}

df = pd.DataFrame(dataset_info)

print("Dataset Analysis")
print(df)

print("\nBasic Statistics")
print(df.describe())
