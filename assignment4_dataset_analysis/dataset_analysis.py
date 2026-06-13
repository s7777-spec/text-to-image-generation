import pandas as pd
import matplotlib.pyplot as plt

# Sample dataset statistics
dataset_info = {
    "Dataset": ["COCO", "Oxford-102 Flowers"],
    "Classes": [80, 102],
    "Images": [118000, 8189],
    "Average Caption Length": [11, 5]
}

df = pd.DataFrame(dataset_info)

print("Dataset Analysis")
print(df)

print("\nBasic Statistics")
print(df.describe())

# Visualization
df.plot(
    x="Dataset",
    y="Images",
    kind="bar",
    title="Number of Images in Datasets"
)

plt.show()
