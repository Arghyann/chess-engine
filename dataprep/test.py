import pandas as pd

# Load CSV
df = pd.read_csv("training_data.csv")

# Remove last row
df = df.iloc[:-1]

# Save it back
df.to_csv("training_data_lastnot.csv", index=False)
