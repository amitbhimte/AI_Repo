import pandas as pd

# Example DataFrame with missing values
data = {
    'A': [1, 2, None, 4],
    'B': [5, None, 7, 8],
    'C': [None, 10, 11, 12]
}
df = pd.DataFrame(data)

# Complete Case Analysis
df_complete_case = df.dropna()
print("Complete Case Analysis:")
print(df_complete_case)
