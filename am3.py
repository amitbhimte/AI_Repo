# Example DataFrame with categorical variable
data_cat = {
    'A': ['red', 'blue', 'blue', None, 'red'],
    'B': ['small', 'large', None, 'small', 'large']
}
df_cat = pd.DataFrame(data_cat)

# Frequent Category Imputation
most_frequent = df_cat.mode().iloc[0]
df_frequent = df_cat.fillna(most_frequent)
print("\nFrequent Category Imputation:")
print(df_frequent)
