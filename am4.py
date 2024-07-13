# Example DataFrame with numerical variable
data_num = {
    'A': [1, 2, None, 4],
    'B': [5, None, 7, 8],
    'C': [None, 10, 11, 12]
}
df_num = pd.DataFrame(data_num)

# Single Imputation (using mean for example)
mean_imputer = df_num.mean()
df_mean_imputed = df_num.fillna(mean_imputer)
print("\nSingle Imputation (Mean):")
print(df_mean_imputed)
