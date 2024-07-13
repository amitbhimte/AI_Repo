from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Example DataFrame with numerical variables
data_multi = {
    'A': [1, 2, None, 4],
    'B': [5, None, 7, 8],
    'C': [None, 10, 11, 12]
}
df_multi = pd.DataFrame(data_multi)

# Multiple Imputation (using IterativeImputer from sklearn)
imputer = IterativeImputer()
df_imputed_multiple = imputer.fit_transform(df_multi)
df_imputed_multiple = pd.DataFrame(df_imputed_multiple, columns=df_multi.columns)
print("\nMultiple Imputation Method:")
print(df_imputed_multiple)
