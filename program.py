
# -------------------- IMPORTS --------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from feature_engine.outliers import Winsorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# -------------------- LOAD DATA --------------------
# CHANGE THIS PATH TO YOUR ACTUAL FILE LOCATION
file_path = r"C:/Users/DELL/Desktop/ML&DL/spotify_data clean.xlsx"

spotify_data = pd.read_excel(file_path)

# -------------------- BASIC EDA --------------------
print("\n---- SHAPE ----")
print("Rows, Columns:", spotify_data.shape)

print("\n---- HEAD ----")
print(spotify_data.head())

print("\n---- TAIL ----")
print(spotify_data.tail())

print("\n---- INFO ----")
print(spotify_data.info())

print("\n---- DESCRIBE (NUMERIC) ----")
print(spotify_data.describe())

print("\n---- DESCRIBE (INCLUDE ALL) ----")
print(spotify_data.describe(include='all'))

# Identify numeric columns we’ll use for stats & scaling
numeric_cols = [
    'track_number',
    'track_popularity',
    'artist_popularity',
    'artist_followers',
    'album_total_tracks',
    'track_duration_min'
]

# -------------------- FIRST MOMENT (MEAN, MEDIAN, MODE) --------------------
print("\n================ FIRST MOMENT (MEAN / MEDIAN / MODE) ================")

print("\n---- MEAN ----")
for col in numeric_cols:
    print(f"Mean {col}: {spotify_data[col].mean()}")

print("\n---- MEDIAN ----")
for col in numeric_cols:
    print(f"Median {col}: {spotify_data[col].median()}")

print("\n---- MODE ----")
for col in numeric_cols:
    print(f"Mode {col}:")
    print(spotify_data[col].mode())

# -------------------- SECOND MOMENT (VARIANCE, STD, RANGE) --------------------
print("\n================ SECOND MOMENT (VAR / STD / RANGE) ================")

print("\n---- VARIANCE ----")
for col in numeric_cols:
    print(f"Variance {col}: {spotify_data[col].var()}")

print("\n---- STANDARD DEVIATION ----")
for col in numeric_cols:
    print(f"Std {col}: {spotify_data[col].std()}")

print("\n---- RANGE (MAX - MIN) ----")
for col in numeric_cols:
    _range = spotify_data[col].max() - spotify_data[col].min()
    print(f"Range {col}: { _range }")

# -------------------- THIRD / FOURTH MOMENT (SKEWNESS / KURTOSIS) --------------------
print("\n================ THIRD MOMENT (SKEWNESS) ================")
print(spotify_data[numeric_cols].skew())

print("\n================ FOURTH MOMENT (KURTOSIS) ================")
print(spotify_data[numeric_cols].kurt())

# -------------------- DATA VISUALIZATION --------------------
# UNIVARIATE PLOTS: HISTOGRAMS
print("\nGenerating histograms...")

for col in numeric_cols:
    plt.figure()
    plt.hist(spotify_data[col], bins=30)
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

# BOXPLOTS to check for outliers
print("\nGenerating boxplots...")

for col in numeric_cols:
    plt.figure()
    plt.boxplot(spotify_data[col])
    plt.title(f'Boxplot of {col}')
    plt.ylabel(col)
    plt.show()

# BOXPLOTS with seaborn
for col in numeric_cols:
    plt.figure()
    sns.boxplot(x=spotify_data[col])
    plt.title(f'Seaborn Boxplot of {col}')
    plt.show()

# SCATTER PLOTS: relationships between two variables
print("\nGenerating scatter plots...")

plt.figure(figsize=(8, 6))
plt.scatter(spotify_data['track_popularity'], spotify_data['artist_popularity'])
plt.xlabel('Track Popularity')
plt.ylabel('Artist Popularity')
plt.title('Track Popularity vs Artist Popularity')
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(spotify_data['track_duration_min'], spotify_data['track_popularity'])
plt.xlabel('Track Duration (min)')
plt.ylabel('Track Popularity')
plt.title('Track Duration vs Track Popularity')
plt.show()

# Correlation matrix (numeric only)
print("\n================ CORRELATION MATRIX ================")
print(spotify_data[numeric_cols].corr())

plt.figure(figsize=(8, 6))
sns.heatmap(spotify_data[numeric_cols].corr(), annot=True)
plt.title('Correlation Heatmap (Numeric Features)')
plt.show()

# -------------------- DATA CLEANING --------------------
print("\n================ DUPLICATE CHECK ================")
duplicates = spotify_data.duplicated()
print("Total duplicate rows:", duplicates.sum())

# Remove duplicate rows
spotify_no_dup = spotify_data.drop_duplicates()
print("Shape after removing duplicates:", spotify_no_dup.shape)

# Work on a copy from here
df = spotify_no_dup.copy()

# -------------------- OUTLIER TREATMENT (WINSORIZATION) --------------------
print("\n================ OUTLIER TREATMENT (WINSORIZATION) ================")

# Columns where outliers are most likely problematic
outlier_cols = [
    'track_popularity',
    'artist_popularity',
    'artist_followers',
    'album_total_tracks',
    'track_duration_min'
]

for col in outlier_cols:
    print(f"\nWinsorizing column: {col}")
    winsor_iqr = Winsorizer(
        capping_method='iqr',
        tail='both',
        fold=1.5,
        variables=[col]
    )
    # Fit & transform only that column; result is a DataFrame
    df[col + '_IQR'] = winsor_iqr.fit_transform(df[[col]])

    # Boxplot after winsorization
    plt.figure()
    sns.boxplot(x=df[col + '_IQR'])
    plt.title(f'Boxplot of {col} after Winsorization')
    plt.show()

# -------------------- MISSING VALUES HANDLING --------------------
print("\n================ MISSING VALUES (BEFORE IMPUTATION) ================")
print(df.isna().sum())

# We have missing values only in 'artist_name' and 'artist_genres'
# Use mode (most_frequent) imputation for these categorical columns
cat_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

df[['artist_name', 'artist_genres']] = cat_imputer.fit_transform(
    df[['artist_name', 'artist_genres']]
)

print("\n================ MISSING VALUES (AFTER IMPUTATION) ================")
print(df.isna().sum())

# -------------------- DUMMY VARIABLES (ONE-HOT ENCODING) --------------------
print("\n================ DUMMY VARIABLES ================")

# Drop pure ID columns before encoding (optional but cleaner)
df_for_dummies = df.drop(['track_id', 'album_id'], axis=1)

# Create dummy variables for categorical columns
df_dummies = pd.get_dummies(df_for_dummies, drop_first=True)

print("Shape after creating dummy variables:", df_dummies.shape)

# -------------------- FEATURE SCALING & NORMALIZATION --------------------
print("\n================ FEATURE SCALING ================")

# Standardization (mean=0, std=1)
scaler = StandardScaler()
df_std_array = scaler.fit_transform(df[numeric_cols])

df_std = pd.DataFrame(df_std_array,
                      columns=[col + '_std' for col in numeric_cols])

print("\nStandardized numeric features description:")
print(df_std.describe())

# Normalization [0,1]
minmax = MinMaxScaler()
df_norm_array = minmax.fit_transform(df[numeric_cols])

df_norm = pd.DataFrame(df_norm_array,
                       columns=[col + '_norm' for col in numeric_cols])

print("\nNormalized numeric features description:")
print(df_norm.describe())

# -------------------- FINAL DATASET FOR POWER BI --------------------
# For Power BI, we usually want:
# - Cleaned original columns (df)
# - Winsorized numeric columns (already added as *_IQR)
# You can choose whether to also include scaled versions; here I’ll add them.

df_final = pd.concat(
    [df.reset_index(drop=True), df_std, df_norm],
    axis=1
)

print("\nFinal dataset shape (for Power BI):", df_final.shape)

# -------------------- EXPORT CLEANED DATA --------------------
output_path = r"C:/Users/DELL/Desktop/ML&DL/spotify_data_powerbi.csv"
df_final.to_csv(output_path, index=False)
print("\nCleaned dataset saved to:")
print(output_path)
