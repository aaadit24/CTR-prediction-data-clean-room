import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess data
def load_and_preprocess_data(feeds_path, ads_path, sample_size=100000):
    df_feeds = pd.read_csv(feeds_path, nrows=sample_size)
    df_ads = pd.read_csv(ads_path, nrows=sample_size)
    df_merged = pd.merge(df_feeds, df_ads, left_on='u_userId', right_on='user_id', how='inner')
    
    features = ['u_phonePrice', 'u_browserLifeCycle', 'u_feedLifeCycle', 'u_refreshTimes', 
                'i_regionEntity', 'i_cat', 'i_dislikeTimes', 'i_upTimes', 
                'age', 'gender', 'city_rank', 'app_score']
    
    X = df_merged[features]
    y = df_merged['label_y']
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Calculate aggregate statistics
def calculate_aggregate_statistics(df):
    stats = {
        'total_users': df['u_userId'].nunique(),
        'average_phone_price': df['u_phonePrice'].mean(),
        'average_refresh_times': df['u_refreshTimes'].mean(),
        'click_through_rate': df['label_y'].mean(),
        'average_app_score': df['app_score'].mean()
    }
    return stats

# Generate synthetic data
def generate_synthetic_data(df, n_samples):
    synthetic_data = {}
    for column in df.columns:
        if df[column].dtype == 'object':
            synthetic_data[column] = np.random.choice(df[column].unique(), n_samples)
        else:
            mean = df[column].mean()
            std = df[column].std()
            synthetic_data[column] = np.random.normal(mean, std, n_samples)
    
    return pd.DataFrame(synthetic_data)

# Evaluate synthetic data
def evaluate_synthetic_data(original_df, synthetic_df, X_test, y_test, features):
    # Compare distributions
    for column in original_df.columns:
        if original_df[column].dtype != 'object':
            plt.figure(figsize=(10, 6))
            sns.kdeplot(data=original_df[column], label='Original')
            sns.kdeplot(data=synthetic_df[column], label='Synthetic')
            plt.title(f'Distribution Comparison: {column}')
            plt.legend()
            plt.savefig(f'distribution_comparison_{column}.png')
            plt.close()

    # Train model on synthetic data and evaluate on original test set
    X_synthetic = synthetic_df[features]
    y_synthetic = synthetic_df['label_y']
    
    model_synthetic = Pipeline([
        ('preprocessor', ColumnTransformer([
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), features)
        ])),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    model_synthetic.fit(X_synthetic, y_synthetic)
    y_pred_synthetic = model_synthetic.predict(X_test)
    
    print("\nModel Performance on Synthetic Data:")
    print(classification_report(y_test, y_pred_synthetic))

if __name__ == "__main__":
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(
        './data/train/train_data_feeds.csv',
        './data/train/train_data_ads.csv'
    )
    
    # Calculate aggregate statistics
    df_merged = pd.concat([X_train, y_train], axis=1)
    aggregate_stats = calculate_aggregate_statistics(df_merged)
    print("Aggregate Statistics:")
    for key, value in aggregate_stats.items():
        print(f"{key}: {value}")
    
    # Generate synthetic data
    n_synthetic_samples = 10000
    synthetic_df = generate_synthetic_data(df_merged, n_synthetic_samples)
    print("\nSynthetic Data Sample:")
    print(synthetic_df.head())
    
    # Evaluate synthetic data
    evaluate_synthetic_data(df_merged, synthetic_df, X_test, y_test, X_train.columns)