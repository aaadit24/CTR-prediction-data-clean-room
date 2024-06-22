import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from imblearn.over_sampling import SMOTE

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create output directory
os.makedirs('output', exist_ok=True)

# Load and preprocess data
def load_and_preprocess_data(feeds_path, ads_path, sample_size=100000):
    df_feeds = pd.read_csv(feeds_path, nrows=sample_size, low_memory=False)
    df_ads = pd.read_csv(ads_path, nrows=sample_size, low_memory=False)
    
    df_merged = pd.merge(df_feeds, df_ads, left_on='u_userId', right_on='user_id', how='inner', suffixes=('_x', '_y'))
    
    features = ['u_phonePrice', 'u_browserLifeCycle', 'u_feedLifeCycle_x', 'u_refreshTimes_x', 
                'i_regionEntity', 'i_cat', 'i_dislikeTimes', 'i_upTimes', 
                'age', 'gender', 'city_rank', 'app_score']
    
    # Check which features are actually available in the merged dataset
    available_features = [f for f in features if f in df_merged.columns]
    
    X = df_merged[available_features]
    y = df_merged['label_y']
    
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Calculate aggregate statistics
def calculate_aggregate_statistics(df):
    stats = {}
    if 'u_phonePrice' in df.columns:
        stats['average_phone_price'] = df['u_phonePrice'].mean()
    if 'u_refreshTimes_x' in df.columns:
        stats['average_refresh_times'] = df['u_refreshTimes_x'].mean()
    if 'label_y' in df.columns:
        stats['click_through_rate'] = df['label_y'].mean()
    if 'app_score' in df.columns:
        stats['average_app_score'] = df['app_score'].mean()
    
    return stats

# Generate synthetic data using SMOTE
def generate_synthetic_data(X, y, n_samples):
    smote = SMOTE(sampling_strategy=0.5, random_state=42)
    X_synthetic, y_synthetic = smote.fit_resample(X, y)
    
    # Ensure we have exactly n_samples
    if len(X_synthetic) > n_samples:
        X_synthetic = X_synthetic[:n_samples]
        y_synthetic = y_synthetic[:n_samples]
    elif len(X_synthetic) < n_samples:
        # If we need more samples, just duplicate some randomly
        extra_samples = n_samples - len(X_synthetic)
        indices = np.random.choice(len(X_synthetic), extra_samples)
        X_synthetic = pd.concat([X_synthetic, X_synthetic.iloc[indices]], ignore_index=True)
        y_synthetic = pd.concat([y_synthetic, y_synthetic.iloc[indices]], ignore_index=True)
    
    synthetic_df = pd.concat([X_synthetic, y_synthetic], axis=1)
    return synthetic_df

# Plot feature importance
def plot_feature_importance(model, features, output_file):
    importances = model.named_steps['classifier'].feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 8))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

# Plot Precision-Recall curve
def plot_pr_curve(y_true, y_scores, output_file):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig(output_file)
    plt.close()

# Evaluate model
def evaluate_model(model, X_train, X_test, y_train, y_test, features, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"\nModel Performance on {model_name} Data:")
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba)}")
    average_precision = average_precision_score(y_test, y_pred_proba)
    print(f"Average Precision Score: {average_precision}")
    
    plot_feature_importance(model, features, f'output/{model_name.lower()}_feature_importance.png')
    plot_pr_curve(y_test, y_pred_proba, f'output/{model_name.lower()}_pr_curve.png')
    
    return average_precision

if __name__ == "__main__":

    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(
        './data/train/train_data_feeds.csv',
        './data/train/train_data_ads.csv'
    )
    
    # Calculate aggregate statistics
    df_for_stats = pd.concat([X_train, y_train], axis=1)
    aggregate_stats = calculate_aggregate_statistics(df_for_stats)
    print("Aggregate Statistics:")
    for key, value in aggregate_stats.items():
        print(f"{key}: {value}")
    
    # Generate synthetic data
    n_synthetic_samples = len(X_train) + len(X_test)
    synthetic_df = generate_synthetic_data(pd.concat([X_train, X_test]), pd.concat([y_train, y_test]), n_synthetic_samples)
    print("\nSynthetic Data Sample:")
    print(synthetic_df.head())
    
    # Prepare models
    features = X_train.columns
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))

    model = Pipeline([('preprocessor', ColumnTransformer([('num', Pipeline([('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())]), features)])),('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weight_dict))])
    
    # Evaluate original model
    original_ap = evaluate_model(model, X_train, X_test, y_train, y_test, features, "Original")
    
    # Evaluate synthetic model
    X_synthetic = synthetic_df[features]
    y_synthetic = synthetic_df['label_y']
    synthetic_ap = evaluate_model(model, X_synthetic, X_test, y_synthetic, y_test, features, "Synthetic")
    
    print(f"\nOriginal Model Average Precision: {original_ap}")
    print(f"Synthetic Model Average Precision: {synthetic_ap}")
    
