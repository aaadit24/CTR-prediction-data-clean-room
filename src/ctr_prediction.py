from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

def create_ctr_model():
    return Pipeline([
        ('preprocessor', ColumnTransformer([
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), ['u_phonePrice', 'u_browserLifeCycle', 'u_feedLifeCycle', 'u_refreshTimes', 
                 'i_regionEntity', 'i_cat', 'i_dislikeTimes', 'i_upTimes', 
                 'age', 'gender', 'city_rank', 'app_score'])
        ])),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Print AUC-ROC score
    print(f"AUC-ROC Score: {roc_auc_score(y_test, y_pred_proba)}")

    # Feature importance
    feature_importance = model.named_steps['classifier'].feature_importances_
    feature_names = model.named_steps['preprocessor'].get_feature_names_out()

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importance, y=feature_names)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')

if __name__ == "__main__":
    # This section would be used for testing the CTR prediction model independently
    # You can add code here to load a sample dataset and test the model
    pass