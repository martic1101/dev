import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')




def preprocess(df):
    """Clean and feature-engineer the Titanic dataset."""
    data = df.copy()

    # --- Fill missing values ---
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
    data['Fare'].fillna(data['Fare'].median(), inplace=True)

    # --- Extract Title from Name ---
    data['Title'] = data['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    # Group rare titles
    rare_titles = [t for t in data['Title'].unique()
                   if data[data['Title'] == t].shape[0] < 10]
    data['Title'] = data['Title'].replace(rare_titles, 'Rare')

    # --- Create FamilySize ---
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

    # --- IsAlone ---
    data['IsAlone'] = (data['FamilySize'] == 1).astype(int)


    le_sex = LabelEncoder()
    data['Sex'] = le_sex.fit_transform(data['Sex'])

    le_emb = LabelEncoder()
    data['Embarked'] = le_emb.fit_transform(data['Embarked'])

    le_title = LabelEncoder()
    data['Title'] = le_title.fit_transform(data['Title'])

    # --- Select features ---
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch',
                'Fare', 'Embarked', 'Title', 'FamilySize', 'IsAlone']

    X = data[features]
    y = data['Survived']

    return X, y, features



print("Loading data...")
train_df = pd.read_csv('train.csv')
X, y, feature_names = preprocess(train_df)
print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Survived ratio: {y.mean():.2%}\n")



print("=" * 50)
print("Fine-tuning Decision Tree...")
print("=" * 50)

dt_params = {
    'max_depth': [3, 4, 5, 6, 7, 8, 10],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 6],
    'criterion': ['gini', 'entropy']
}

dt_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    dt_params,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
dt_grid.fit(X, y)

best_dt = dt_grid.best_estimator_
print(f"Best DT params: {dt_grid.best_params_}")
print(f"Best DT GridSearch accuracy: {dt_grid.best_score_:.4f}")

# Plot decision tree
fig, ax = plt.subplots(1, 1, figsize=(24, 12))
plot_tree(
    best_dt,
    feature_names=feature_names,
    class_names=['Died', 'Survived'],
    filled=True,
    rounded=True,
    fontsize=8,
    ax=ax
)
ax.set_title("Fine-tuned Decision Tree - Titanic", fontsize=16)
plt.tight_layout()
plt.savefig('decision_tree.png', dpi=150, bbox_inches='tight')
plt.close()
print("Decision tree plot saved to: decision_tree.png\n")



print("=" * 50)
print("5-Fold Cross Validation - Decision Tree")
print("=" * 50)

dt_cv_scores = cross_val_score(best_dt, X, y, cv=5, scoring='accuracy')
print(f"Fold accuracies: {dt_cv_scores}")
print(f"Average accuracy: {dt_cv_scores.mean():.4f} (+/- {dt_cv_scores.std():.4f})\n")



print("=" * 50)
print("Fine-tuning Random Forest...")
print("=" * 50)

rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, 10],
    'max_features': ['sqrt', 'log2'],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_params,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
rf_grid.fit(X, y)

best_rf = rf_grid.best_estimator_
print(f"Best RF params: {rf_grid.best_params_}")
print(f"Best RF GridSearch accuracy: {rf_grid.best_score_:.4f}")

rf_cv_scores = cross_val_score(best_rf, X, y, cv=5, scoring='accuracy')
print(f"\n5-Fold CV accuracies: {rf_cv_scores}")
print(f"Average accuracy: {rf_cv_scores.mean():.4f} (+/- {rf_cv_scores.std():.4f})\n")



print("=" * 50)
print("COMPARISON SUMMARY")
print("=" * 50)
print(f"Decision Tree  - Avg Accuracy: {dt_cv_scores.mean():.4f} (std: {dt_cv_scores.std():.4f})")
print(f"Random Forest  - Avg Accuracy: {rf_cv_scores.mean():.4f} (std: {rf_cv_scores.std():.4f})")
print()

if rf_cv_scores.mean() > dt_cv_scores.mean():
    print("=> Random Forest performs better than Decision Tree.")
else:
    print("=> Decision Tree performs better than Random Forest.")

