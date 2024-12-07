from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from typing import Union
import seaborn as sns
import pandas as pd
import numpy as np
import random
import shap


def pre_process_df(df: pd.DataFrame, drop_correlated: bool) -> pd.DataFrame:
    """
    Preprocesses the airline passenger satisfaction dataset.

    Steps:
    1. Checks if the 'data' folder exists. If not, creates it and downloads the dataset.
    2. Loads the dataset and sets the 'ID' column as the index.
    3. Handles missing values:
       - Fills missing values in the 'Arrival Delay' column with 0.
    4. Converts 'Arrival Delay' to integer, as it doesn't require decimals.
    5. Label encodes columns where the order matters:
       - 'Satisfaction': 0 for 'Neutral or Dissatisfied', 1 for 'Satisfied'.
       - 'Class': 0 for 'Economy', 1 for 'Economy Plus', 2 for 'Business'.
    6. One-hot encodes categorical columns where order does not matter:
       - 'Gender', 'Customer Type', and 'Type of Travel'.
       - Renames one-hot encoded columns for better readability.
    7. Ensures all one-hot encoded columns are converted to integer type.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    df.drop(columns=['ID'], inplace=True)
    # 1. Handle missing values using KNN imputer
    knn_imputer = KNNImputer(n_neighbors=5)
    df['Arrival Delay'] = knn_imputer.fit_transform(df[['Arrival Delay']])
    df['Arrival Delay'] = df['Arrival Delay'].astype(int)

    # Create Total Delay feature by summing Arrival Delay and Departure Delay
    df['Total Delay'] = df['Arrival Delay'] + df['Departure Delay']

    # Drop the original delay columns to avoid redundancy
    df.drop(columns=['Arrival Delay', 'Departure Delay'], inplace=True)

    # 2. Label encode columns where order matters
    df['Satisfaction'] = df['Satisfaction'].apply(lambda x: 0 if x == 'Neutral or Dissatisfied' else 1)
    df['Class'] = df['Class'].apply(lambda x: 0 if x == 'Economy' else 1 if x == 'Economy Plus' else 2)
    # 3. One-hot encode categorical columns where order does not matter
    df = pd.get_dummies(df, columns=['Gender'])
    df.rename(columns={'Gender_Female': 'Female',
                       'Gender_Male': 'Male'}, inplace=True)
    df = pd.get_dummies(df, columns=['Customer Type'])
    df.rename(columns={'Customer Type_First-time': 'First-time',
                       'Customer Type_Returning': 'Returning'}, inplace=True)
    df = pd.get_dummies(df, columns=['Type of Travel'])
    df.rename(columns={'Type of Travel_Business': 'Business',
                       'Type of Travel_Personal': 'Personal'}, inplace=True)
    # 4. Ensure all one-hot encoded columns are converted to integer type
    df = df.astype({col: 'int' for col in df.select_dtypes(include=['bool']).columns})
    # 5. Remove highly correlated features
    if drop_correlated:
        correlation_matrix = df.corr()
        upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        excluded_columns =['Satisfaction', 'Class', 'Female', 'Male', 'First-time', 'Returning', 'Business', 'Personal']
        # highly_correlated=[]
        to_drop = []
        for i in range(len(upper_triangle.columns)):
                for j in range(i+1, len(upper_triangle.columns)):
                    feature1 = upper_triangle.columns[i]
                    feature2 = upper_triangle.columns[j]
                    corr_value = upper_triangle.iloc[i, j]

                    if abs(corr_value) > 0.5:
                        # Exclude pairs if either feature is in the excluded columns
                        if feature1 not in excluded_columns and feature2 not in excluded_columns:
                            # Calculate correlation with target
                            target_corr_feature1 = correlation_matrix.loc[feature1, 'Satisfaction']
                            target_corr_feature2 = correlation_matrix.loc[feature2, 'Satisfaction']
                            
                            # Add to the list
                            # highly_correlated.append((feature1, feature2, corr_value))
                            # Determine which feature to keep based on correlation with target
                            if target_corr_feature1 >= target_corr_feature2:
                                # print(f"Keeping: {feature1} | Dropping: {feature2}")
                                to_drop.append(feature2)
                            else:
                                # print(f"Keeping: {feature2} | Dropping: {feature1}")
                                to_drop.append(feature1)
        features_to_drop = ['Ease of Online Booking', 'Food and Drink', 'In-flight Service']

        # Remover as features do DataFrame
        df.drop(columns=features_to_drop, inplace=True)
    return df


# Data Analysis Functions
def visualize_correlation(df: pd.DataFrame) -> None:
    """
    Visualizes the correlation matrix for the dataset using a heatmap.

    Parameters:
        df (pd.DataFrame): The preprocessed DataFrame.
    """
    plt.figure(figsize=(14, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix Heatmap')
    plt.show()


def visualize_class_imbalance(df: pd.DataFrame, target_column: str = 'Satisfaction') -> None:
    """
    Visualizes the class imbalance for the target column in the dataset.

    Parameters:
        df (pd.DataFrame): The preprocessed DataFrame.
        target_column (str): The target column to analyze for class imbalance (default is 'Satisfaction').
    """
    plt.figure(figsize=(8, 6))
    sns.countplot(x=target_column, data=df)
    plt.title(f"Class Imbalance for '{target_column}' Column")
    plt.xlabel('Class Labels')
    plt.ylabel('Frequency')
    plt.show()


def visualize_feature_distributions(df: pd.DataFrame) -> None:
    """
    Visualizes the distributions of the features in the dataset using histograms.

    Parameters:
        df (pd.DataFrame): The preprocessed DataFrame.
    """
    df.hist(bins=20, figsize=(18, 10), edgecolor='black')
    plt.subplots_adjust(hspace=0.5, wspace=0.4)
    plt.suptitle('Feature Distributions', fontsize=16)
    plt.show()


# Classification Functions

def cross_validation_acc(data: pd.DataFrame, model: Union[DecisionTreeClassifier, RandomForestClassifier],
                      cv_fold: int = 5, apply_smote: bool = False) -> float:
    """
    Performs cross-validation for a classification model with SMOTE applied to address class imbalance.

    Parameters:
        data (pd.DataFrame): The dataset containing features and the target.
        model (Union[DecisionTreeClassifier, RandomForestClassifier]): The classification model to evaluate.
        cv_fold (int): Number of cross-validation folds (default is 10).
        apply_smote (bool): Whether to apply SMOTE to the dataset (default is False).

    Returns:
        float: Mean cross-validation score rounded to 3 decimal places.
    """
    X = data.drop('Satisfaction', axis=1)
    y = data['Satisfaction']
    if apply_smote:
        smote = SMOTE(random_state=42)
        X,y = smote.fit_resample(X, y)
    return round(cross_val_score(model, X, y, cv=cv_fold).mean() * 100, 3)


def holdout_accuracy(data: pd.DataFrame, model: Union[DecisionTreeClassifier, RandomForestClassifier], test_size=0.2, apply_smote: bool = False) -> float:
    """
    Trains and evaluates a classification model on the dataset with SMOTE applied to address class imbalance.

    Steps:
    1. Splits the data into training and test sets.
    2. Applies SMOTE to the training set to address class imbalance.
    3. Trains the model on the resampled training set.
    4. Predicts target values for the test set.
    5. Calculates and returns the accuracy score.

    Parameters:
        data (pd.DataFrame): The dataset containing features and the target.
        model (Union[DecisionTreeClassifier, RandomForestClassifier]): The classification model to evaluate.
        test_size (float): The proportion of the dataset to include in the test split (default is 0.2).
        apply_smote (bool): Whether to apply SMOTE to the training set (default is False).

    Returns:
        float: Accuracy score rounded to 3 decimal places.
    """
    X = data.drop('Satisfaction', axis=1)
    y = data['Satisfaction']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    if apply_smote:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return round(accuracy_score(y_test, y_pred)*100, 3)


# Task 2:
def analyze_tree_complexity(tree: DecisionTreeClassifier, df: pd.DataFrame,
                            target_column: str = 'Satisfaction') -> None:
    """
    Produces graphs showing detailed information about the complexity and structure of the decision tree.

    Parameters:
        tree (DecisionTreeClassifier): Trained decision tree model
        df (pd.DataFrame): DataFrame containing the features and target
        target_column (str): The name of the target column
    """
    # Infer feature names
    feature_names = df.drop(columns=[target_column]).columns.tolist()

    # Basic tree properties
    depth = tree.get_depth()
    n_leaves = tree.get_n_leaves()
    n_nodes = tree.tree_.node_count

    # Tree structure analysis
    tree_structure = tree.tree_
    children_left = tree_structure.children_left
    children_right = tree_structure.children_right

    # Count internal nodes vs leaf nodes
    internal_nodes = 0
    leaf_nodes = 0
    for i in range(n_nodes):
        if children_left[i] == children_right[i]:
            leaf_nodes += 1
        else:
            internal_nodes += 1

    print(f"Tree Depth: {depth}\n"
          f"Number of Leaves: {n_leaves}\n"
          f"Number of Nodes: {n_nodes}\n"
          f"Internal Nodes: {internal_nodes}\n"
          f"Leaf Nodes: {leaf_nodes}")

    feature_importances = tree.feature_importances_

    feature_importance_info = sorted(zip(feature_names, feature_importances), key=lambda x: x[1], reverse=True)
    feature_names_sorted, importances_sorted = zip(*feature_importance_info)
    importances_percentage = [importance * 100 for importance in importances_sorted]

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names_sorted, importances_percentage, color='skyblue')
    plt.xlabel('Importance (%)')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()
    plt.show()


# Task 3
## Task 3.1: Simplification-based Technique
def apply_simplification_based_xai(model: RandomForestClassifier, data: pd.DataFrame) -> None:
    """
    Applies a simplification-based XAI technique (using a decision tree to approximate a random forest).

    Parameters:
        model (RandomForestClassifier): The trained black-box model.
        data (pd.DataFrame): The dataset containing features and the target.

    Returns:
        None: Prints accuracy and displays the visualized decision tree.
    """
    X = data.drop('Satisfaction', axis=1)
    y = data['Satisfaction']

    # Train a decision tree to approximate the random forest
    surrogate_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    surrogate_tree.fit(X, model.fit(X,y).predict(X))

    # Evaluate the surrogate model
    accuracy = holdout_accuracy(data, surrogate_tree)
    print(f"Surrogate Model Accuracy: {accuracy}")


## Task 3.2: Feature-based Techniques
def apply_feature_based_xai(model: RandomForestClassifier, data: pd.DataFrame) -> None:
    """
    Applies two feature-based XAI techniques (Permutation Importance and SHAP Values) to the random forest model.

    Parameters:
        model (RandomForestClassifier): The trained black-box model.
        data (pd.DataFrame): The dataset containing features and the target.

    Returns:
        None: Prints and compares feature importance metrics.
    """
    X = data.drop('Satisfaction', axis=1)
    y = data['Satisfaction']

    # Permutation Importance
    print("Permutation Importance:")
    perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    perm_importances_df = pd.Series(perm_importance.importances_mean, index=X.columns)
    perm_importances_df = perm_importances_df.sort_values(ascending=False)
    print(perm_importances_df)

    # SHAP Values
    print("\nSHAP Values:")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X, plot_type="bar")

    # Compare insights
    print("\nFeature-based explanations compared:")
    print("- Permutation Importance and SHAP provide different but complementary insights into feature relevance.")


## Task 3.3: Example-based Techniques
def apply_example_based_xai(model: RandomForestClassifier, data: pd.DataFrame, target_column: str,
                            num_examples: int = 3) -> None:
    """
    Applies an example-based XAI technique (SHAP force plot) for individual predictions.

    Parameters:
        model (RandomForestClassifier): The trained black-box model.
        data (pd.DataFrame): The dataset containing features and the target.
        target_column (str): Name of the target column in the dataset.
        num_examples (int): Number of random examples to explain. Default is 3.

    Returns:
        None: Displays SHAP force plots for the given examples.
    """
    # Split features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Initialize SHAP TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Randomly select examples
    random_indices = random.sample(range(len(X)), num_examples)

    for i, idx in enumerate(random_indices):
        print(f"Explanation for Example {i + 1} (Index: {idx}):")
        sample = X.iloc[idx]
        shap.force_plot(
            explainer.expected_value[1],
            shap_values[1][idx],
            sample,
            matplotlib=True
        )
