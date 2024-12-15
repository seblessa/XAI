from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from lime.lime_tabular import LimeTabularExplainer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from alibi.explainers import AnchorTabular
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
from sklearn.tree import _tree
from pandas import DataFrame
from typing import Union
import seaborn as sns
import pandas as pd
import numpy as np
import tqdm
import shap
import yaml


def pre_process_df(df: pd.DataFrame, drop_correlated: bool) -> pd.DataFrame:
    """
    Preprocesses the airline passenger satisfaction dataset.

    Steps:
    1. Checks if the 'data' folder exists. If not, creates it and downloads the dataset.
    2. Loads the dataset and sets the 'ID' column as the index.
    3. Handles missing values:
       - Fills missing values in the 'Arrival Delay' column with 0.
    4. Converts 'Arrival Delay' to integer, and creates a new feature 'Total Delay' by summing 'Arrival Delay' and 'Departure Delay'.
    5. Label encodes columns where the order matters:
       - 'Satisfaction': 0 for 'Neutral or Dissatisfied', 1 for 'Satisfied'.
       - 'Class': 0 for 'Economy', 1 for 'Economy Plus', 2 for 'Business'.
    6. One-hot encodes categorical columns where order does not matter:
       - 'Gender', 'Customer Type', and 'Type of Travel'.
       - Renames one-hot encoded columns for better readability.
    7. Ensures all one-hot encoded columns are converted to integer type.
    8. Drops highly correlated features if 'drop_correlated' is True.

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
        excluded_columns = ['Satisfaction', 'Class', 'Female', 'Male', 'First-time', 'Returning', 'Business',
                            'Personal']
        # highly_correlated=[]
        to_drop = []
        for i in range(len(upper_triangle.columns)):
            for j in range(i + 1, len(upper_triangle.columns)):
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


def visualize_data_with_pca(X: pd.DataFrame, y: pd.Series):
    """
    Visualize the dataset in 2D using PCA, with color indicating the target classes.

    Parameters:
    - X (pd.DataFrame): Feature dataset.
    - y (pd.Series): Target variable (0: Neutral or Dissatisfied, 1: Satisfied).

    Returns:
    - None: Displays the PCA visualization.
    """
    # Standardize the data for PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA to reduce to 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Print the explained variance ratio for each principal component
    print("Explained variance ratio per component:")
    print(pca.explained_variance_ratio_)

    # Create a scatter plot for the two classes
    plt.figure(figsize=(10, 8))

    # Plot Neutral or Dissatisfied (class 0)
    plt.scatter(
        X_pca[y == 0, 0], X_pca[y == 0, 1],
        c='red', label='Neutral or Dissatisfied (0)',
        alpha=0.7, edgecolor='k', s=50
    )

    # Plot Satisfied (class 1)
    plt.scatter(
        X_pca[y == 1, 0], X_pca[y == 1, 1],
        c='green', label='Satisfied (1)',
        alpha=0.7, edgecolor='k', s=50
    )

    # Add plot details
    plt.title('PCA Visualization of Dataset')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


# Classification Functions
def split_data(X: pd.DataFrame, y: pd.DataFrame, test_size: float = 0.2, apply_smote: bool = False) -> tuple:
    """
    Splits the dataset into features and target variables, and further splits them into training and test sets.

    Parameters:
        X (pd.DataFrame): The dataset containing features.
        y (pd.DataFrame): The target variable.
        test_size (float): The proportion of the dataset to include in the test split (default is 0.2).
        apply_smote (bool): Whether to apply SMOTE to the training set (default is False).

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    if apply_smote:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
    return X_train, X_test, y_train, y_test


def holdout_accuracy(X: pd.DataFrame, y: pd.DataFrame, model: Union[DecisionTreeClassifier, RandomForestClassifier],
                     test_size=0.2, apply_smote: bool = False) -> float:
    """
    Trains and evaluates a classification model on the dataset with SMOTE applied to address class imbalance.

    Steps:
    1. Splits the data into training and test sets.
    2. Applies SMOTE to the training set to address class imbalance.
    3. Trains the model on the resampled training set.
    4. Predicts target values for the test set.
    5. Calculates and returns the accuracy score.

    Parameters:
        X (pd.DataFrame): The dataset containing features.
        y (pd.DataFrame): The target variable.
        model (Union[DecisionTreeClassifier, RandomForestClassifier]): The classification model to evaluate.
        test_size (float): The proportion of the dataset to include in the test split (default is 0.2).
        apply_smote (bool): Whether to apply SMOTE to the training set (default is False).

    Returns:
        float: Accuracy score rounded to 3 decimal places.
    """
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size, apply_smote=apply_smote)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return round(accuracy_score(y_test, y_pred) * 100, 3)


def cross_validation_acc(X: pd.DataFrame, y: pd.DataFrame, model: Union[DecisionTreeClassifier, RandomForestClassifier],
                         cv_fold: int = 5, apply_smote: bool = False) -> float:
    """
    Performs cross-validation for a classification model with SMOTE applied to address class imbalance.

    Parameters:
        X (pd.DataFrame): The dataset containing features.
        y (pd.DataFrame): The target variable.
        model (Union[DecisionTreeClassifier, RandomForestClassifier]): The classification model to evaluate.
        cv_fold (int): Number of cross-validation folds (default is 10).
        apply_smote (bool): Whether to apply SMOTE to the dataset (default is False).

    Returns:
        float: Mean cross-validation score rounded to 3 decimal places.
    """
    if apply_smote:
        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y)
    return round(cross_val_score(model, X, y, cv=cv_fold).mean() * 100, 3)


# Task 2:
def analyze_tree_complexity(X: pd.DataFrame, y: pd.DataFrame, tree: DecisionTreeClassifier) -> None:
    """
    Produces graphs showing detailed information about the complexity and structure of the decision tree.

    Parameters:
        X (pd.DataFrame): The dataset containing features.
        y (pd.DataFrame): The target variable.
        tree (DecisionTreeClassifier): Trained decision tree model
    """

    tree.fit(X, y)

    feature_names = X.columns.tolist()

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

    print(f"Tree Depth: {depth} - "
          f"Number of Leaves: {n_leaves} - "
          f"Number of Nodes: {n_nodes} - "
          f"Internal Nodes: {internal_nodes} - "
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
def apply_surrogate_models_xai(X: pd.DataFrame, y: pd.DataFrame, model: RandomForestClassifier) -> None:
    """
    Applies a simplification-based XAI technique (using a decision tree to approximate a random forest).

    Parameters:
        X (pd.DataFrame): The dataset containing features.
        y (pd.DataFrame): The target variable.
        model (RandomForestClassifier): The trained black-box model.

    Returns:
        None: Prints accuracy, agreement rate and log loss,.
    """

    # Split the dataset into training and testing sets (for more reliable evaluation)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train a decision tree to approximate the random forest
    surrogate_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    surrogate_tree.fit(X_train, model.fit(X_train, y_train).predict(X_train))

    # Predict on the test set using both the surrogate model and the original model
    y_pred_surrogate = surrogate_tree.predict(X_test)
    y_pred_original = model.predict(X_test)

    # Calculate Agreement Rate
    agreement_rate = np.mean(y_pred_surrogate == y_pred_original)
    print(f"Agreement Rate: {agreement_rate * 100:.2f}%")

    # Calculate Log Loss (needs probabilities)
    y_pred_surrogate_prob = surrogate_tree.predict_proba(X_test)
    y_pred_original_prob = model.predict_proba(X_test)

    # We assume that y is categorical with integer labels, so we calculate log loss
    log_loss_value = log_loss(y_test, y_pred_surrogate_prob, labels=np.unique(y), sample_weight=None)
    print(f"Log Loss (Surrogate vs Original): {log_loss_value:.4f}")

    # Print the accuracy of the surrogate model
    accuracy = np.mean(y_pred_surrogate == y_test)
    print(f"Surrogate Model Accuracy: {accuracy * 100:.2f}%")


def apply_rule_extraction_xai(X: pd.DataFrame, y: pd.DataFrame, model: RandomForestClassifier,
                              min_coverage: int = 10) -> tuple[DataFrame, DataFrame]:
    """
    Extracts, prunes, and summarizes rules from a Random Forest model.

    Parameters:
        X (pd.DataFrame): The dataset containing features.
        y (pd.DataFrame): The target variable.
        model (RandomForestClassifier): The Random Forest model.
        min_coverage (int): The minimum coverage threshold to keep a rule.

    Returns:
        pd.DataFrame: A DataFrame containing the pruned rules and their metadata.
    """
    model.fit(X, y)

    feature_names = X.columns
    rules_list = []

    # Step 1: Rule Extraction
    def extract_rules_from_tree(tree, feature_names):
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]

        def recurse(node, conditions):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                left = tree_.children_left[node]
                right = tree_.children_right[node]

                # Add condition for the left branch
                recurse(left, conditions + [f"{name} <= {threshold:.2f}"])
                # Add condition for the right branch
                recurse(right, conditions + [f"{name} > {threshold:.2f}"])
            else:
                # Leaf node: collect rule and its class
                value = tree_.value[node]
                predicted_class = value.argmax()
                coverage = int(value.sum())
                rules_list.append({
                    "rule": " AND ".join(conditions),
                    "predicted_class": predicted_class,
                    "coverage": coverage
                })

        recurse(0, [])  # Start recursion from root

    # Extract rules from all trees in the forest
    for idx, tree in enumerate(model.estimators_):
        extract_rules_from_tree(tree, feature_names)

    # Convert rules to a DataFrame
    rules_df = pd.DataFrame(rules_list)

    # Step 2: Prune Rules
    def prune_rules(rules_df: pd.DataFrame, min_coverage: int) -> pd.DataFrame:
        """
        Prunes the extracted rules by filtering out those with low coverage.

        Parameters:
            rules_df (pd.DataFrame): The DataFrame containing extracted rules.
            min_coverage (int): The minimum coverage threshold to keep a rule.

        Returns:
            pd.DataFrame: The pruned DataFrame of rules.
        """
        # Filter rules with coverage above the threshold
        pruned_rules = rules_df[rules_df['coverage'] >= min_coverage].copy()
        return pruned_rules.reset_index(drop=True)

    pruned_rules_df = prune_rules(rules_df, min_coverage)

    # Step 3: Summarize Rules
    def extract_feature_name(condition: str) -> str:
        """
        Extracts all the words to the left of the comparison operator ('>', '<=', etc.) from a condition.

        Parameters:
            condition (str): A single condition string, e.g., "In-flight service today > 45".

        Returns:
            str: The feature name, e.g., "In-flight service today".
        """
        # Define the operators in order of priority
        operators = ['>=', '<=', '>', '<']

        # Find and split using the first operator encountered
        for op in operators:
            if op in condition:
                return condition.split(op)[0].strip()

        # Return the condition itself if no operator is found
        return condition.strip()

    def summarize_rules(rules_df: pd.DataFrame) -> DataFrame:
        """
        Summarizes the extracted rules.

        Parameters:
            rules_df (pd.DataFrame): The DataFrame containing extracted rules.

        Returns:
            None: Prints a summary of rules and their characteristics.
        """
        # Count rules by predicted class
        class_summary = rules_df.groupby('predicted_class').size()

        # Analyze feature frequency in rules
        feature_frequency = {}
        for rule in rules_df['rule']:
            # Split the rule into individual conditions
            conditions = rule.split(" AND ")
            # Extract the full feature name from each condition
            for condition in conditions:
                feature_name = extract_feature_name(condition)
                feature_frequency[feature_name] = feature_frequency.get(feature_name, 0) + 1

        # Create a sorted DataFrame for feature frequency
        feature_summary = pd.DataFrame.from_dict(feature_frequency, orient='index', columns=['frequency'])
        feature_summary = feature_summary.sort_values(by='frequency', ascending=False)

        return feature_summary

    feature_summary = summarize_rules(pruned_rules_df)

    return feature_summary, pruned_rules_df


def plot_feature_importance_from_rules(feature_summary: pd.DataFrame):
    """
    Plots a bar chart of feature importance based on the frequency of feature occurrences in the extracted rules.

    Parameters:
        feature_summary (pd.DataFrame): The DataFrame containing the feature frequencies from rule extraction.

    Returns:
        None: Displays a bar chart of feature importance.
    """
    # Normalize the importance (to sum up to 100%)
    feature_summary['Importance (%)'] = (feature_summary['frequency'] / feature_summary['frequency'].sum()) * 100

    # Sort the features by importance
    feature_summary = feature_summary.sort_values(by='Importance (%)', ascending=False)

    # Plot the graph
    plt.figure(figsize=(10, 6))
    plt.barh(feature_summary.index, feature_summary['Importance (%)'], color='skyblue')
    plt.xlabel('Importance (%)')
    plt.title('Feature Importance Based on Rule Extraction')
    plt.gca().invert_yaxis()  # Invert the y-axis to show the most important features on top
    plt.show()


def parse_condition(condition: str):
    """
        Parse a condition like "Feature1 <= 45" into feature, operator, and value.

        Parameters:
            condition (str): A condition string like 'Feature1 <= 45'.

        Returns:
            feature (str): The name of the feature (e.g., 'Feature1').
            operator (str): The comparison operator (e.g., '<=' or '>').
            value (float): The value to compare (e.g., 45).
        """
    operators = ['<=', '<', '>=', '>', '=']

    # Try to find the operator in the condition
    for operator in operators:
        if operator in condition:
            feature, value = condition.split(operator)
            return feature.strip(), operator, float(value.strip())

    # If no operator is found, raise an error
    raise ValueError(f"Invalid condition format: {condition}")


## Task 3.2: Feature-based Techniques
def apply_permutation_importance_xai(X: pd.DataFrame, y: pd.DataFrame, model: RandomForestClassifier) -> None:
    """
    Explain a prediction using permutation importance.

    Parameters:
    - X: Features (pandas DataFrame).
    - y: Target variable (pandas DataFrame or Series).
    - model: Trained RandomForestClassifier model.

    Returns:
    A bar chart visualizing the Permutation Importance of each feature.
    """
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train the model
    model.fit(X_train, y_train)

    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    perm_importances_df = pd.Series(perm_importance.importances_mean, index=X_train.columns)
    perm_importances_df = perm_importances_df.sort_values(ascending=False)

    # Plotting Permutation Importance
    plt.figure(figsize=(10, 6))
    perm_importances_df.plot(kind='barh', color='skyblue', edgecolor='black')
    plt.title("Permutation Importance (Feature Importance)")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.gca().invert_yaxis()
    plt.show()


def apply_lime_xai(X: pd.DataFrame, y: pd.DataFrame, model: RandomForestClassifier,
                   sample_index: int = 0) -> None:
    """
    Explain a prediction using LIME and display the explanation.

    Parameters:
    - X_train: Training features (pandas DataFrame).
    - X_test: Test features (pandas DataFrame).
    - y_train: Training target variable (pandas Series).
    - y_test: Test target variable (pandas Series).
    - model: Trained model with a predict_proba method.
    - sample_index: Index of the sample in the test set to explain.

    Returns:
    - None: Displays the LIME explanation.
    """
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train the model
    model.fit(X_train, y_train)

    # Create a LimeTabularExplainer
    explainer = LimeTabularExplainer(
        X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=[str(cls) for cls in np.unique(y_train)],
        mode='classification'
    )

    # Select the sample to explain
    sample = X_test.iloc[sample_index].values

    # Define the prediction function
    def predict_proba_fn(data):
        data_df = pd.DataFrame(data, columns=X_train.columns)
        return model.predict_proba(data_df)

    # Check the shape of the prediction to determine the output format
    sample_prediction = predict_proba_fn([sample])
    if sample_prediction.ndim == 1 or sample_prediction.shape[1] == 1:
        # Model returns a single probability; infer the probability of class 0
        def predict_proba_fn_adjusted(data):
            probs = predict_proba_fn(data)
            if probs.ndim == 1:
                probs = np.expand_dims(probs, axis=1)
            return np.hstack([1 - probs, probs])
    else:
        # Model returns probabilities for both classes
        predict_proba_fn_adjusted = predict_proba_fn


    # Generate explanation
    explanation = explainer.explain_instance(
        sample,
        predict_proba_fn_adjusted,
        num_features=X_train.shape[1]
    )

    # Display the explanation
    #explanation.show_in_notebook()
    explanation.as_pyplot_figure()
    # make the title of the plot so it appears the number of the sample
    plt.title(f"LIME Explanation for Sample {sample_index}")
    plt.show()

    # Evaluate fidelity of LIME explanation using local fidelity (MSE)
    # Get original model's prediction
    original_model_prediction = predict_proba_fn_adjusted([sample])[0][1]

    # Get surrogate model's prediction
    surrogate_prediction = explanation.local_pred

    # Compute fidelity score (MSE)
    fidelity_score = mean_squared_error([original_model_prediction], [surrogate_prediction])

    print(f"Local fidelity Error (MSE) of the LIME explanation for sample {sample_index}: {fidelity_score:.4f}")


def apply_shap_xai(X: pd.DataFrame, y: pd.DataFrame, model: RandomForestClassifier, subset_sizes: int = [10, 50, 100]) -> None:
    """
    Explain a prediction using SHAP with a subset of the data for faster computation.

    Parameters:
    - X: Features (pandas DataFrame).
    - X_train: Training features (pandas DataFrame).
    - X_test: Test features (pandas DataFrame).
    - y_train: Training target variable (pandas DataFrame or Series).
    - y_test: Test target variable (pandas DataFrame or Series).
    - model: RandomForestClassifier model.
    - subset_size: Number of samples to use for SHAP explanation. Default is 100.

    Returns:
    - SHAP explanation visualization in the notebook.
    """
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train the model
    model.fit(X_train, y_train)

    for subset_size in subset_sizes:
        # Take a subset of the test data
        subset = X_test.sample(subset_size, random_state=42)

        # Create a SHAP explainer for the Random Forest model
        explainer = shap.TreeExplainer(model)

        # Compute SHAP values for the subset
        shap_values = explainer.shap_values(subset)
        shap_values_class_0 = shap_values[0]  # Valores SHAP para a classe 0
        shap_values_class_1 = shap_values[1]

        base_value_0 = explainer.expected_value[0]  # Base value for class 0
        # print("Base value for classe 0", base_value_0)
        base_value_1 = explainer.expected_value[1]  # Base value for class 1
        # print("Base value for classe 1", base_value_1)

    # SHAP bar plot (average absolute SHAP value per feature)
    plt.figure(figsize=(10, 6))
    plt.title("SHAP Bar Plot (Feature Importance)")
    shap.summary_plot(shap_values, subset)
    # Summary plot para a classe 0
    plt.title("Summary plot for class 0 - Neutral or Dissatisfied")
    shap.summary_plot(shap_values_class_0, subset)

    # Summary plot para a classe 1
    plt.title("Summary plot for class 1 - Satisfied")
    shap.summary_plot(shap_values_class_1, subset)


## Task 3.3: Example-based Techniques
def find_max_and_min_coverage(X: pd.DataFrame, y: pd.DataFrame, model: RandomForestClassifier) -> tuple:
    """
    Calls the generate_anchor_rule and finds the data instance with the maximum and minimum coverage for an anchor rule.

    Parameters:
    - X (pd.DataFrame): The feature dataset.
    - y (pd.Series): The target variable.
    - model (sklearn.base.BaseEstimator): The trained model.

    Returns:
    - tuple: The index of the data instance with the highest and lowest coverage for the anchor rule.

    """

    # return the values found of the max and min coverage after running this function one time
    return 3786, 3243, 3221, 3848

    max_coverage = {0: float('-inf'), 1: float('-inf')}
    min_coverage = {0: float('inf'), 1: float('inf')}
    high_cov_idx = {0: 0, 1: 0}
    low_cov_idx = {0: 0, 1: 0}

    for i in tqdm(range(X.shape[0]), desc="Finding Max and Min Coverage"):
        explanation = generate_anchor_rule(X, y, model, i)
        label = y[i]
        if explanation.coverage > max_coverage[label]:
            max_coverage[label] = explanation.coverage
            high_cov_idx[label] = i
            with open(f'max_coverage{label}.yaml', 'w') as file:
                yaml.dump({'max_coverage': max_coverage[label], 'max_id': high_cov_idx[label]}, file)
        if explanation.coverage < min_coverage[label]:
            min_coverage[label] = explanation.coverage
            low_cov_idx[label] = i
            with open(f'min_coverage{label}.yaml', 'w') as file:
                yaml.dump({'min_coverage': min_coverage[label], 'min_id': low_cov_idx[label]}, file)

    with open('final_coverages.yaml', 'w') as file:
        yaml.dump({
            'max_coverage0': max_coverage[0], 'max_id0': high_cov_idx[0],
            'min_coverage0': min_coverage[0], 'min_id0': low_cov_idx[0],
            'max_coverage1': max_coverage[1], 'max_id1': high_cov_idx[1],
            'min_coverage1': min_coverage[1], 'min_id1': low_cov_idx[1]
        }, file)

    return high_cov_idx[0], high_cov_idx[1], low_cov_idx[0], low_cov_idx[1]


def generate_anchor_rule(X, y, model, instance_index):
    """
    Generate and print an anchor rule for a specific data instance using a RandomForestClassifier.

    Parameters:
    - X (pd.DataFrame): The feature dataset.
    - y (pd.Series): The target variable.
    - model (sklearn.base.BaseEstimator): The trained model.
    - instance_index (int): Index of the sample to explain.
    """
    # Train the model
    model.fit(X, y)

    # Define the predictor function to handle feature names
    def predict_fn(data):
        data_df = pd.DataFrame(data, columns=X.columns)
        return model.predict(data_df)

    # Initialize the AnchorTabular explainer
    explainer = AnchorTabular(predictor=predict_fn, feature_names=X.columns.tolist())
    explainer.fit(X.values)

    # Select the instance to explain
    instance = X.iloc[instance_index].values.reshape(1, -1)

    # Generate the explanation
    return explainer.explain(instance[0], threshold=0.85)


def evaluate_explanation(explanation, instance_index, target_value):
    """
    Evaluates the quality of an anchor explanation.

    Parameters:
    - explanation: AnchorTabular explanation object.
    - instance_index: Index of the explained instance.
    - target_value: Target value of the instance.
    """
    print(f"\nEvaluation for row {instance_index}:")
    print(f" - Target value: {target_value}")
    print(f" - Rule: {' AND '.join(explanation.anchor)}")
    print(f" - Precision: {explanation.precision * 100:.2f}% (accuracy of rule predictions)")
    print(f" - Coverage: {explanation.coverage * 100:.2f}% (dataset fraction where rule applies)")
    print(f" - Sparsity: {len(explanation.anchor)} feature(s) used in the rule (simplicity)")
