import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.metrics import accuracy_score, f1_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
import warnings
import os
warnings.filterwarnings('ignore')

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run song genre classification experiments with logistic regression.')
parser.add_argument('--csv_path', type=str, help='Path to the CSV file containing the music data', required=True)
parser.add_argument('--output_path', type=str, help='Path to save the output CSV file', default='results.csv')
args = parser.parse_args()

# Load the CSV file
try:
    music_numeric = pd.read_csv(args.csv_path)
except FileNotFoundError:
    print(f"Error: File '{args.csv_path}' not found.")
    exit(1)
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit(1)

def preprocess_none(X):
    """No preprocessing—just return scaled data."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def preprocess_lasso(X, y_numeric):
    """Apply LASSO for feature selection, expecting numeric y."""
    # Ensure X is a DataFrame for feature names
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=music_numeric.drop(columns=['terms']).columns)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lasso = Lasso(alpha=0.01)  # Tune alpha as needed
    selector = SelectFromModel(lasso, prefit=False)
    X_selected = selector.fit_transform(X_scaled, y_numeric)
    
    # Get original and selected feature names
    original_features = X.columns.tolist()
    selected_features = X.columns[selector.get_support()].tolist()
    return X_selected, scaler, original_features, selected_features

def preprocess_pca(X, n_components=None):
    """Apply PCA to reduce to n_components (default None, use all features)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    if n_components is None:
        n_components = min(X.shape[1], X.shape[0] - 1)  # Use all features or max possible
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, scaler

def preprocess_covmatrix(X):
    """Use covariance matrix to drop highly correlated features (>0.9)."""
    # Ensure X is a DataFrame for feature names
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=music_numeric.drop(columns=['terms']).columns)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    corr_matrix = pd.DataFrame(X_scaled, columns=X.columns).corr()
    # Find features to drop (correlation > 0.9, keep one of each pair)
    to_drop = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.9:
                colname = corr_matrix.columns[i]
                to_drop.add(colname)
    X_reduced = pd.DataFrame(X_scaled, columns=X.columns).drop(columns=list(to_drop))
    
    # Get original and selected feature names
    original_features = X.columns.tolist()
    selected_features = X_reduced.columns.tolist()
    return X_reduced.values, scaler, original_features, selected_features

def handle_year(df, method='median'):
    """Handle year = 0: 'median' or '-1'."""
    if method == 'median':
        non_zero_years = df['year'][df['year'] != 0]
        median_year = non_zero_years.median()
        df['year'] = df['year'].replace(0, median_year)
    elif method == '-1':
        df['year'] = df['year'].replace(0, -1)
    return df

# Experiment parameters
class_configs = ['All', 'Top100', 'Top75', 'Top50']
preprocessing_methods = ['None', 'LASSO', 'PCA', 'CovMatrix']
year_methods = ['median', 'neg1']

# Create a label encoder for terms (string to numeric)
le = LabelEncoder()

# Initialize results list
results = []

# Write header to output CSV with tab delimiter if file doesn't exist
if not os.path.exists(args.output_path):
    pd.DataFrame([{
        'Method': 'LogisticRegression',
        'Pre-processing': '',
        'Num of classes': '',
        'Year handling': '',
        'Accuracy': '',
        'F1-score': ''
    }]).to_csv(args.output_path, index=False, sep='\t')

# Loop through all combinations
for num_classes in class_configs:
    # Filter dataset based on number of classes
    if num_classes == 'All':
        music = music_numeric.copy()
    else:
        top_n = int(num_classes.replace('Top', ''))
        top_genres = music_numeric['terms'].value_counts().head(top_n).index
        music = music_numeric[music_numeric['terms'].isin(top_genres)].copy()
    
    print(f"\nProcessing {num_classes} classes, {music.shape[0]} samples")

    for year_method in year_methods:
        # Handle year
        music_processed = handle_year(music.copy(), year_method)
        
        # Features and target
        X = music_processed.drop(columns=['terms'])
        y = music_processed['terms']

        # Encode y to numeric for LASSO and logistic regression
        y_encoded = le.fit_transform(y)  # Fit on full y, transform for train/test

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
        y_train_encoded = le.transform(y_train)  # Transform train/test labels
        y_test_encoded = le.transform(y_test)
        
        for pre_method in preprocessing_methods:
            print(f"  Year: {year_method}, Preprocessing: {pre_method}")
            
            # Apply preprocessing
            if pre_method == 'None':
                X_processed, scaler = preprocess_none(X_train)
                X_test_processed = scaler.transform(X_test)
                y_use = y_train_encoded  # Use encoded for consistency
            elif pre_method == 'LASSO':
                X_processed, scaler, original_features, selected_features = preprocess_lasso(X_train, y_train_encoded)
                # Align X_test with all original features, then transform with scaler
                X_test_aligned = X_test.reindex(columns=original_features, fill_value=0)
                X_test_scaled = scaler.transform(X_test_aligned)
                # Subset to selected_features after scaling
                X_test_processed = X_test_scaled[:, [original_features.index(feat) for feat in selected_features]]
                y_use = y_train_encoded
            elif pre_method == 'PCA':
                X_processed, scaler = preprocess_pca(X_train, n_components=X_train.shape[1])  # Use all features
                X_test_processed = scaler.transform(X_test)  # PCA scales, then transforms
                y_use = y_train_encoded
            elif pre_method == 'CovMatrix':
                X_processed, scaler, original_features, selected_features = preprocess_covmatrix(X_train)
                # Align X_test with all original features, then transform with scaler
                X_test_aligned = X_test.reindex(columns=original_features, fill_value=0)
                X_test_scaled = scaler.transform(X_test_aligned)
                # Subset to selected_features after scaling
                X_test_processed = X_test_scaled[:, [original_features.index(feat) for feat in selected_features]]
                y_use = y_train_encoded

            # Train logistic regression
            try:
                log_reg = LogisticRegression(random_state=0, max_iter=1000, multi_class='multinomial')
                log_reg.fit(X_processed, y_use)

                # Predict and evaluate (use encoded for prediction, decode for metrics)
                y_pred_encoded = log_reg.predict(X_test_processed)
                y_pred = le.inverse_transform(y_pred_encoded)  # Decode back to original labels
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')  # Use original labels for F1

                # Store and save results immediately
                result = {
                    'Method': 'LogisticRegression',
                    'Pre-processing': pre_method,
                    'Num of classes': num_classes,
                    'Year handling': year_method,
                    'Accuracy': accuracy,
                    'F1-score': f1
                }
                results.append(result)
                # Append to CSV with tab delimiter without duplicating header
                temp_df = pd.DataFrame([result])
                if os.path.getsize(args.output_path) == 0:  # If file is empty, write header
                    temp_df.to_csv(args.output_path, index=False, sep='\t')
                else:  # Otherwise, append without header
                    temp_df.to_csv(args.output_path, mode='a', header=False, index=False, sep='\t')
                print(f"    Accuracy: {accuracy:.4f}, F1-score: {f1:.4f}")
            except Exception as e:
                print(f"    Error: {e}")
                # Store and save error result
                error_result = {
                    'Method': 'LogisticRegression',
                    'Pre-processing': pre_method,
                    'Num of classes': num_classes,
                    'Year handling': year_method,
                    'Accuracy': None,
                    'F1-score': None
                }
                results.append(error_result)
                temp_df = pd.DataFrame([error_result])
                if os.path.getsize(args.output_path) == 0:  # If file is empty, write header
                    temp_df.to_csv(args.output_path, index=False, sep='\t')
                else:  # Otherwise, append without header
                    temp_df.to_csv(args.output_path, mode='a', header=False, index=False, sep='\t')

# No final save needed since we save after each experiment
print(f"\nResults are being saved incrementally to {args.output_path}")