import os
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# Reusing your existing code for data loading and feature extraction
def extract_features(node, depth=0, parent_tag=None, sibling_count=0, parent_tag_html=None):
    # Your existing extract_features function (unchanged)
    features = []
    
    tag = node.get("tag", "")
    node_data = node.get("node", {})
    node_type = str(node_data.get("type", ""))

    text = node_data.get("characters", "")
    text_length = len(text)
    word_count = len(text.split()) if text else 0
    contains_number = any(ch.isdigit() for ch in text)
    contains_special_chars = any(not ch.isalnum() and not ch.isspace() for ch in text)
    
    children = node.get("children", [])
    num_children = len(children)
    is_leaf = 1 if num_children == 0 else 0
    
    feature = {
        "tag": tag,
        "type": node_type,
        "x": node_data.get("x", 0),
        "y": node_data.get("y", 0),
        "width": node_data.get("width", 0),
        "height": node_data.get("height", 0),
        "characters": text,
        "has_text": int(bool(text)),
        "depth": depth,
        "num_children": num_children,
        "parent_tag": parent_tag if parent_tag else "",
        "parent_tag_html": parent_tag_html if parent_tag_html else "",
        "sibling_count": sibling_count,
        "is_leaf": is_leaf,
        "font_size": node_data.get("fontSize", 16),
        "has_font_size": int("fontSize" in node_data),
        "font_name": node_data.get("fontName", {}).get("style", "") if node_data.get("fontName") else "normal",
        "has_text_color": 0, "color_r": 0, "color_g": 0, "color_b": 0,
        "has_background_color": 0, "background_r": 0, "background_g": 0, "background_b": 0,
        "border_radius": 0,
        "border_r": 0, "border_g": 0, "border_b": 0,
        "has_border": 0, "border_opacity": 0,
        "border_weight": node_data.get("strokeWeight", 0),
        "has_shadow": 0, "shadow_r": 0, "shadow_g": 0, "shadow_b": 0,
        "shadow_radius": 0, 
        "text_length": text_length,
        "word_count": word_count,
        "contains_number": int(contains_number),
        "contains_special_chars": int(contains_special_chars),
    }
    
    # Extract fills (background and text color)
    fills = node_data.get("fills", [])
    for fill in fills:
        if fill.get("type") == "SOLID" and "color" in fill:
            r, g, b = (
                int(fill["color"].get("r", 0) * 255),
                int(fill["color"].get("g", 0) * 255),
                int(fill["color"].get("b", 0) * 255),
            )
            feature["color_r"], feature["color_g"], feature["color_b"] = r, g, b
            feature["has_text_color"] = 1
            
            feature["background_r"], feature["background_g"], feature["background_b"] = r, g, b
            feature["has_background_color"] = 1
            break  
    
    # Extract strokes (borders)
    strokes = node_data.get("strokes", [])
    if strokes:
        stroke = strokes[0]
        feature["has_border"] = 1
        if "color" in stroke:
            feature["border_r"], feature["border_g"], feature["border_b"] = (
                int(stroke["color"].get("r", 0) * 255),
                int(stroke["color"].get("g", 0) * 255),
                int(stroke["color"].get("b", 0) * 255),
            )
        feature["border_opacity"] = stroke.get("opacity", 0)
    
    # Extract border radius
    br_top_left = node_data.get("topLeftRadius", 0)
    br_top_right = node_data.get("topRightRadius", 0)
    br_bottom_left = node_data.get("bottomLeftRadius", 0)
    br_bottom_right = node_data.get("bottomRightRadius", 0)
    
    if any([br_top_left, br_top_right, br_bottom_left, br_bottom_right]):
        feature["border_radius"] = (br_top_left + br_top_right + br_bottom_left + br_bottom_right) / 4
    
    # Extract shadow
    effects = node_data.get("effects", [])
    for effect in effects:
        if effect.get("type") == "DROP_SHADOW":
            feature["has_shadow"] = 1
            if "color" in effect:
                feature["shadow_r"], feature["shadow_g"], feature["shadow_b"] = (
                    int(effect["color"].get("r", 0) * 255),
                    int(effect["color"].get("g", 0) * 255),
                    int(effect["color"].get("b", 0) * 255),
                )
            feature["shadow_radius"] = effect.get("radius", 0)
            break  
    
    features.append(feature)
    
    for child in children:
        features.extend(extract_features(child, depth=depth+1, parent_tag=node_type, sibling_count=len(children)-1, parent_tag_html=tag))
    
    return features

# Main execution script
def process_figma_data(data_folder, output_csv_file):
    # Modified version of your processing script that retains all features for analysis
    
    normalize_columns = [
    "area",
    "word_count",
    # "text_length",
    "font_size",
    # "sibling_count",
    # "num_children",
    "height",
    "width",
    "depth"
    ]
    
    # Collection of all dataframes for combined analysis
    all_dfs = []
    
    # Iterate over all JSON files in the data folder
    for filename in os.listdir(data_folder):
        if filename.endswith(".json"):
            file_path = os.path.join(data_folder, filename)
            print(f"Processing {file_path}...")
            
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            
            # Extract features using the recursive function starting at the root
            features_list = extract_features(data, depth=0, parent_tag=None, sibling_count=0, parent_tag_html=None)
            if not features_list:
                continue  # Skip if no features extracted
            
            df = pd.DataFrame(features_list)
            
            # Normalize positions and add derived features
            min_x = df['x'].min() if df['x'].notnull().any() else 0
            min_y = df['y'].min() if df['y'].notnull().any() else 0
            df['x_normalized'] = df['x'] - min_x
            df['y_normalized'] = df['y'] - min_y
            
            df['x_center'] = df['x'] + df['width'] / 2
            df['y_center'] = df['y'] + df['height'] / 2
            
            # Compute dimensions
            body_node = df[df['tag'] == 'BODY']
            if not body_node.empty:
                total_width = body_node.iloc[0]['width']
                total_height = body_node.iloc[0]['height']
            else:
                total_width = (df['x'] + df['width']).max()
                total_height = (df['y'] + df['height']).max()
            
            # Avoid division by zero
            if total_width and total_height:
                df['x_quarter'] = df['x_center'] / total_width
                df['y_quarter'] = df['y_center'] / total_height
            else:
                df['x_quarter'] = 0
                df['y_quarter'] = 0
            
            df['aspect_ratio'] = df.apply(
                lambda row: row['width'] / row['height'] if row['height'] and row['height'] != 0 else 0, axis=1
            )
            df['area'] = df['width'] * df['height']
            
            if total_width:
                df['normalized_width'] = df['width'] / total_width
            else:
                df['normalized_width'] = 0
                
            if total_height:
                df['normalized_height'] = df['height'] / total_height
            else:
                df['normalized_height'] = 0
                
            
            # Text density (characters per area)
            df['text_density'] = df.apply(
                lambda row: row['text_length'] / row['area'] if row['area'] > 0 else 0, axis=1
            )


            

            # Compute min and max for each column
            min_max_values = {col: (df[col].min(), df[col].max()) for col in normalize_columns}

            # Apply Min-Max normalization (scaling between 0 and 1)
            for col in normalize_columns:
                min_val, max_val = min_max_values[col]
                if max_val > min_val and max_val != 0:  # Avoid division by zero
                    df[col] = df[col] / max_val
                else:
                    df[col] = 0  # If min and max are the same, set to 0

            df = df.drop(columns=['x'])
            df = df.drop(columns=['y'])
            df = df.drop(columns=['x_normalized'])
            df = df.drop(columns=['y_normalized'])
            df = df.drop(columns=['x_center'])
            df = df.drop(columns=['y_center'])
            df = df.drop(columns=['characters'])
            df = df.drop(columns=['font_size'])
            df = df.drop(columns=['font_name'])
            df = df.drop(columns=['color_r'])
            df = df.drop(columns=['color_g'])
            df = df.drop(columns=['color_b'])
            df = df.drop(columns=['background_r'])
            df = df.drop(columns=['background_g'])
            df = df.drop(columns=['background_b'])
            df = df.drop(columns=['border_radius'])
            df = df.drop(columns=['border_r'])
            df = df.drop(columns=['border_g'])
            df = df.drop(columns=['border_b'])
            df = df.drop(columns=['border_opacity'])
            df = df.drop(columns=['border_weight'])
            df = df.drop(columns=['shadow_r'])
            df = df.drop(columns=['shadow_g'])
            df = df.drop(columns=['shadow_b'])
            df = df.drop(columns=['shadow_radius'])
            df = df.drop(columns=['word_count'])
            df = df.drop(columns=['normalized_width'])
            df = df.drop(columns=['normalized_height'])
            df = df.drop(columns=['contains_special_chars'])
            df = df.drop(columns=['contains_number'])
            df = df.drop(columns=['has_shadow'])
            df = df.drop(columns=['has_border'])
            df = df.drop(columns=['has_text_color'])
            df = df.drop(columns=['has_font_size'])
            df = df.drop(columns=['height'])
            df = df.drop(columns=['has_text'])
            df = df.drop(columns=['depth'])
            df = df.drop(columns=['x_quarter'])
            df = df.drop(columns=['y_quarter'])
            df = df.drop(columns=['area'])
            df = df.drop(columns=['text_density'])


            
            # Add to collection of all dataframes
            all_dfs.append(df)
    
    # Combine all dataframes
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Save the full dataset for analysis
        combined_df.to_csv("figma_full_dataset.csv", index=False)
        
        # Perform feature analysis
        combined_df = combined_df[~combined_df['tag'].str.contains('-')]

        analyze_features(combined_df, output_csv_file)
    else:
        print("No data processed.")

def analyze_features(df, output_csv_file):
    """Memory-efficient implementation for analyzing features, performing PCA, and creating optimized datasets
    that can handle large datasets without excessive memory usage"""
    
    # Keep a copy of the tag column for classification
    tag_column = df['tag'].copy()
    
    # Get unique tag values and print them
    unique_tags = tag_column.unique()
    print(f"\nUnique tag values ({len(unique_tags)}):")
    print(unique_tags)
    
    # Remove tag temporarily for correlation and PCA analysis
    df = df.drop(columns=['tag'])
    
    # First, handle categorical features by encoding them
    categorical_cols = ['type', 'parent_tag', 'parent_tag_html']
    
    # Only include categorical columns that exist in the DataFrame
    existing_categorical_cols = [col for col in categorical_cols if col in df.columns]
    
    # Convert categorical features to one-hot encoding if they exist
    if existing_categorical_cols:
        df_encoded = pd.get_dummies(df, columns=existing_categorical_cols)
    else:
        df_encoded = df.copy()
    
    # Select only numeric columns for correlation and PCA
    numeric_df = df_encoded.select_dtypes(include=['int64', 'float64'])
    
    # Drop columns with all zeros or all the same value
    numeric_df = numeric_df.loc[:, (numeric_df != numeric_df.iloc[0]).any()]
    
    # Replace NaN values with 0 for analysis
    numeric_df = numeric_df.fillna(0)
    
    # Check dataset size and warn if too large
    num_samples, num_features = numeric_df.shape
    print(f"Dataset shape: {num_samples} samples, {num_features} features")
    
    # Create feature names list for future reference
    feature_names = numeric_df.columns.tolist()
    
    # 1. Memory-Efficient Correlation Analysis 
    # Only compute correlation for a subset of features if the dataset is large
    MAX_FEATURES_FOR_CORRELATION = 1000  # Maximum number of features for correlation analysis
    
    if num_features > MAX_FEATURES_FOR_CORRELATION:
        print(f"\nToo many features ({num_features}) for full correlation analysis. Using feature selection first.")
        
        # Use variance threshold to reduce dimensionality for correlation analysis
        from sklearn.feature_selection import VarianceThreshold
        selector = VarianceThreshold(threshold=0.01)  # Remove features with low variance
        reduced_data = selector.fit_transform(numeric_df)
        
        # Get selected feature names
        selected_mask = selector.get_support()
        selected_features = [feature for i, feature in enumerate(feature_names) if selected_mask[i]]
        
        # If still too many features, take top features by variance
        if len(selected_features) > MAX_FEATURES_FOR_CORRELATION:
            # Get variances of each feature
            variances = selector.variances_[selected_mask]
            # Get indices of top features by variance
            top_indices = np.argsort(variances)[-MAX_FEATURES_FOR_CORRELATION:]
            selected_features = [selected_features[i] for i in top_indices]
        
        print(f"Reduced to {len(selected_features)} features for correlation analysis")
        correlation_subset = numeric_df[selected_features]
        correlation_matrix = correlation_subset.corr()
    else:
        print("\nPerforming correlation analysis...")
        correlation_matrix = numeric_df.corr()
    
    # Save correlation matrix visualization (if not too large)
    if correlation_matrix.shape[0] <= 100:  # Only create heatmap if not too large
        plt.figure(figsize=(16, 14))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm', 
                    linewidths=0.5, vmin=-1, vmax=1)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png')
        plt.close()
    
    # Save top correlations
    top_correlations = get_top_correlations(correlation_matrix)
    with open("correlation_analysis.txt", "w", encoding="utf-8") as f:
        f.write(top_correlations)
    
    # 2. Memory-Efficient Dimensionality Reduction
    # Use efficient methods to handle large datasets
    print("\nPerforming dimensionality reduction...")
    
    # 2.1 - First, use feature selection to reduce dimensionality if needed
    from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
    
    # Encode the 'tag' column for classification
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(tag_column)
    
    # Save the label mapping for reference
    label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
    pd.DataFrame(list(label_mapping.items()), columns=['Encoded', 'Tag']).to_csv('tag_encoding.csv', index=False)
    
    # Print stats
    print(f"Number of samples: {num_samples}")
    print(f"Number of features: {num_features}")
    print(f"Number of unique tags: {len(np.unique(y))}")
    
    # Apply variance threshold to remove constant features
    var_selector = VarianceThreshold(threshold=0.01)
    X_var_selected = var_selector.fit_transform(numeric_df)
    
    # Get selected feature names after variance threshold
    var_selected_mask = var_selector.get_support()
    var_selected_features = [feature for i, feature in enumerate(feature_names) if var_selected_mask[i]]
    
    print(f"After variance threshold: {X_var_selected.shape[1]} features")
    
    # Determine maximum number of features to keep for efficient processing
    MAX_FEATURES_FOR_PCA = 2000  # Adjust based on available memory
    
    # If still too many features, use SelectKBest with f_classif for additional reduction
    if X_var_selected.shape[1] > MAX_FEATURES_FOR_PCA:
        print(f"Still too many features. Using SelectKBest to reduce to {MAX_FEATURES_FOR_PCA} features")
        k_best = min(MAX_FEATURES_FOR_PCA, X_var_selected.shape[1] // 2)
        k_selector = SelectKBest(f_classif, k=k_best)
        X_selected = k_selector.fit_transform(X_var_selected, y)
        
        # Get final selected feature names 
        k_selected_mask = k_selector.get_support()
        final_features = [feature for i, feature in enumerate(var_selected_features) if k_selected_mask[i]]
        
        # Create a DataFrame with selected features for later use
        final_X = pd.DataFrame(X_selected, columns=[f"feature_{i}" for i in range(X_selected.shape[1])])
    else:
        X_selected = X_var_selected
        final_features = var_selected_features
        final_X = pd.DataFrame(X_selected, columns=final_features)
    
    print(f"Final feature count for dimensionality reduction: {X_selected.shape[1]}")
    
    # 2.2 Perform PCA with a chunked approach for large datasets
    from sklearn.decomposition import PCA, IncrementalPCA
    from sklearn.preprocessing import StandardScaler
    
    # Use IncrementalPCA for very large datasets
    pca_n_components = min(50, X_selected.shape[1], X_selected.shape[0] // 10)
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    
    # Use regular PCA if dataset is small enough, otherwise use IncrementalPCA
    if num_samples * X_selected.shape[1] < 10_000_000:  # If data matrix is small enough
        print("Using standard PCA...")
        pca = PCA(n_components=pca_n_components)
        pca_result = pca.fit_transform(X_scaled)
    else:
        print("Using IncrementalPCA for large dataset...")
        batch_size = min(1000, X_scaled.shape[0] // 10)  # Determine batch size
        ipca = IncrementalPCA(n_components=pca_n_components, batch_size=batch_size)
        pca_result = ipca.fit_transform(X_scaled)
        pca = ipca  # Use ipca as the PCA object
    
    # Save explained variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    # Plot explained variance
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_variance)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance')
    plt.grid(True)
    plt.savefig('pca_explained_variance.png')
    plt.close()
    
    # Create a DataFrame with PCA transformed data
    pca_df = pd.DataFrame(
        pca_result,
        columns=[f'PC{i+1}' for i in range(pca_result.shape[1])]
    )
    
    # Add original tag values to the PCA dataset
    pca_df['tag'] = tag_column.values
    
    # Save PCA transformed dataset
    pca_df.to_csv('figma_pca_dataset.csv', index=False)
    print(f"\nPCA transformed dataset saved to 'figma_pca_dataset.csv'")
    
    # 3. Memory-Efficient Feature Importance
    print("\nCalculating feature importance...")
    
    # Use a memory-efficient approach to calculate feature importance
    # 3.1 Train a Random Forest with reduced complexity
    from sklearn.ensemble import RandomForestClassifier
    
    # Lower n_estimators for large datasets
    n_estimators = 50 if num_samples > 10000 else 100
    max_depth = 20  # Limit tree depth
    
    # Use class weights if classes are imbalanced
    class_weight = None
    if len(unique_tags) > 1:
        class_counts = np.bincount(y)
        if max(class_counts) / min(class_counts) > 10:
            class_weight = 'balanced'
    
    # Train Random Forest classifier on selected features
    rf = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth,
        class_weight=class_weight, 
        random_state=42,
        n_jobs=-1  # Use all available cores
    )
    
    rf.fit(X_selected, y)
    
    # Get feature importances
    importances = rf.feature_importances_
    
    # Create feature names for importance report
    if len(importances) != len(final_features):
        # If dimensions don't match, use generic feature names
        importance_features = [f"feature_{i}" for i in range(len(importances))]
    else:
        importance_features = final_features
    
    feature_importance = pd.DataFrame({
        'Feature': importance_features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # Save feature importances
    feature_importance.to_csv('feature_importance.csv', index=False)
    
    # Plot top 20 features (or fewer if less available)
    plt.figure(figsize=(12, 8))
    top_n = min(20, len(feature_importance))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(top_n))
    plt.title(f'Top {top_n} Features by Importance for Tag Classification')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    # 4. Feature importance on PCA components
    print("\nCalculating feature importance for tag classification on PCA components...")
    
    # Train a simpler Random Forest on PCA components (which should be much fewer)
    rf_pca = RandomForestClassifier(
        n_estimators=100, 
        random_state=42,
        class_weight=class_weight,
        n_jobs=-1  # Use all available cores
    )
    rf_pca.fit(pca_result, y)
    
    # Get feature importances for PCA components
    pca_importances = rf_pca.feature_importances_
    pca_importance = pd.DataFrame({
        'Component': [f'PC{i+1}' for i in range(len(pca_importances))],
        'Importance': pca_importances
    }).sort_values(by='Importance', ascending=False)
    
    # Save PCA feature importances
    pca_importance.to_csv('pca_feature_importance.csv', index=False)
    
    # Plot PCA component importance
    plt.figure(figsize=(12, 8))
    top_n_pca = min(20, len(pca_importance))
    sns.barplot(x='Importance', y='Component', data=pca_importance.head(top_n_pca))
    plt.title(f'Top {top_n_pca} PCA Components by Importance for Tag Classification')
    plt.tight_layout()
    plt.savefig('pca_feature_importance.png')
    plt.close()
    
    # 5. Create optimized datasets
    # 5a. Create feature selected dataset
    print("\nCreating feature selected dataset...")
    
    # Determine number of features to select (at most 100 for manageability)
    n_features_to_select = min(100, len(feature_importance), max(10, len(feature_importance) // 10))
    
    # Select top features
    selected_features = feature_importance.head(n_features_to_select)['Feature'].tolist()
    
    print(f"Selected {len(selected_features)} important features")
    
    # Create feature-selected dataset
    if len(selected_features) != len(final_features):
        # We're using generic feature names
        # Change this line:
        
        # To something like this:
        selected_indices = []
        for f in selected_features:
            parts = f.split('_')
            if len(parts) > 1 and parts[0] == 'feature':
                try:
                    selected_indices.append(int(parts[1]))
                except ValueError:
                    # Handle the case where the second part isn't a number
                    print(f"Warning: Could not parse feature name: {f}")
            else:
                print(f"Warning: Feature name does not match expected format: {f}")
                # You might need to handle this case differently based on your data
        feature_selected_data = X_selected[:, selected_indices]
        feature_selected_df = pd.DataFrame(
            feature_selected_data,
            columns=[f"feature_{i}" for i in selected_indices]
        )
    else:
        # We can use the original feature names
        feature_selected_df = final_X[selected_features].copy()
    
    # Add the original tag column 
    feature_selected_df['tag'] = tag_column.values
    
    # Save the feature-selected dataset
    feature_selected_df.to_csv("figma_feature_selected.csv", index=False)
    print(f"Feature-selected dataset saved to 'figma_feature_selected.csv'")
    
    # 5b. Create PCA-transformed dataset with important components
    print("\nCreating optimized PCA dataset...")
    
    # Select top PCA components based on feature importance
    n_components_to_select = min(10, len(pca_importance))
    top_pca_components = pca_importance.head(n_components_to_select)['Component'].tolist()
    
    # Create optimized PCA dataset with important components
    optimized_pca_df = pca_df[top_pca_components + ['tag']]
    
    # Save the optimized PCA dataset
    optimized_pca_df.to_csv(output_csv_file, index=False)
    print(f"Optimized PCA dataset saved to {output_csv_file}")
    
    # 6. Create visualization of PCA separation by tag
    print("\nCreating PCA visualization by tag...")
    
    # Use first two principal components for visualization
    if pca_result.shape[1] >= 2:
        plt.figure(figsize=(12, 10))
        
        # If too many unique tags, limit to top N most frequent
        if len(unique_tags) > 10:
            # Get top 10 most frequent tags
            tag_counts = pd.Series(tag_column).value_counts()
            top_tags = tag_counts.head(10).index.tolist()
            
            # Create mask for samples with top tags
            mask = np.isin(tag_column, top_tags)
            scatter_tags = tag_column[mask]
            scatter_data = pca_result[mask]
            
            # Create a categorical colormap
            cmap = plt.cm.get_cmap('tab10', len(top_tags))
            
            # Create a mapping of tags to integers for coloring
            tag_to_int = {tag: i for i, tag in enumerate(top_tags)}
            scatter_colors = [tag_to_int[tag] for tag in scatter_tags]
            
            scatter = plt.scatter(
                scatter_data[:, 0], 
                scatter_data[:, 1], 
                c=scatter_colors, 
                cmap=cmap, 
                alpha=0.6, 
                s=50
            )
            
            # Add legend
            handles, _ = scatter.legend_elements()
            plt.legend(handles, top_tags, title="Tags")
        else:
            # If few enough tags, show all
            # Create a categorical colormap
            cmap = plt.cm.get_cmap('tab10', len(unique_tags))
            
            # Create a mapping of tags to integers for coloring
            tag_to_int = {tag: i for i, tag in enumerate(unique_tags)}
            colors = [tag_to_int[tag] for tag in tag_column]
            
            scatter = plt.scatter(
                pca_result[:, 0], 
                pca_result[:, 1], 
                c=colors, 
                cmap=cmap, 
                alpha=0.6, 
                s=50
            )
            
            # Add legend
            handles, _ = scatter.legend_elements()
            plt.legend(handles, unique_tags, title="Tags")
        
        plt.title('PCA Visualization of Figma Elements by Tag')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.tight_layout()
        plt.savefig('pca_tag_visualization.png')
        plt.close()
    
    # 7. Estimate classification accuracy with cross-validation
    print("\nEstimating classification accuracy...")
    
    # Use the PCA result for a quick cross-validation
    from sklearn.model_selection import cross_val_score
    
    # Simple cross-validation to avoid memory issues
    n_folds = min(5, len(unique_tags))  # Don't use more folds than classes
    
    try:
        # Cross-validate on PCA data, which should be memory-efficient
        cv_scores = cross_val_score(
            RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
            pca_result,
            y,
            cv=n_folds
        )
        
        print(f"Cross-validation accuracy on PCA data: {cv_scores.mean():.4f} (std: {cv_scores.std():.4f})")
        
        # Save cross-validation results
        with open("classification_results.txt", "w") as f:
            f.write(f"Cross-validation accuracy on PCA data:\n")
            f.write(f"Mean accuracy: {cv_scores.mean():.4f}\n")
            f.write(f"Standard deviation: {cv_scores.std():.4f}\n")
            f.write(f"Individual fold scores: {cv_scores}\n")
    except Exception as e:
        print(f"Error during cross-validation: {e}")
    
    return {
        'n_samples': num_samples,
        'n_features': num_features,
        'n_selected_features': len(selected_features),
        'n_pca_components': len(top_pca_components),
        'unique_tags': len(unique_tags)
    }


def get_top_correlations(correlation_matrix, threshold=0.5):
    """Get feature pairs with correlation above threshold, sorted by absolute correlation"""
    corr_pairs = []
    
    # Get correlation pairs above threshold
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                corr_pairs.append((
                    correlation_matrix.columns[i],
                    correlation_matrix.columns[j],
                    correlation_matrix.iloc[i, j]
                ))
    
    # Sort by absolute correlation value (descending)
    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    # Limit to top 1000 pairs to avoid massive output
    corr_pairs = corr_pairs[:1000]
    
    # Format as string
    result = "Top correlated feature pairs:\n\n"
    for pair in corr_pairs:
        result += f"{pair[0]} and {pair[1]}: {pair[2]:.4f}\n"
    
    return result

if __name__ == "__main__":
    # Folder containing JSON files and output path
    data_folder = "json_data"
    output_csv_file = "figma_optimized_dataset.csv"
    
    # Process data and perform feature analysis

    all_dfs = pd.read_csv('figma_full_dataset.csv')

    # Combine all dataframes
    
    # Perform feature analysis
    all_dfs = all_dfs[~all_dfs['tag'].str.contains('-')]
    analyze_features(all_dfs, output_csv_file)