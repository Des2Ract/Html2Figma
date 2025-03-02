import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

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

            
            # Add to collection of all dataframes
            all_dfs.append(df)
    
    # Combine all dataframes
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Save the full dataset for analysis
        combined_df.to_csv("figma_full_dataset.csv", index=False)
        
        # Perform feature analysis
        analyze_features(combined_df, output_csv_file)
    else:
        print("No data processed.")

def analyze_features(df, output_csv_file):
    """Analyze features, perform PCA and create optimized dataset"""
    
    # First, handle categorical features by encoding them
    # For simplicity, we'll use one-hot encoding for tag and type
    categorical_cols = ['tag', 'type', 'parent_tag', 'parent_tag_html']
    temp_tag= df['tag']
    
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
    
    # 1. Correlation Analysis
    print("\nPerforming correlation analysis...")
    correlation_matrix = numeric_df.corr()
    
    # Save correlation matrix
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
    
    # 2. Principal Component Analysis (PCA)
    print("\nPerforming PCA...")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)
    
    # Determine number of components to keep 95% of variance
    pca = PCA(n_components=0.95)
    pca_result = pca.fit_transform(scaled_data)
    
    print(f"Number of components to keep 95% of variance: {pca.n_components_}")
    
    # Plot explained variance
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance')
    plt.grid(True)
    plt.savefig('pca_explained_variance.png')
    plt.close()
    
    # Save component loadings
    component_df = pd.DataFrame(
        pca.components_, 
        columns=numeric_df.columns,
        index=[f'PC{i+1}' for i in range(pca.n_components_)]
    )
    component_df.to_csv('pca_components.csv')
    
    # 3. Feature Importance using Random Forest
    print("\nCalculating feature importance...")
    
    # Create a simple target for demonstration (using is_leaf as an example)
    # In a real scenario, you would use your actual classification target
    if 'is_leaf' in numeric_df.columns:
        X = numeric_df.drop('is_leaf', axis=1)
        y = numeric_df['is_leaf']
        
        # Train a Random Forest to get feature importances
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Get feature importances
        importances = rf.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        # Save feature importances
        feature_importance.to_csv('feature_importance.csv', index=False)
        
        # Plot top 20 features
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
        plt.title('Top 20 Features by Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
        
        # Select top features using SelectFromModel
        selector = SelectFromModel(rf, prefit=True, threshold='mean')
        X_selected = selector.transform(X)
        selected_features = X.columns[selector.get_support()]
        
        print(f"\nSelected {len(selected_features)} important features out of {len(X.columns)}")
        
        # Create final dataset with selected features
        selected_cols = list(selected_features)
        if 'is_leaf' in df.columns:  # Add back the target if it exists
            selected_cols.append('is_leaf')
            
        # Keep original categorical features
        for col in existing_categorical_cols:
            # Find all one-hot encoded columns for this categorical feature
            one_hot_cols = [c for c in df_encoded.columns if c.startswith(f"{col}_")]
            selected_cols.extend(one_hot_cols)
        
        # Create optimized dataset
        optimized_df = df_encoded[selected_cols]
        optimized_df = optimized_df.replace({False: 0, True: 1})
        optimized_df['tag'] = temp_tag

        optimized_df.to_csv(output_csv_file, index=False)
        
        print(f"\nOptimized dataset with selected features saved to {output_csv_file}")
    else:
        print("'is_leaf' column not found for feature importance calculation.")
        # Just use PCA results for the final dataset
        pca_df = pd.DataFrame(
            pca_result,
            columns=[f'PC{i+1}' for i in range(pca.n_components_)]
        )
        pca_df.to_csv(output_csv_file, index=False)
        print(f"\nPCA transformed dataset saved to {output_csv_file}")

def get_top_correlations(correlation_matrix, threshold=0.7):
    """Extract top correlations from the matrix"""
    
    # Convert to a DataFrame for easier manipulation
    corr_df = pd.DataFrame(correlation_matrix.stack()).reset_index()
    corr_df.columns = ['Feature1', 'Feature2', 'Correlation']
    
    # Remove self-correlations and duplicates
    corr_df = corr_df[corr_df['Feature1'] != corr_df['Feature2']]
    corr_df['pair'] = corr_df.apply(lambda row: tuple(sorted([row['Feature1'], row['Feature2']])), axis=1)
    corr_df = corr_df.drop_duplicates(subset=['pair'])
    corr_df = corr_df.drop('pair', axis=1)
    
    # Get absolute correlation and sort
    corr_df['AbsCorrelation'] = corr_df['Correlation'].abs()
    corr_df = corr_df.sort_values('AbsCorrelation', ascending=False)
    
    # Filter by threshold
    strong_correlations = corr_df[corr_df['AbsCorrelation'] >= threshold]
    
    # Format as string
    result = "Strong Feature Correlations (|r| >= {}):\n".format(threshold)
    result += "=" * 50 + "\n"
    
    for _, row in strong_correlations.iterrows():
        result += f"{row['Feature1']} ↔ {row['Feature2']}: {row['Correlation']:.4f}\n"
    
    return result

if __name__ == "__main__":
    # Folder containing JSON files and output path
    data_folder = "json_data"
    output_csv_file = "figma_optimized_dataset.csv"
    
    # Process data and perform feature analysis
    process_figma_data(data_folder, output_csv_file)