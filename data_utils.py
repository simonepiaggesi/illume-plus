import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_instance_names(X_file):
    """
    Load instance names from X files (train_X, test_X, val_X).
    Files contain just instance names, one per row, no headers.
    """
    print(f"Loading instance names from {X_file}...")
    
    # Read CSV without header (header=None prevents skipping first row)
    df = pd.read_csv(X_file, header=None)
    instance_names = df.iloc[:, 0].values
    
    print(f"Loaded {len(instance_names)} instances")
    print(f"First few instances: {instance_names[:3]}")
    
    return instance_names

def load_labels_predictions(Y_file):
    """
    Load labels/predictions from Y files (train_Y, test_Y, val_Y).
    Format: 
    ,0
    1909,-0.2594548731766541
    1909,0.4737919514578127
    """
    print(f"Loading labels/predictions from {Y_file}...")
    
    # Skip first row (,0 header), no header, get second column
    df = pd.read_csv(Y_file, skiprows=1, header=None)
    labels = df.iloc[:, 1].values  # Second column contains the values we want
    
    print(f"Loaded {len(labels)} labels/predictions")
    print(f"First few values: {labels[:3]}")
    
    return labels


def load_probas_predictions(Y_file):
    """
    Load labels/predictions from Y files (train_Y, test_Y, val_Y).
    Format: 
    ,0
    1909,-0.2594548731766541, -0.4825945731766541, -0.5425945487317661 
    1909,0.4737919514578127, -0.3725945487766541, -0.7625945487316541
    """
    print(f"Loading labels/predictions from {Y_file}...")
    
    # Skip first row (,0 header), no header, get second column
    df = pd.read_csv(Y_file, skiprows=1, header=None)
    labels = df.iloc[:, 1:].values  # Second column contains the values we want
    
    print(f"Loaded {len(labels)} labels/predictions")
    print(f"First few values: {labels[:3]}")
    
    return labels

def load_transcriptomics_data(transcriptomics_file='transcriptomics.feather'):
    """Load transcriptomics data and index by instance names."""
    print(f"Loading transcriptomics data from {transcriptomics_file}...")
    
    transcriptomics = pd.read_feather(transcriptomics_file)
    print(f"Transcriptomics shape: {transcriptomics.shape}")
    # Set first column (instance names) as index for easy lookup
    instance_col = transcriptomics.columns[0]
    transcriptomics_indexed = transcriptomics.set_index(instance_col)
    transcriptomics_indexed.drop('Source', axis = 1, inplace=True)
    
    return transcriptomics_indexed

def load_feature_order(feature_order_file='column.csv'):
    """Load ordered feature names from column.csv."""
    print(f"Loading feature order from {feature_order_file}...")
    
    feature_order_df = pd.read_csv(feature_order_file, header=None)
    ordered_features = feature_order_df.iloc[:, 0].values
    
    print(f"Number of ordered features: {len(ordered_features)}")
    
    return ordered_features


def create_feature_matrix(instance_names, transcriptomics_indexed, ordered_features):
    """
    Create feature matrix with instances as rows and features as columns.
    Features are ordered according to column.csv.
    """
    print("Creating feature matrix...")
    
    # Get available feature columns (skip dataset info column - assumed column 1)
    gene_features = transcriptomics_indexed.columns

    # Filter to only features that exist in transcriptomics data
    final_features = [f for f in ordered_features if f in gene_features]
    print(f"Columns entries found among genes: {len(final_features)} out of {len(ordered_features)}")
    
    # Find instances that exist in transcriptomics data
    available_instances = [inst for inst in instance_names 
                          if inst in transcriptomics_indexed.index]
    missing_instances = [inst for inst in instance_names 
                        if inst not in transcriptomics_indexed.index]
    
    # print(f"Instances found: {len(available_instances)} out of {len(instance_names)}")
    if missing_instances:
        print(f"Missing instances: {len(missing_instances)}")
        exit()
    else:
        print("All cell lines (instances) in splits are in transcriptomics")
    
    # Create feature matrix
    feature_matrix = np.zeros((len(available_instances), len(final_features)))
    
    for i, instance in enumerate(available_instances):
        instance_data = transcriptomics_indexed.loc[instance]
        for j, feature in enumerate(final_features):
            feature_matrix[i, j] = instance_data[feature]
    
    print(f"Feature matrix shape: {feature_matrix.shape}")
    return feature_matrix, np.array(available_instances), np.array(final_features)

def apply_standard_scaling(X, scaler=None, fit_scaler=True):
    """Apply standard scaling to feature matrix."""
    print("Applying standard scaling...")
    
    print(f"Before scaling - Mean: {X.mean():.4f}, Std: {X.std():.4f}")
    
    if scaler is None:
        scaler = StandardScaler()
    
    if fit_scaler:
        X_scaled = scaler.fit_transform(X)
        print("Fitted new scaler on training data")
    else:
        X_scaled = scaler.transform(X)
        print("Applied existing scaler")
    
    print(f"After scaling - Mean: {X_scaled.mean():.4f}, Std: {X_scaled.std():.4f}")
    
    return X_scaled, scaler


def process_dataset(dataset_name, file_path, transcriptomics_indexed, ordered_features, 
                   scaler=None, fit_scaler=True):
    """
    Process a complete dataset (train, test, or val).
    Returns scaled feature matrix and corresponding labels.
    """
    print(f"\n=== PROCESSING {dataset_name.upper()} DATASET ===")
    
    # File names
    X_file = file_path + f"{dataset_name}_X.csv"
    Y_file = file_path + f"{dataset_name}_Y.csv"
    
    # Load data
    instance_names = load_instance_names(X_file)
    labels = load_labels_predictions(Y_file)
    
    # Create feature matrix
    X, available_instances, final_features = create_feature_matrix(
        instance_names, transcriptomics_indexed, ordered_features
    )
    
    # Filter labels to match available instances (in case some instances in the splits data were not in the transcriptomics)
    instance_to_idx = {inst: i for i, inst in enumerate(instance_names)}
    available_indices = [instance_to_idx[inst] for inst in available_instances 
                        if inst in instance_to_idx]
    y = labels[available_indices]
    
    # Apply scaling
    X_scaled, scaler = apply_standard_scaling(X, scaler, fit_scaler)
    
    print(f"Final {dataset_name} dataset - X: {X_scaled.shape}, y: {y.shape}")
    
    return {
        'X': X_scaled,
        'y': y,
        'instances': available_instances,
        'features': final_features,
        'scaler': scaler
    }


def load_top_genes(top_genes_file):
    """
    Load list of top genes to use as filter.
    File format:
    ,0
    TNFRSF12A,228.18973541259766
    BCL2,126.99969863891602
    ...
    
    Returns numpy array of gene names (first column, skip header).
    """
    print(f"Loading top genes from {top_genes_file}...")
    
    # Skip first row (,0 header), no header, get first column (gene names)
    df = pd.read_csv(top_genes_file, skiprows=1, header=None)
    top_genes = df.iloc[:, 0].values  # First column contains gene names
    
    print(f"Loaded {len(top_genes)} top genes")
    print(f"First few genes: {top_genes[:3]}")
    
    return top_genes

def filter_transcriptomics_by_genes(transcriptomics_indexed, top_genes):
    """
    Filter transcriptomics data to only include specified genes.
    
    Args:
        transcriptomics_indexed: DataFrame with transcriptomics data
        top_genes: Array of gene names to keep
    
    Returns:
        Filtered transcriptomics DataFrame
    """
    print("Filtering transcriptomics data by top genes...")
    
    # Get current feature columns  
    current_features = transcriptomics_indexed.columns 
    print(f"Transcriptomics columns: {len(current_features)}")
    
    # Find which top genes are available in the data for the specific drug
    available_top_genes = [gene for gene in top_genes if gene in current_features]
    missing_top_genes = [gene for gene in top_genes if gene not in current_features]
    
    print(f"Top genes found: {len(available_top_genes)} out of {len(top_genes)}")
    if missing_top_genes:
        print(f"There are genes in the feature columns which are not in the transcriptomics data: {len(missing_top_genes)}")
        print(f"First few missing: {missing_top_genes[:3]}")
        exit()
    
    # Create filtered dataset: keep instance column + dataset column + selected genes
    columns_to_keep = available_top_genes # [transcriptomics_indexed.columns[0]] + available_top_genes  # dataset column + genes
    transcriptomics_filtered = transcriptomics_indexed[columns_to_keep]
    
    print(f"Filtered transcriptomics shape: {transcriptomics_filtered.shape}")
    print(f"Original shape was: {transcriptomics_indexed.shape}")
    
    return transcriptomics_filtered

