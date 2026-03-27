import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda(data_path="data/heart-disease.csv", output_dir="notebooks"):
    print(f"--- Running Exploratory Data Analysis (EDA) ---")
    
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        return
        
    df = pd.read_csv(data_path)
    
    print("\n[1] Dataset Shape, Column Names, Data Types:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nData Types:")
    print(df.dtypes)
    
    print("\n[2] Checking Missing Values:")
    print(df.isnull().sum())
    
    print("\n[3] Class Distribution (0 = No Disease, 1 = Disease):")
    if 'target' in df.columns:
        print(df['target'].value_counts())
    else:
        print("Target column 'target' not found!")
        
    print("\n[4] Generating Plots...")
    
    # 1. Class Distribution Plot
    plt.figure(figsize=(6, 4))
    sns.countplot(x='target', data=df, palette='Set2')
    plt.title("Class Distribution (0 = No Disease, 1 = Disease)")
    plt.savefig(os.path.join(output_dir, "class_distribution.png"))
    plt.close()
    
    # 2. Correlation Heatmap
    plt.figure(figsize=(12, 10))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title("Correlation Heatmap")
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
    plt.close()
    
    # 3. Histograms for all variables
    df.hist(figsize=(15, 12), bins=20, color='skyblue', edgecolor='black')
    plt.suptitle("Feature Histograms", fontsize=16)
    plt.savefig(os.path.join(output_dir, "feature_histograms.png"))
    plt.close()
    
    # 4. Boxplots for outliers
    plt.figure(figsize=(15, 10))
    sns.boxplot(data=df, orient="h", palette="Set2")
    plt.title("Boxplots for Outlier Detection")
    plt.savefig(os.path.join(output_dir, "feature_boxplots.png"))
    plt.close()
    
    print(f"EDA Complete. Plots saved to '{output_dir}'.")

if __name__ == "__main__":
    run_eda()
