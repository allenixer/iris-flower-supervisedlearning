"""
Iris Flower Classification using Machine Learning
This project demonstrates supervised learning using the famous Iris dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def load_and_explore_data():
    """Load the Iris dataset and perform initial exploration"""
    print("=" * 60)
    print("LOADING IRIS DATASET")
    print("=" * 60)
    
    # Load dataset
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    
    print("\nDataset Shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nDataset Info:")
    print(df.describe())
    print("\nClass Distribution:")
    print(df['species_name'].value_counts())
    
    return df, iris

def visualize_data(df):
    """Create visualizations to understand the data"""
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)
    
    # Pairplot
    plt.figure(figsize=(12, 10))
    sns.pairplot(df, hue='species_name', markers=['o', 's', 'D'])
    plt.suptitle('Iris Dataset - Pairplot', y=1.02)
    plt.tight_layout()
    plt.savefig('iris_pairplot.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: iris_pairplot.png")
    
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    correlation = df.iloc[:, :-2].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: correlation_heatmap.png")
    
    # Box plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    features = df.columns[:-2]
    for idx, feature in enumerate(features):
        row, col = idx // 2, idx % 2
        sns.boxplot(data=df, x='species_name', y=feature, ax=axes[row, col])
        axes[row, col].set_title(f'{feature} by Species')
    plt.tight_layout()
    plt.savefig('feature_boxplots.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: feature_boxplots.png")

def train_models(X_train, X_test, y_train, y_test):
    """Train multiple models and compare their performance"""
    print("\n" + "=" * 60)
    print("TRAINING MODELS")
    print("=" * 60)
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=200, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {'model': model, 'accuracy': accuracy, 'predictions': y_pred}
        print(f"✓ {name} Accuracy: {accuracy:.4f}")
    
    return results

def evaluate_best_model(results, X_test, y_test, target_names):
    """Evaluate the best performing model in detail"""
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    # Find best model
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best_model = results[best_model_name]['model']
    best_predictions = results[best_model_name]['predictions']
    
    print(f"\nBest Model: {best_model_name}")
    print(f"Accuracy: {results[best_model_name]['accuracy']:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, best_predictions, target_names=target_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, best_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: confusion_matrix.png")
    
    # Model comparison
    accuracies = {name: data['accuracy'] for name, data in results.items()}
    plt.figure(figsize=(10, 6))
    bars = plt.bar(accuracies.keys(), accuracies.values(), color=['#2ecc71', '#3498db', '#e74c3c'])
    plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0.8, 1.0)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=11)
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: model_comparison.png")
    
    return best_model_name, best_model

def main():
    """Main execution function"""
    print("\n" + "=" * 60)
    print("IRIS FLOWER CLASSIFICATION PROJECT")
    print("=" * 60)
    
    # Load and explore data
    df, iris = load_and_explore_data()
    
    # Visualize data
    visualize_data(df)
    
    # Prepare data
    print("\n" + "=" * 60)
    print("PREPARING DATA")
    print("=" * 60)
    X = iris.data
    y = iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print("✓ Features scaled")
    
    # Train models
    results = train_models(X_train, X_test, y_train, y_test)
    
    # Evaluate best model
    best_model_name, best_model = evaluate_best_model(
        results, X_test, y_test, iris.target_names
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("PROJECT SUMMARY")
    print("=" * 60)
    print(f"\n✓ Dataset: Iris Flower Dataset (150 samples, 4 features, 3 classes)")
    print(f"✓ Best Model: {best_model_name}")
    print(f"✓ Best Accuracy: {results[best_model_name]['accuracy']:.4f}")
    print(f"✓ Visualizations saved: 4 images")
    print("\nProject completed successfully!")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
