"""
Exploratory Data Visualizations for Customer Analytics

This script creates visualizations to understand the distribution and relationships
in the customer data before performing detailed analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Configuration
DATA_PATH = '/Users/dev/python1_cogitate/outputs/customer_analytics_merged.csv'


def load_data():
    """Load and prepare the data for visualization."""
    print("Loading data...")
    df = pd.read_csv(DATA_PATH, parse_dates=['registration_date'])
    
    # Basic data cleaning
    numeric_cols = ['age', 'customer_tenure_days', 'total_policies', 'active_policies',
                   'avg_annual_premium', 'total_coverage', 'policy_tenure_days',
                   'total_claims', 'total_claimed_amount', 'avg_claim_amount',
                   'detection_count', 'avg_confidence', 'claims_per_policy', 'fraud_rate']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def plot_numeric_distributions(df):
    """Plot distributions of key numeric variables."""
    print("Plotting numeric distributions...")
    
    # Select numeric columns for visualization
    numeric_cols = ['age', 'customer_tenure_days', 'total_policies', 'active_policies',
                   'avg_annual_premium', 'total_claimed_amount', 'policy_tenure_days']
    
    # Filter to only include columns that exist in the dataframe
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    
    # Set up subplots
    n_cols = 2
    n_rows = (len(numeric_cols) + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        sns.histplot(df[col].dropna(), kde=True, ax=ax, bins=30)
        ax.set_title(f'Distribution of {col}')
        ax.set_xlabel('')
        
        # Rotate x-tick labels if needed
        if len(ax.get_xticklabels()) > 5:
            ax.tick_params(axis='x', rotation=45)
    
    # Remove empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()


def plot_categorical_distributions(df):
    """Plot distributions of categorical variables."""
    print("Plotting categorical distributions...")
    
    # Select categorical columns
    cat_cols = ['city']  # Add other categorical columns if available
    
    for col in cat_cols:
        if col in df.columns:
            plt.figure(figsize=(12, 6))
            value_counts = df[col].value_counts()
            
            # For columns with many categories, show top N
            if len(value_counts) > 10:
                top_n = value_counts.nlargest(10)
                sns.barplot(x=top_n.index, y=top_n.values)
                plt.xticks(rotation=45, ha='right')
                plt.title(f'Top 10 {col} (showing top 10 of {len(value_counts)} categories)')
            else:
                sns.countplot(data=df, x=col)
                plt.xticks(rotation=45, ha='right')
                plt.title(f'Distribution of {col}')
            
            plt.tight_layout()
            plt.show()


def plot_numeric_relationships(df):
    """Plot relationships between numeric variables."""
    print("Plotting numeric relationships...")
    
    # Select numeric columns for correlation
    numeric_cols = ['age', 'customer_tenure_days', 'total_policies', 'active_policies',
                   'avg_annual_premium', 'total_claimed_amount', 'policy_tenure_days',
                   'total_claims', 'fraud_rate']
    
    # Filter to only include columns that exist in the dataframe
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    
    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlation Heatmap of Numeric Variables')
    plt.tight_layout()
    plt.show()
    
    # Scatter plots for key relationships
    pairs = [
        ('age', 'avg_annual_premium'),
        ('customer_tenure_days', 'total_claimed_amount'),
        ('policy_tenure_days', 'total_claims'),
        ('avg_annual_premium', 'total_claimed_amount')
    ]
    
    for x, y in pairs:
        if x in df.columns and y in df.columns:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=df, x=x, y=y, alpha=0.6)
            plt.title(f'Relationship between {x} and {y}')
            plt.tight_layout()
            plt.show()


def plot_time_series(df):
    """Plot time series data if available."""
    print("Plotting time series data...")
    
    if 'registration_date' in df.columns:
        # Convert to datetime if not already
        df['registration_date'] = pd.to_datetime(df['registration_date'])
        
        # Group by date and count registrations
        time_series = df.set_index('registration_date').resample('M').size()
        
        plt.figure(figsize=(14, 6))
        time_series.plot()
        plt.title('Customer Registrations Over Time')
        plt.xlabel('Date')
        plt.ylabel('Number of Registrations')
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def main():
    """Main function to run all visualizations."""
    try:
        # Load the data
        df = load_data()
        
        # Generate visualizations
        plot_numeric_distributions(df)
        plot_categorical_distributions(df)
        plot_numeric_relationships(df)
        plot_time_series(df)
        
        print("\nExploratory visualization complete!")
        
    except Exception as e:
        print(f"Error during visualization: {str(e)}")
        raise


if __name__ == "__main__":
    main()
