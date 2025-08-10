"""
Customer Analytics and Segmentation

This script analyzes customer data to calculate key metrics and segment customers
based on risk and lifetime value.
"""

import pandas as pd
import numpy as np
from datetime import datetime

# Configuration
DATA_PATH = '/Users/dev/python1_cogitate/outputs/customer_analytics_merged.csv'
OUTPUT_PATH = '/Users/dev/python1_cogitate/outputs/customer_segments.csv'


def load_and_prepare_data(filepath):
    """Load and prepare the customer analytics data."""
    print("Loading data...")
    df = pd.read_csv(filepath, parse_dates=['registration_date'])
    
    # Data cleaning
    df['policy_tenure_days'] = pd.to_numeric(df['policy_tenure_days'], errors='coerce')
    df['fraud_detected'] = df['fraud_detected'].fillna(False)
    df['detection_count'] = df['detection_count'].fillna(0)
    
    return df


def calculate_metrics(df):
    """Calculate key customer metrics."""
    print("Calculating metrics...")
    
    # Create a copy to avoid SettingWithCopyWarning
    result = df.copy()
    
    # 1. Policy tenure (already in days)
    result['policy_tenure_days'] = result['policy_tenure_days'].fillna(0)
    
    # 2. Total claims and claim amount (already in the data)
    result['total_claims'] = result['total_claims'].fillna(0)
    result['total_claim_amount'] = result['total_claimed_amount'].fillna(0)
    
    # 3. Annual premium sum (sum of active policies * avg_annual_premium)
    result['annual_premium_sum'] = result['active_policies'] * result['avg_annual_premium']
    
    # 4. Fraud claims count
    result['fraud_claims'] = result['detection_count']
    
    return result


def calculate_lifetime_metrics(df):
    """Calculate lifetime value and loss ratio."""
    print("Calculating lifetime metrics...")
    
    result = df.copy()
    
    # Calculate lifetime value (annual_premium_sum - total_claim_amount)
    result['lifetime_value'] = result['annual_premium_sum'] - result['total_claim_amount']
    
    # Calculate loss ratio (total_claim_amount / annual_premium_sum)
    # Handle division by zero
    result['loss_ratio'] = np.where(
        result['annual_premium_sum'] > 0,
        result['total_claim_amount'] / result['annual_premium_sum'],
        np.nan  # or 0, depending on business logic
    )
    
    return result


def calculate_risk_score(df):
    """Calculate risk score based on multiple factors."""
    print("Calculating risk scores...")
    
    result = df.copy()
    
    # 1. Loss ratio component (0-50 points)
    loss_ratio_score = np.clip(result['loss_ratio'] * 50, 0, 50)
    
    # 2. Fraud claims component (0-30 points)
    # Scale fraud claims to 0-30 range (assuming max 10 fraud claims = 30 points)
    fraud_score = np.clip(result['fraud_claims'] * 3, 0, 30)
    
    # 3. Claim frequency component (0-20 points)
    # Claims per policy year (with min 1 day to avoid division by zero)
    claims_per_year = result['total_claims'] / (result['policy_tenure_days'].clip(lower=1) / 365.25)
    # Scale to 0-20 range (assuming 5 claims/year = 20 points)
    frequency_score = np.clip(claims_per_year * 4, 0, 20)
    
    # Sum components to get final risk score (0-100)
    result['risk_score'] = loss_ratio_score + fraud_score + frequency_score
    
    return result


def assign_segments(df):
    """Assign customers to segments based on LTV and risk score."""
    print("Assigning customer segments...")
    
    result = df.copy()
    
    # Define segment conditions
    conditions = [
        # Premium Partner (LTV ≥ 0 & risk ≤ 40)
        (result['lifetime_value'] >= 0) & (result['risk_score'] <= 40),
        # Growth Prospect (LTV ≥ 0 & 40 < risk ≤ 60)
        (result['lifetime_value'] >= 0) & (result['risk_score'] > 40) & (result['risk_score'] <= 60),
        # Risk Management (LTV < 0 & risk > 60)
        (result['lifetime_value'] < 0) & (result['risk_score'] > 60),
        # Watch List (All others)
        True  # Default
    ]
    
    # Define segment names
    segments = [
        'Premium Partner',
        'Growth Prospect',
        'Risk Management',
        'Watch List'
    ]
    
    # Assign segments
    result['segment'] = np.select(conditions, segments, default='Watch List')
    
    return result


def main():
    """Main function to run the analysis."""
    try:
        # 1. Load and prepare data
        df = load_and_prepare_data(DATA_PATH)
        
        # 2. Calculate basic metrics
        df = calculate_metrics(df)
        
        # 3. Calculate lifetime metrics
        df = calculate_lifetime_metrics(df)
        
        # 4. Calculate risk scores
        df = calculate_risk_score(df)
        
        # 5. Assign segments
        df = assign_segments(df)
        
        # 6. Select and save final output
        output_cols = [
            'customer_id', 'name', 'lifetime_value', 'loss_ratio', 
            'risk_score', 'segment', 'total_claims', 'total_claim_amount',
            'annual_premium_sum', 'fraud_claims', 'policy_tenure_days'
        ]
        
        result = df[output_cols].sort_values('lifetime_value', ascending=False)
        
        # Save results
        result.to_csv(OUTPUT_PATH, index=False)
        print(f"\nAnalysis complete! Results saved to: {OUTPUT_PATH}")
        
        # Print summary
        print("\nSegment distribution:")
        print(result['segment'].value_counts())
        
        return result
    
    except Exception as e:
        print(f"Error: {str(e)}")
        raise


if __name__ == "__main__":
    results = main()
    print("\nSample of results:")
    print(results.head())
