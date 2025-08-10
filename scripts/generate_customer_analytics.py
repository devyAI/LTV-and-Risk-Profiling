import pandas as pd
import numpy as np
from datetime import datetime

def load_data():
    """Load all required datasets."""
    print("Loading data files...")
    customers = pd.read_csv('../data/customers_sample.csv')
    policies = pd.read_csv('../data/policies_sample.csv')
    claims = pd.read_csv('../data/claims_sample.csv')
    fraud = pd.read_csv('../data/fraud_detection_sample.csv')
    return customers, policies, claims, fraud

def preprocess_data(customers, policies, claims, fraud):
    """Preprocess and clean the data."""
    print("Preprocessing data...")
    # Convert dates to datetime
    customers['registration_date'] = pd.to_datetime(customers['registration_date'])
    policies['start_date'] = pd.to_datetime(policies['start_date'])
    claims['claim_date'] = pd.to_datetime(claims['claim_date'])
    fraud['detection_date'] = pd.to_datetime(fraud['detection_date'])
    
    # Calculate customer tenure in days
    current_date = pd.to_datetime('today')
    customers['customer_tenure_days'] = (current_date - customers['registration_date']).dt.days
    
    # Calculate policy tenure (days since first policy)
    first_policy = policies.groupby('customer_id')['start_date'].min().reset_index()
    first_policy['policy_tenure_days'] = (current_date - first_policy['start_date']).dt.days
    first_policy = first_policy[['customer_id', 'policy_tenure_days']]
    
    return customers, policies, claims, fraud, first_policy

def create_customer_analytics(customers, policies, claims, fraud, first_policy):
    """Create customer analytics by merging all datasets."""
    print("Creating customer analytics...")
    
    # 1. Policy metrics by customer
    policy_metrics = policies.groupby('customer_id').agg(
        total_policies=('policy_id', 'count'),
        active_policies=('status', lambda x: (x == 'ACTIVE').sum()),
        avg_annual_premium=('annual_premium', 'mean'),
        total_coverage=('coverage_amount', 'sum'),
        policy_types=('policy_type', lambda x: x.value_counts().to_dict()),
        policy_ids=('policy_id', list)
    ).reset_index()
    
    # 2. Claims metrics by policy
    claims_metrics = claims.groupby('policy_id').agg(
        total_claims=('claim_id', 'count'),
        total_claimed_amount=('claim_amount', 'sum'),
        avg_claim_amount=('claim_amount', 'mean'),
        claim_statuses=('status', lambda x: x.value_counts().to_dict())
    ).reset_index()
    
    # 3. Fraud metrics by claim
    fraud_metrics = fraud.groupby('claim_id').agg(
        fraud_detected=('is_fraudulent', 'max'),
        detection_count=('is_fraudulent', 'sum'),
        avg_confidence=('confidence_score', 'mean')
    ).reset_index()
    
    # 4. Merge all datasets
    # 4.1 Create policy-customer mapping
    policy_customer_map = policies[['policy_id', 'customer_id']].drop_duplicates()
    
    # 4.2 Merge claims metrics with policy-customer mapping
    policy_claims = pd.merge(
        policy_customer_map,
        claims_metrics,
        on='policy_id',
        how='left'
    )
    
    # 4.3 Aggregate claims at customer level
    customer_claims = policy_claims.groupby('customer_id').agg({
        'total_claims': 'sum',
        'total_claimed_amount': 'sum',
        'avg_claim_amount': 'mean'
    }).reset_index()
    
    # 4.4 Merge with fraud data
    claims_fraud = pd.merge(
        claims[['claim_id', 'policy_id']],
        fraud_metrics,
        on='claim_id',
        how='left'
    )
    
    # 4.5 Aggregate fraud metrics at customer level
    policy_fraud = pd.merge(
        claims_fraud,
        policy_customer_map,
        on='policy_id',
        how='left'
    )
    
    customer_fraud_metrics = policy_fraud.groupby('customer_id').agg({
        'fraud_detected': 'max',
        'detection_count': 'sum',
        'avg_confidence': 'mean'
    }).reset_index()
    
    # 5. Merge all metrics
    # 5.1 Merge policy metrics with claims metrics
    customer_metrics = pd.merge(
        policy_metrics,
        customer_claims,
        on='customer_id',
        how='left'
    )
    
    # 5.2 Merge with fraud metrics
    customer_metrics = pd.merge(
        customer_metrics,
        customer_fraud_metrics,
        on='customer_id',
        how='left'
    )
    
    # 5.3 Add policy tenure
    customer_metrics = pd.merge(
        customer_metrics,
        first_policy,
        on='customer_id',
        how='left'
    )
    
    # 5.4 Merge with customer data
    customer_analytics = pd.merge(
        customers,
        customer_metrics,
        on='customer_id',
        how='left'
    )
    
    # 6. Fill NaN values
    numeric_cols = customer_analytics.select_dtypes(include=[np.number]).columns
    customer_analytics[numeric_cols] = customer_analytics[numeric_cols].fillna(0)
    
    # 7. Calculate derived metrics
    customer_analytics['claims_per_policy'] = customer_analytics['total_claims'] / customer_analytics['total_policies'].replace(0, np.nan)
    customer_analytics['fraud_rate'] = customer_analytics['detection_count'] / customer_analytics['total_claims'].replace(0, np.nan)
    
    # 8. Select and order columns
    final_columns = [
        'customer_id', 'name', 'age', 'email', 'city', 'registration_date',
        'customer_tenure_days', 'policy_tenure_days',
        'total_policies', 'active_policies', 'policy_types',
        'total_claims', 'total_claimed_amount', 'avg_claim_amount',
        'fraud_detected', 'detection_count', 'avg_confidence',
        'claims_per_policy', 'fraud_rate',
        'avg_annual_premium', 'total_coverage', 'policy_ids'
    ]
    
    return customer_analytics[final_columns]

def main():
    # Load and preprocess data
    customers, policies, claims, fraud, first_policy = preprocess_data(*load_data())
    
    # Create customer analytics
    customer_analytics = create_customer_analytics(customers, policies, claims, fraud, first_policy)
    
    # Save to CSV
    output_path = '../outputs/customer_analytics_enhanced.csv'
    customer_analytics.to_csv(output_path, index=False)
    print(f"\nEnhanced customer analytics saved to: {output_path}")
    print(f"Total customers processed: {len(customer_analytics)}")

if __name__ == "__main__":
    main()
