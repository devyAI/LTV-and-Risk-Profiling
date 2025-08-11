import pandas as pd

def analyze_customers(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Count customers in each segment
    segment_counts = df['segment'].value_counts()
    
    # Find highest-risk customers (lowest lifetime value and highest loss ratio)
    high_risk = df.sort_values(by=['lifetime_value', 'loss_ratio'], 
                              ascending=[True, False])
    
    # Get the top 3 highest risk customers
    top_risks = high_risk.head(3)
    
    # Print the summary
    print("CUSTOMER SEGMENT ANALYSIS")
    print("-" * 80)
    print("\nCUSTOMER SEGMENT DISTRIBUTION:")
    for segment, count in segment_counts.items():
        print(f"- {segment}: {count} customers")
    
    print("\nHIGHEST-RISK CUSTOMERS:")
    for _, row in top_risks.iterrows():
        print(f"\nCustomer ID: {row['customer_id']}")
        print(f"Segment: {row['segment']}")
        print(f"Lifetime Value: ${row['lifetime_value']:,.2f}")
        if pd.notna(row['loss_ratio']):
            print(f"Loss Ratio: {row['loss_ratio']:.2f}%")
    
    print("\nRECOMMENDED NEXT STEPS:")
    print("1. Immediate manual review of the top 3 highest-risk customers")
    print("2. Investigate customers with missing loss ratio values")
    print("3. Consider risk mitigation strategies for 'Risk Management' segment")
    print("4. Review underwriting criteria for 'Watch List' segment")

if __name__ == "__main__":
    analyze_customers("outputs/final_output.csv")
