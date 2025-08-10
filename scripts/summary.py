import pandas as pd
segmented_customers = pd.read_csv("/Users/dev/python1_cogitate/outputs/final.csv")

def generate_summary(segmented_customers):
    """Generate a summary of customer segments and high-risk customers."""
    # Segment counts
    segment_counts = segmented_customers['segment'].value_counts()
    
    # Highest risk customers (top 3)
    high_risk = segmented_customers.nlargest(3, 'risk_score')
    
    print("\n=== CUSTOMER SEGMENTATION SUMMARY ===")
    print("\nNumber of customers in each segment:")
    for segment, count in segment_counts.items():
        print(f"- {segment}: {count} customers")
    
    print("\nHighest-risk customers (immediate review recommended):")
    for _, row in high_risk.iterrows():
        print(f"\nCustomer ID: {row['customer_id']}")
        print(f"Risk Score: {row['risk_score']:.1f}")
        print(f"Segment: {row['segment']}")
        print(f"Lifetime Value: ${row['lifetime_value']:,.2f}")
        print(f"Loss Ratio: {row['loss_ratio']*100 if pd.notnull(row['loss_ratio']) else 0:.1f}%")
        print("Recommended Action: Manual review required - consider policy adjustment or investigation")
    
    print("\n=== END OF SUMMARY ===\n")

# Add this to your main function after segmentation is done
generate_summary(segmented_customers)