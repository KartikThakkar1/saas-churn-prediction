# generate_dataset.py
# Section 1: Generate SaaS Customer Churn Dataset

import pandas as pd
import numpy as np
import random
import os 

# Set random seed so we get same data every time
np.random.seed(42)
random.seed(42)

def create_saas_dataset(n_customers=3000):
    """
    Create realistic SaaS customer data for churn prediction.
    
    Features we'll create:
    - Customer demographics and subscription info
    - Usage patterns (logins, features used)  
    - Support interactions
    - Payment/billing data
    - Churn label (what we want to predict)
    """
    
    print(f"Creating dataset with {n_customers} customers...")
    
    customers = []
    
    for i in range(n_customers):
        # Basic info
        customer_id = f"CUST_{i+1:05d}"
        
        # Company characteristics
        company_size = np.random.choice(['Small', 'Medium', 'Large'], p=[0.5, 0.35, 0.15])
        plan = np.random.choice(['Basic', 'Pro', 'Enterprise'], p=[0.6, 0.3, 0.1])
        
        # How long they've been a customer (months)
        months_active = max(1, int(np.random.exponential(8)))
        
        # Monthly revenue based on plan
        if plan == 'Basic':
            monthly_revenue = 29
        elif plan == 'Pro':
            monthly_revenue = 99
        else:  # Enterprise
            monthly_revenue = 299
            
        # Usage metrics - these are KEY for predicting churn!
        days_since_last_login = int(np.random.exponential(5))  # How recently they logged in
        monthly_logins = max(0, int(np.random.gamma(3, 3)))     # How often they login
        features_used = max(1, min(10, int(np.random.gamma(2, 2))))  # How many features they use
        
        # Support and satisfaction
        support_tickets = int(np.random.poisson(1))  # Support requests
        satisfaction_score = round(np.random.normal(4.0, 1.0), 1)  # 1-5 rating
        satisfaction_score = max(1, min(5, satisfaction_score))
        
        # Payment info
        payment_failures = int(np.random.poisson(0.2))  # Failed payments
        
        # Now create the churn probability based on business logic
        churn_prob = 0.1  # Base 10% churn rate
        
        # Bad signs that increase churn risk:
        if days_since_last_login > 14:  # Haven't logged in recently
            churn_prob += 0.4
        elif days_since_last_login > 7:
            churn_prob += 0.2
            
        if monthly_logins < 5:  # Low activity
            churn_prob += 0.3
            
        if features_used < 3:  # Not using the product much
            churn_prob += 0.3
            
        if satisfaction_score < 3:  # Unhappy
            churn_prob += 0.4
            
        if payment_failures > 0:  # Payment issues
            churn_prob += 0.3
            
        if months_active < 3:  # New customers churn more
            churn_prob += 0.2
            
        # Good signs that reduce churn risk:
        if plan == 'Enterprise':  # Enterprise customers are stickier
            churn_prob -= 0.1
            
        if months_active > 12:  # Long-term customers are loyal
            churn_prob -= 0.1
            
        # Keep probability between 0 and 1
        churn_prob = max(0.02, min(0.8, churn_prob))
        
        # Actually decide if they churned
        churned = np.random.random() < churn_prob
        
        # Save customer data
        customer = {
            'customer_id': customer_id,
            'company_size': company_size,
            'subscription_plan': plan,
            'months_active': months_active,
            'monthly_revenue': monthly_revenue,
            'days_since_last_login': days_since_last_login,
            'monthly_logins': monthly_logins,
            'features_used': features_used,
            'support_tickets': support_tickets,
            'satisfaction_score': satisfaction_score,
            'payment_failures': payment_failures,
            'churned': churned  # This is what we want to predict!
        }
        
        customers.append(customer)
        
        # Show progress
        if (i + 1) % 500 == 0:
            print(f"Generated {i + 1} customers...")
    
    # Convert to DataFrame
    df = pd.DataFrame(customers)
    
    # Show some stats
    churn_rate = df['churned'].mean()
    print(f"\nDataset created!")
    print(f"Total customers: {len(df)}")
    print(f"Churn rate: {churn_rate:.1%}")
    print(f"Churned customers: {df['churned'].sum()}")
    print(f"Active customers: {(~df['churned']).sum()}")
    
    return df

if __name__ == "__main__":
    # Create necessary directories if they don't exist
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    print("✓ Created data directories")
    
    # Create the dataset
    dataset = create_saas_dataset(3000)
    
    # Save to CSV file in raw data folder
    dataset.to_csv('data/raw/saas_churn_data.csv', index=False)
    print("\nDataset saved as 'data/raw/saas_churn_data.csv'")
    
    # Show first few rows
    print("\nFirst 5 customers:")
    print(dataset.head())
    
    print("\nDataset is ready")