'''
Generates a reusable sample payload/input.
The sample payload is a pytest fixture so that any test can ask for it.
This keeps each test clean since there is no need to retype the sample input JSON every time in each test.
'''


import pytest

@pytest.fixture
def sample_payload():
    return {
        "customer_id": "CUST_12345",
        "company_size": "Small",
        "subscription_plan": "Basic",
        "months_active": 5,
        "monthly_revenue": 29,
        "days_since_last_login": 9,
        "monthly_logins": 6,
        "features_used": 3,
        "support_tickets": 1,
        "satisfaction_score": 2.8,
        "payment_failures": 0,
    }