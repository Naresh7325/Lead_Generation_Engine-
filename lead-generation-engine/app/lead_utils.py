import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def clean_data(df):
    df.columns = df.columns.str.strip().str.lower()
    df.dropna(inplace=True)
    return df

def feature_engineer(df):
    df['total_shipments'] = df.groupby('company_name')['shipment_id'].transform('count')
    df['unique_suppliers'] = df.groupby('company_name')['supplier_name'].transform('nunique')
    return df.drop_duplicates(subset='company_name')

def score_leads(df):
    # Basic formula including service quality
    df['lead_score'] = (
        df['total_shipments'] / (df['unique_suppliers'] + 1)
    ) * (df['services_rating'] / 5)

    # Normalize lead_score to 0–100
    scaler = MinMaxScaler()
    df['lead_score'] = scaler.fit_transform(df[['lead_score']]) * 100

    # Tag good leads
    df['is_good_lead'] = df['lead_score'] > df['lead_score'].median()
    
    return df
