from database_connector import get_bank_table
import pandas as pd
import numpy as np
import joblib
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def process_hard_data(filename="", train=True):
    """
    Gets hard data pipeline
    TODO

    Returns:
        X, y
    """
    categorical_cols = ["Occupation", "Marital Status", "Education Level", "Housing Loan", "Personal Loan", "Contact Method"]
    numeric = ['Age', 'Campaign Calls', 'Previous Contact Days']
    preferred_column_order = ['Occupation_blue-collar', 'Occupation_retired',
       'Occupation_self-employed', 'Occupation_student',
       'Occupation_unemployed', 'Occupation_unknown',
       'Occupation_white-collar', 'Marital Status_divorced',
       'Marital Status_married', 'Marital Status_single',
       'Marital Status_unknown', 'Education Level_high.school.ab',
       'Education Level_professional.course',
       'Education Level_university.degree', 'Education Level_unknown',
       'Housing Loan_yes', 'Housing Loan_no', 'Housing Loan_unknown',
       'Personal Loan_yes', 'Personal Loan_no', 'Personal Loan_unknown',
       'Contact Method_cellular', 'Contact Method_telephone', 'Age',
       'Campaign Calls', 'Previous Contact Days',
       'Contact Days Mask', 'Subscription Status']

    encoder = OneHotEncoder(categories=[
        ['blue-collar', 'retired', 'self-employed', 'student', 'unemployed', 'unknown', 'white-collar'], # for Occupation
        ['divorced', 'married', 'single', 'unknown'], # for Marital Status
        ['high.school.ab', 'professional.course', 'university.degree', 'unknown'], # for 'Education level'
        ['yes', 'no', 'unknown'],      # for Housing Loan
        ['yes', 'no', 'unknown'],      # for Personal Loan
        ['cellular', 'telephone'],      # for Contact Method
    ], handle_unknown='ignore')
    
    if (train):
        bank_df = get_bank_table()
        bank_df_2 = bank_df.copy()

        bank_df_2["Age"] = bank_df_2["Age"].str.extract(r'(\d+)').astype(int)
        bank_df_2.loc[bank_df_2["Age"] > 100, 'Age'] = np.nan       
        bank_df_2.drop('Client ID', axis=1, inplace=True)

        bank_df_2.loc[bank_df_2["Housing Loan"].isna(), "Housing Loan"] = 'unknown'
        bank_df_2.loc[bank_df_2["Personal Loan"].isna(), "Personal Loan"] = 'unknown'
        bank_df_2.loc[bank_df_2["Contact Method"] == 'Cell', "Contact Method"] = 'cellular'
        bank_df_2.loc[bank_df_2["Contact Method"] == 'Telephone', "Contact Method"] = 'telephone'
        bank_df_2["Campaign Calls"] = bank_df_2["Campaign Calls"].abs()

        bank_df_2.drop("Subscription Status", axis=1, inplace=True)
        bank_df_2.drop("Credit Default", axis=1, inplace=True)
        bank_df_2.loc[bank_df_2['Occupation'].isin(['blue-collar', 'housemaid', 'technician']), 'Occupation'] = 'blue-collar' 
        bank_df_2.loc[bank_df_2['Occupation'].isin(['admin.', 'management', 'services']), 'Occupation'] = 'white-collar'
        bank_df_2.loc[bank_df_2['Occupation'].isin(['entrepreneur', 'self-employed']), 'Occupation'] = 'self-employed'
        bank_df_2.loc[bank_df_2['Education Level'].isin(['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate']), 'Education Level'] = 'high.school.ab'                

        ct = ColumnTransformer([
            ('cat', encoder, categorical_cols)
        ], remainder='passthrough')

        imputed_array = ct.fit_transform(bank_df_2)
        encoded_feature_names = ct.named_transformers_['cat'].get_feature_names_out(categorical_cols)
        all_column_names = list(encoded_feature_names) + numeric  # re-add numeric columns

        # Convert back to DataFrame
        df_encoded = pd.DataFrame(imputed_array, columns=all_column_names, index=bank_df_2.index)        

        # Masking column for Previous contact days
        df_encoded.loc[df_encoded["Previous Contact Days"] == 999, "Contact Days Mask"] = 1
        df_encoded.loc[df_encoded["Previous Contact Days"] != 999, "Contact Days Mask"] = 0
        df_encoded.loc[df_encoded["Previous Contact Days"] == 999, "Previous Contact Days"] = 0

        imputer = KNNImputer(n_neighbors=3)
        age_imputed_array = imputer.fit_transform(df_encoded)
        df_encoded = pd.DataFrame(age_imputed_array, columns=df_encoded.columns, index=df_encoded.index)

        df_encoded["Subscription Status"] = bank_df["Subscription Status"]
        df_encoded.loc[df_encoded["Subscription Status"] == 'yes', 'Subscription Status'] = 1
        df_encoded.loc[df_encoded["Subscription Status"] == 'no', 'Subscription Status'] = 0
        df_encoded["Subscription Status"] = df_encoded["Subscription Status"].astype(int)
                
        df_encoded = df_encoded.reindex(columns=preferred_column_order)
        y = df_encoded["Subscription Status"].copy()
        X = df_encoded.drop(columns="Subscription Status")

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.20,
            stratify=y,   
            random_state=0
        )                
                
        return X_train, X_val, y_train, y_val
    else:
        # Evaluation mode
        if len(filename) == 0:
            raise ValueError("Need a filename for the pipeline")

        bank_df = pd.read_csv(filename)
        
        bank_df_2 = bank_df.copy()

        bank_df_2["Age"] = bank_df_2["Age"].str.extract(r'(\d+)').astype(int)
        bank_df_2.loc[bank_df_2["Age"] > 100, 'Age'] = np.nan       
        bank_df_2.drop('Client ID', axis=1, inplace=True)

        bank_df_2.loc[bank_df_2["Housing Loan"].isna(), "Housing Loan"] = 'unknown'
        bank_df_2.loc[bank_df_2["Personal Loan"].isna(), "Personal Loan"] = 'unknown'
        bank_df_2.loc[bank_df_2["Contact Method"] == 'Cell', "Contact Method"] = 'cellular'
        bank_df_2.loc[bank_df_2["Contact Method"] == 'Telephone', "Contact Method"] = 'telephone'
        bank_df_2["Campaign Calls"] = bank_df_2["Campaign Calls"].abs()

        bank_df_2.drop("Subscription Status", axis=1, inplace=True)
        bank_df_2.drop("Credit Default", axis=1, inplace=True)
        bank_df_2.loc[bank_df_2['Occupation'].isin(['blue-collar', 'housemaid', 'technician']), 'Occupation'] = 'blue-collar' 
        bank_df_2.loc[bank_df_2['Occupation'].isin(['admin.', 'management', 'services']), 'Occupation'] = 'white-collar'
        bank_df_2.loc[bank_df_2['Occupation'].isin(['entrepreneur', 'self-employed']), 'Occupation'] = 'self-employed'
        bank_df_2.loc[bank_df_2['Education Level'].isin(['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate']), 'Education Level'] = 'high.school.ab'                

        ct = ColumnTransformer([
            ('cat', encoder, categorical_cols)
        ], remainder='passthrough')

        imputed_array = ct.fit_transform(bank_df_2)
        encoded_feature_names = ct.named_transformers_['cat'].get_feature_names_out(categorical_cols)
        all_column_names = list(encoded_feature_names) + numeric  # re-add numeric columns

        # Convert back to DataFrame
        df_encoded = pd.DataFrame(imputed_array, columns=all_column_names, index=bank_df_2.index)        

        # Masking column for Previous contact days
        df_encoded.loc[df_encoded["Previous Contact Days"] == 999, "Contact Days Mask"] = 1
        df_encoded.loc[df_encoded["Previous Contact Days"] != 999, "Contact Days Mask"] = 0
        df_encoded.loc[df_encoded["Previous Contact Days"] == 999, "Previous Contact Days"] = 0

        imputer = KNNImputer(n_neighbors=3)
        age_imputed_array = imputer.fit_transform(df_encoded)
        df_encoded = pd.DataFrame(age_imputed_array, columns=df_encoded.columns, index=df_encoded.index)

        df_encoded["Subscription Status"] = bank_df["Subscription Status"]
        df_encoded.loc[df_encoded["Subscription Status"] == 'yes', 'Subscription Status'] = 1
        df_encoded.loc[df_encoded["Subscription Status"] == 'no', 'Subscription Status'] = 0
        df_encoded["Subscription Status"] = df_encoded["Subscription Status"].astype(int)
                
        df_encoded = df_encoded.reindex(columns=preferred_column_order)
        y = df_encoded["Subscription Status"].copy()
        X = df_encoded.drop(columns="Subscription Status")

        return X, y

    
