from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

import os 
import pandas as pd
import numpy as np

MODEL_FILE='model.pkl'
PIPELINE_FILE= 'pipeline.pkl'

def build_pipeline(num_attribute,cat_attribute):

    num_pipeline= Pipeline([
        ('imputer',SimpleImputer(strategy='median')), # impute missing value
        ('scaler',StandardScaler()) # scale values to common range
    ])

    cat_pipeline = Pipeline([
        ('onehot',OneHotEncoder(handle_unknown='ignore'))
    ])

    full_pipeline = ColumnTransformer([
        ('num',num_pipeline,num_attribute),
        ('cat',cat_pipeline,cat_attribute)
    ])

    return full_pipeline

if not os.path.exists(MODEL_FILE):

    # training process
    housing = pd.read_csv('housing.csv') # load dataset

    bins = np.linspace(housing['median_house_value'].min(),housing['median_house_value'].max(),11)
    housing['price_bins']= pd.cut(housing['median_house_value'],
                        bins=bins, include_lowest=True, labels=[1,2,3,4,5,6,7,8,9,10] )
    
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index,test_index in split.split(housing,housing['price_bins']):
        strat_train= housing.loc[train_index]
        strat_test= housing.loc[test_index]

    

    housing.drop('price_bins',axis=1,inplace=True)

    strat_train.drop('price_bins', axis=1,inplace=True)

    y_train= housing['median_house_value'].copy()
    housing_features = housing.drop('median_house_value',axis=1) # us

    num_att = housing_features.drop('ocean_proximity',axis=1).columns.tolist()
    cat_att= ['ocean_proximity']

    pipeline = build_pipeline(num_att,cat_att)
    x_train = pipeline.fit_transform(housing_features)   

    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(random_state=42)
    model.fit(x_train,y_train)

    #save the model and pipeline
    import joblib
    joblib.dump(model,MODEL_FILE)
    joblib.dump(pipeline,PIPELINE_FILE)

    print('model trained and saved...')

else:
    import joblib
    # inference phase
    model = joblib.load(MODEL_FILE)
    pipeline= joblib.load(PIPELINE_FILE)

    input_data= pd.read_csv('test.csv')
    transformed_data= pipeline.transform(input_data)
    predictions= model.predict(transformed_data)
    input_data['median_house_value']= predictions

    input_data.to_csv('test_output.csv',index=False)
    print('inference completed...')