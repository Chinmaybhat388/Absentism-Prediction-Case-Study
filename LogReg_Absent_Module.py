#!/usr/bin/env python
# coding: utf-8

# In[5]:


#Import the needed libraries.
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

#The custom scaler that only scales the non-dummy value columns.
class MyScaler(BaseEstimator,TransformerMixin):
    def __init__(self,columns,with_mean=True,with_std=True,copy=True):
        self.scaler = StandardScaler(copy,with_mean,with_std)
        self.columns = columns
        self.mean_ = None
        self.var_ = None
    
    def fit(self,X,y=None):
        self.scaler.fit(X[self.columns],y)
        self.mean_ = np.array(np.mean(X[self.columns]))
        self.var_ = np.array(np.var(X[self.columns]))
        return self
    
    def transform(self,X,y=None,copy=None):
        initial_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]),columns=self.columns)
        X_not_scaled = X.loc[:,~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled,X_scaled],axis=1)[initial_col_order]
    
#The class that we are going to use from here on to predict new data
class absenteeism_model():
    def __init__(self,model_file,scaler_file):
        with open('Absenteeism_Model','rb') as model_file,open('Custom_Scaler','rb') as scaler_file:
            self.log_reg = pickle.load(model_file)             #Load the previously saved model 
            self.scaler = pickle.load(scaler_file)             #and scaler.
            self.data = None
            
#take a data file (*.csv) and preprocess it.
    def load_and_clean_data(self, data_file):
        absent_df = pd.read_csv(data_file,delimiter=',')
        self.df = absent_df.copy()
        
        #Drop the 'ID' column
        absent_df.drop(['ID'],axis=1,inplace=True)
        
        # to preserve the code we've created in the previous section, we will add a column with 'NaN' strings
        absent_df['Absenteeism Time in Hours'] = 'NaN'
        
        #Replace the reason for absence with one of 4 groups.
        absent_df['Reason for Absence'] = absent_df['Reason for Absence'].replace(to_replace = list(range(1,15)),value=1)
        absent_df['Reason for Absence'] = absent_df['Reason for Absence'].replace(to_replace = list(range(15,18)),value=2)
        absent_df['Reason for Absence'] = absent_df['Reason for Absence'].replace(to_replace = list(range(18,22)),value=3)
        absent_df['Reason for Absence'] = absent_df['Reason for Absence'].replace(to_replace = list(range(22,29)),value=4)
        
        #Dummy encode the feature 'Reason for Absence'
        dummies = pd.get_dummies(absent_df['Reason for Absence'],drop_first=True,prefix='Reason',prefix_sep='_')
        
        #If there are missing reasons for absence
        col_val = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4']
        for i in (set(col_val) - set(dummies.columns)):
            dummies[i] = 0
        absent_df = pd.concat([absent_df,dummies],axis=1)
        
        #Drop the original column 'Reason for Absence'
        absent_df.drop(['Reason for Absence'],axis=1,inplace = True)
        
        #From date, extract day of the week and month and drop the original date column.
        absent_df['Date'] = pd.to_datetime(absent_df['Date'], format="%d/%m/%Y")
        absent_df['Day of the Week'] = absent_df['Date'].dt.dayofweek
        absent_df['month'] = pd.DatetimeIndex(absent_df['Date']).month
        absent_df.drop(['Date'],axis=1,inplace=True)
        
        # Reorder the columns.
        col_order = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4','month','Day of the Week', 'Transportation Expense',                      'Distance to Work', 'Age', 'Daily Work Load Average', 'Body Mass Index', 'Education', 'Children',                     'Pets', 'Absenteeism Time in Hours']
        absent_df = absent_df[col_order]
        
        #Turn the data from the ‘Education’ column into binary data
        absent_df['Education'] = absent_df['Education'].replace(to_replace = [1], value = 0)
        absent_df['Education'] = absent_df['Education'].replace(to_replace = [2,3,4], value = 1)
        
        #Replace the NaN values
        absent_df = absent_df.fillna(value=0)
        
        #Drop the original absenteeism time
        absent_df = absent_df.drop(['Absenteeism Time in Hours'],axis=1)
            
        #Drop the variables whose coefficients were close to 0 to simplify the model.
        #absent_df = absent_df.drop(['Day of the Week','Daily Work Load Average','Distance to Work'],axis=1)
        
        #Included this line of code if we have to call the 'preprocessed data'
        self.preprocessed_data = absent_df.copy()
        
        self.data = self.scaler.transform(absent_df)
        
        #Function which outputs the probability of a data point to be 1
    def predicted_probability(self):
        if (self.data is not None):
            pred = self.log_reg.predict_proba(self.data)[:,1]
            return pred
        
        #Function which outputs 0 or 1 based on our model
    def predicted_output_category(self):
        if (self.data is not None):
            predicted_class = self.log_reg.predict(self.data)
            return predicted_class
            
        #Predict the outputs and the probabilities and add columns with these values at the end of the new data
    def predicted_outputs(self):
        if (self.data is not None):
            self.preprocessed_data['Probability'] = self.log_reg.predict_proba(self.data)[:,1]
            self.preprocessed_data['Prediction'] = self.log_reg.predict(self.data)
            return self.preprocessed_data
                
            


# In[ ]:




