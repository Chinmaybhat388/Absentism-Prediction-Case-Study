# Absentism-Prediction-Case-Study
A case study on the employees' tendency to be excessively absent from work based on certain factors.

The data consists of the following independant variables.
ID,Reason for Absence,Date,Transportation Expense,Distance to Work,Age,Daily Work Load Average,Body Mass Index,Education,Children,Pets

We also have "Absenteeism Time in Hours" which records the number of hours the employee is absent from work.The aim of this project is to
identify employees who are excessively absent from work. We will consider the median value of "Absenteeism Time in Hours" and classify the 
employees as either excessively absent or not.

By examining the original data, we can merge some of the categories and reduce the number of categories in Reason for absence variable.
We can convert the feature "Reason for absence" from 28 categories to 4 categories.

- Class 1 to 14 is related to diseases and ilnesses.
- Class 15 to 17 is related to pregnancy and child birth.
- Class 18 to 21 is related to poisoning,injury etc.
- Class 22 to 28 is related to medical consultations and appointments.

We will also, Turn the data from the ‘Education’ column into binary data, by mapping the value of 0 to the values of 1, and the value of 1 to the
rest of the values found in this column.
1 : Highschool
2 : Undergrad
3 : Postgrad
4 : Doctorate

The model has been coverted into a module for future use.

The repository consists of 4 files :
- 1. Absenteeism_classification - Data Preprocessing and EDA
- 2. Absenteeism_LR_model - The machine learning model
- 3. LogReg_Absent_Module - The model and preprocessing converted into a module for deployment.
- 4. EndUSer_Absenteeism - Prediction on new data.
