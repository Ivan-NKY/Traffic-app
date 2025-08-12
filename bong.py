#!/usr/bin/env python
# coding: utf-8

# # <font color='#FFB703'>EATC Assignment</font>
# ## Group members:
# 1. Ivan
# 2. Ming xuan

# # <font color='#FFB703'>Importing Libraries</font>

# In[22]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# # <font color='#FFB703'>Exploratory Data Analysis</font>

# In[23]:


df = pd.read_csv("Train_data.csv")
df.head()


# In[24]:


#Convert the label feature to a binary representation, 0 for 'normal' and 1 for 'anomaly'
le = LabelEncoder()
df['class'] = df['class'].apply(lambda x: 0 if x == 'normal' else 1)
df.head()


# In[25]:


#Check count of normal and malicious attacks
df['class'].value_counts()


# In[26]:


#Encode the categoritical feature of the dataset
le = LabelEncoder()
for column in df.select_dtypes(include='object').columns:
    df[column] = le.fit_transform(df[column])


# In[27]:


#Checking for missing or null values
df.isnull().sum()


# In[28]:


df.drop(columns=['num_outbound_cmds','is_host_login','is_guest_login','land','wrong_fragment', 'urgent', 'num_access_files','num_shells','su_attempted','num_file_creations'], inplace=True)


# In[29]:


#Split into train/test sets
X = df.drop(["class"], axis=1)
y = df["class"]


# In[30]:


#Separate categorical and numerical features for preprocessing
cat_cols = ['protocol_type', 'service', 'flag']
num_cols = [col for col in X.columns if col not in cat_cols]


# In[31]:


# ColumnTransformer one hot encodes categorical features and standardizes numerical features
preprocess = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),('num', StandardScaler(), num_cols)])


# In[32]:


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.1,random_state=43) 


# In[33]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline


# # Logistic Regression

# In[34]:


pipe_lr = Pipeline(steps=[
    ('prep', preprocess),
    ('model', LogisticRegression(max_iter=1000))
])

pipe_lr.fit(X_train, y_train)
y_pred_lr = pipe_lr.predict(X_test)

acc_lr = accuracy_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)

print("Logistic Regression ")
print(f"Accuracy: {acc_lr:.3f}")
print(f"F1 Score: {f1_lr:.3f}")


# In[35]:


cm = confusion_matrix(y_test, y_pred_lr)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal (0)", "Anomaly (1)"], yticklabels=["Normal (0)", "Anomaly (1)"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix Heatmap")
plt.tight_layout()
plt.show()


# # Random Forest

# In[36]:


pipe_rf = Pipeline(steps=[
    ('prep', preprocess),
    ('model', RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1))
])

pipe_rf.fit(X_train, y_train)
y_pred_rf = pipe_rf.predict(X_test)

acc_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

print("Random Forest")
print(f"Accuracy: {acc_rf:.4f}")
print(f"F1 Score: {f1_rf:.4f}")


# In[37]:


cm = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal (0)", "Anomaly (1)"], yticklabels=["Normal (0)", "Anomaly (1)"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix Heatmap")
plt.tight_layout()
plt.show()


# # Decision Tree

# In[38]:


pipe_dt = Pipeline(steps=[
    ('prep', preprocess),
    ('model', DecisionTreeClassifier())
])

pipe_dt.fit(X_train, y_train)
y_pred_dt = pipe_dt.predict(X_test)

acc_dt = accuracy_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)

print("Decision Tree")
print(f"Accuracy: {acc_dt:.3f}")
print(f"F1 Score: {f1_dt:.3f}")


# In[39]:


cm = confusion_matrix(y_test, y_pred_dt)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal (0)", "Anomaly (1)"], yticklabels=["Normal (0)", "Anomaly (1)"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix Heatmap")
plt.tight_layout()
plt.show()


# # Gradient Boosting

# In[40]:


pipe_gb = Pipeline(steps=[
    ('prep', preprocess),
    ('model', GradientBoostingClassifier())
])

pipe_gb.fit(X_train, y_train)
y_pred_gb = pipe_gb.predict(X_test)

acc_gb = accuracy_score(y_test, y_pred_gb)
f1_gb = f1_score(y_test, y_pred_gb)

print("Gradient Boosting")
print(f"Accuracy: {acc_gb:.3f}")
print(f"F1 Score: {f1_gb:.3f}")


# In[41]:


cm = confusion_matrix(y_test, y_pred_gb)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal (0)", "Anomaly (1)"], yticklabels=["Normal (0)", "Anomaly (1)"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix Heatmap")
plt.tight_layout()
plt.show()


# In[42]:


#app
import streamlit as st
#basic introduction to app
st.markdown("# <font color='#FFB703'>Traffic Anomaly Detection App</font>", unsafe_allow_html=True)
st.write("This app uses a Random Forest model to detect anomalies in traffic data.")
st.write("Upload your traffic data file (CSV format) to make predictions.")
#Template for csv file
st.markdown("### Need a sample CSV template?")
#try catch error handling for file not found
try:
    with open("template.csv", "rb") as file:
        template_bytes = file.read()

    st.download_button(
        label="Download Input Template CSV",
        data=template_bytes,
        file_name="network_traffic_template.csv",
        mime="text/csv"
    )
    st.info("Use this template format to ensure your uploaded data matches the model’s structure.")

except FileNotFoundError:
    st.warning("⚠️ 'template.csv' not found in your app folder.")
# Load trained pipeline
pipeline = pipe_rf

# File uploader
uploaded_file = st.file_uploader("Choose a file", type="csv")

if uploaded_file is not None:
    try:
        # Try reading the CSV
        data = pd.read_csv(uploaded_file)
        
        # Empty file check
        if data.empty or data.shape[0] == 0:
            st.error("❌ The uploaded file is empty or contains no rows. Please check your data.")
        else:
            st.write("## Data Preview:")
            st.dataframe(data.head(5))

            try:
                with st.spinner('Processing data...'):
                    # --- Column matching step ---
                    expected_columns = pipeline.feature_names_in_
                    missing_cols = set(expected_columns) - set(data.columns)

                    if missing_cols:
                        st.error(f"❌ Missing required columns: {missing_cols}")
                        st.info("💡 Tip: Download and use the provided CSV template for correct formatting.")
                    else:
                        # Reorder to match training
                        data_input = data[expected_columns]

                        # Predict
                        predictions = pipeline.predict(data_input)

                        # Add predictions
                        data['predictions'] = predictions
                        data['predictions'] = data['predictions'].apply(lambda x: 'anomaly' if x == 1 else 'normal')

                        st.success('✅ Data processed successfully!')

                        # Chart
                        st.subheader("Count of Anomalies")
                        plt.figure(figsize=(7, 5))
                        sns.countplot(x='predictions', data=data, palette="husl")
                        st.pyplot(plt)

                        # Table
                        st.write("## Predictions:")
                        st.dataframe(data)

                        # Download button
                        csv = data.to_csv(index=False)
                        if st.download_button(label="📥 Download Predictions", 
                                              data=csv, 
                                              file_name='predictions.csv', 
                                              mime='text/csv'):
                            st.toast("Predictions downloaded successfully!", icon="✅")

            except ValueError as ve:
                st.error(f"❌ **Model Prediction Error**: {str(ve)}")
                st.error("This usually means your data doesn't match the expected format.")
            except Exception as pe:
                st.error(f"❌ **Prediction Error**: An unexpected error occurred: {str(pe)}")
    
    except pd.errors.EmptyDataError:
        st.error("❌ **File Error**: The uploaded file appears to be empty or corrupted.")
    except pd.errors.ParserError as e:
        st.error(f"❌ **CSV Format Error**: Unable to parse CSV file. {str(e)}")
    except MemoryError:
        st.error("❌ **Memory Error**: The file is too large to process.")
    except Exception as e:
        st.error(f"❌ **Unexpected Error**: {str(e)}")
else:
    st.info('📁 Please upload a CSV file to begin making predictions.')

