import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
import pickle
import streamlit as st
df = pd.read_csv(r"C:/Users/91934/OneDrive/Documents/AD2/datasets/heart.csv")

# HEADINGS
st.title('Heart Disease Checkup')
st.sidebar.header('Patient Data')
st.subheader('Training Data Stats')
st.write(df.describe())


# X AND Y DATA
x = df.drop(['target'], axis = 1)
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)

# FUNCTION
def user_report():
  age = st.sidebar.slider('age', 29, 77, 56 )
  sex = st.sidebar.slider('Gender', 0, 1)
  cp = st.sidebar.slider('chest pain', 0,3, 2 )
  trestbps = st.sidebar.slider('persons resting blood pressure', 94,200, 130 )
  chol = st.sidebar.slider('cholesterol', 126,564, 240 )
  fbs = st.sidebar.slider('Fasting blood sugar', 0,1)
  restecg = st.sidebar.slider('Resting electrographic results', 0,2, 1 )
  thalach = st.sidebar.slider('Maximum heart rate achieved', 71,202, 152 )
  exang = st.sidebar.slider('Exercise induced angina', 0,1, 0 )
  oldpeak = st.sidebar.slider('ST depression induced by exercise relative to rest',0.0, 6.2, 0.8)
  slope = st.sidebar.slider('Slope of the peak',0,2,1)
  ca = st.sidebar.slider('Number of major vessels colored by fluoroscopy',0,4)
  thal = st.sidebar.slider('Defect type:0= normal 2= fixed 3= reversible defect',0,3,2)
  
  user_report_data = {
      'age':age,
      'sex':sex,
      'cp':cp,
      'trestbps':trestbps,
      'chol':chol,
      'fbs':fbs,
      'restecg':restecg,
      'thalach':thalach,
      'exang' :exang,
      'oldpeak' : oldpeak,
      'slope' : slope,
      'ca' : ca,
      'thal' : thal
      
  }
  report_data = pd.DataFrame(user_report_data, index=[0])
  return report_data

user_data = user_report()
st.subheader('Patient Data')
st.write(user_data)

rf  = RandomForestClassifier()
rf.fit(x_train, y_train)
user_result = rf.predict(user_data)

st.title('Visualised Patient Report')

if user_result[0]==0:
      color = 'blue'
else:
  color = 'red'
  
# Age vs Gender
st.header('Gender count Graph (Others vs Yours)')
fig_preg = plt.figure()
ax1 = sns.scatterplot(x = 'age', y = 'sex', data = df, hue = 'target', palette = 'Greens')
ax2 = sns.scatterplot(x = user_data['age'], y = user_data['sex'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,1))
plt.title('Gender Count')
st.pyplot(fig_preg)
# Age vs chest pain
st.header('chest pain Value Graph (Others vs Yours)')
fig_chestpain = plt.figure()
ax3 = sns.scatterplot(x = 'age', y = 'cp', data = df, hue = 'target' , palette='magma')
ax4 = sns.scatterplot(x = user_data['age'], y = user_data['cp'], s = 200, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,4))
plt.title('Chest Pain Value')
st.pyplot(fig_chestpain)
# AGE vs trestbps
st.header('Persons Resting Blood Pressure Value Graph (Others vs Yours)')
fig_trestbp = plt.figure()
ax5 = sns.scatterplot(x = 'age', y = 'trestbps', data = df, hue = 'target', palette='Reds')
ax6 = sns.scatterplot(x = user_data['age'], y = user_data['trestbps'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,200,10))
plt.title('Persons Resting Blood Pressure')
st.pyplot(fig_trestbp)
#Age vs chol
st.header('cholesterol Graph (Others vs Yours)')
fig_chol = plt.figure()
ax5 = sns.scatterplot(x = 'age', y = 'chol', data = df, hue = 'target', palette='Blues')
ax6 = sns.scatterplot(x = user_data['age'], y = user_data['chol'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,570,45))
plt.title('Persons Resting Blood Pressure')
st.pyplot(fig_chol)
#Age vs fbs
st.header('Fasting blood sugar Graph (Others vs Yours)')
fig_fbs = plt.figure()
ax5 = sns.scatterplot(x = 'age', y = 'fbs', data = df, hue = 'target', palette='Blues')
ax6 = sns.scatterplot(x = user_data['age'], y = user_data['fbs'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,1))
plt.title('Fasting blood sugar')
st.pyplot(fig_fbs)
#Age vs restecg
st.header('Resting electrographic results Graph (Others vs Yours)')
fig_restecg = plt.figure()
ax5 = sns.scatterplot(x = 'age', y = 'restecg', data = df, hue = 'target', palette='rocket')
ax6 = sns.scatterplot(x = user_data['age'], y = user_data['restecg'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,2))
plt.title('Resting electrographic results')
st.pyplot(fig_restecg)
#Age vs thalach
st.header('Maximum heart rate achieved Graph (Others vs Yours)')
fig_thalach = plt.figure()
ax5 = sns.scatterplot(x = 'age', y = 'thalach', data = df, hue = 'target', palette='rainbow')
ax6 = sns.scatterplot(x = user_data['age'], y = user_data['thalach'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,210,10))
plt.title('Maximum heart rate achieved')
st.pyplot(fig_thalach)
#Age vs exang
st.header('Exercise induced angina Graph (Others vs Yours)')
fig_exang = plt.figure()
ax5 = sns.scatterplot(x = 'age', y = 'exang', data = df, hue = 'target', palette='rainbow')
ax6 = sns.scatterplot(x = user_data['age'], y = user_data['exang'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,1))
plt.title('Exercise induced angina')
st.pyplot(fig_exang)
#Age vs oldpeak
st.header('ST depression Graph (Others vs Yours)')
fig_oldpeak = plt.figure()
ax5 = sns.scatterplot(x = 'age', y = 'oldpeak', data = df, hue = 'target', palette='rainbow')
ax6 = sns.scatterplot(x = user_data['age'], y = user_data['oldpeak'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0.0,6.2))
plt.title('ST depression')
st.pyplot(fig_oldpeak)
#Age vs slope
st.header('Slope of the peak Graph (Others vs Yours)')
fig_slope = plt.figure()
ax5 = sns.scatterplot(x = 'age', y = 'slope', data = df, hue = 'target', palette='rainbow')
ax6 = sns.scatterplot(x = user_data['age'], y = user_data['slope'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,3))
plt.title('Slope of the peak')
st.pyplot(fig_slope)
#Age vs ca
st.header('Number of major vessels colored by fluoroscopy Graph (Others vs Yours)')
fig_ca = plt.figure()
ax5 = sns.scatterplot(x = 'age', y = 'ca', data = df, hue = 'target', palette='rainbow')
ax6 = sns.scatterplot(x = user_data['age'], y = user_data['ca'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,5))
plt.title('Number of major vessels colored by fluoroscopy')
st.pyplot(fig_ca)
#Age vs thal
st.header('Defect Type Graph (Others vs Yours)')
fig_thal = plt.figure()
ax5 = sns.scatterplot(x = 'age', y = 'thal', data = df, hue = 'target', palette='rainbow')
ax6 = sns.scatterplot(x = user_data['age'], y = user_data['thal'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,4))
plt.title('Defect Type')
st.pyplot(fig_thal)

# OUTPUT
st.subheader('Your Report: ')
output=''
if user_result[0]==0:
  output = 'Congrats! You are not having any Heart Disease'
else:
  output = 'Sorry! You are Having Heart Disease' 
st.title(output)
st.subheader('Accuracy: ')
st.write(str(accuracy_score(y_test, rf.predict(x_test))*100)+'%')