import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sqlalchemy import create_engine

engine = create_engine(r"sqlite:///../data/medical.db")


df = pd.read_sql_query("SELECT * FROM patients", engine)
print(df.shape)

print(df.describe)

#visualization for glucose
plt.figure(figsize=(8,5))
sns.histplot(df["glucose"], kde=True, color='red')
plt.title("glucose level distribution")
plt.xlabel("glucose")
plt.ylabel("count")
#plt.show()

#visualization for bmi
plt.figure(figsize=(8,5))
sns.histplot(df["bmi"], kde=True, color ='green')
plt.title("BMI Distribution")
plt.xlabel("BMI") 
plt.ylabel("count")
#plt.show()

#visualization for correlatioon
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("correlation between features (heatmap)")
plt.show()

#glucose vs age outcome

plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x = "age", y = "glucose", hue="outcome", palette="coolwarm")
plt.title("Glucose vs Age (Diabetic vs Non-Diabetic)")
plt.xlabel("Age")
plt.ylabel("Glucose Level")
plt.show()