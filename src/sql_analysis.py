import pandas as pd
from sqlalchemy import create_engine

engine = create_engine(r"sqlite:///../data/medical.db")

query1 = "SELECT * FROM patients LIMIT 5;"
df_preview = pd.read_sql_query(query1, engine)

print(df_preview)

query2 = "SELECT COUNT(*) AS totalPatients FROM patients"
print(pd.read_sql_query(query2, engine))

query3 = "SELECT AVG(glucose) AS avgGlucose, AVG(bmi) AS avgBMI, AVG(age) AS avgAge FROM patients"
print(pd.read_sql_query(query3, engine))