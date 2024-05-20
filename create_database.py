import sqlite3
import pandas as pd

# Load the CSV data into a DataFrame
df = pd.read_csv('heart_disease.csv')

# Connect to SQLite database (it will be created if it doesn't exist)
conn = sqlite3.connect('heart_disease.db')

# Save DataFrame to the SQLite database
df.to_sql('heart_disease', conn, if_exists='replace', index=False)

# Close the connection
conn.close()
