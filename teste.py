import pandas

df = pandas.read_csv('life_expectancy_ids_copy.csv')

# Replace 'Developing' with 0 and 'Developed' with 1 in the 'Status' column
df['Status'] = df['Status'].replace({'Developing': 0, 'Developed': 1})

print(df)