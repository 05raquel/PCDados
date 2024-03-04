from pandas import read_csv, DataFrame
import csv
from matplotlib.pyplot import figure, savefig, show
from dslabs_functions import plot_bar_chart
from pandas import Series, to_numeric, to_datetime

'''
# Define the input and output file paths
input_file = 'amostra.csv'
output_file = 'amostra_ids.csv'

# Function to add ID to each line
def add_id_to_csv(input_file, output_file):
    with open(input_file, 'r', newline='') as csv_in:
        with open(output_file, 'w', newline='') as csv_out:
            reader = csv.reader(csv_in)
            writer = csv.writer(csv_out)

            # Add ID and write to output CSV
            for i , row in enumerate(reader, 0):
                if i == 0:
                    new_row = ["id"] + row
                    writer.writerow(new_row)
                else:
                    new_row = [str(i)] + row  # Add ID to the beginning of each row
                    writer.writerow(new_row)



# Call the function to add ID to CSV
add_id_to_csv(input_file, output_file)

filename = "amostra_ids.csv"
file_tag = "amostra"
data: DataFrame = read_csv(filename, na_values="", index_col="id")

print(data.shape)
'''

'''
figure(figsize=(4, 2))
values: dict[str, int] = {"nr records": data.shape[0], "nr variables": data.shape[1]}
plot_bar_chart(
    list(values.keys()), list(values.values()), title="Nr of records vs nr variables"
)
savefig(f"images/{file_tag}_records_variables.png")
show()

'''

'''

mv: dict[str, int] = {}
for var in data.columns:
    nr: int = data[var].isna().sum()
    if nr > 0:
        mv[var] = nr

figure()
plot_bar_chart(
    list(mv.keys()),
    list(mv.values()),
    title="Nr of missing values per variable",
    xlabel="variables",
    ylabel="nr missing values",
)
savefig(f"images/{file_tag}_mv.png")
show()
'''

'''
print(data.dtypes)



def get_variable_types(df: DataFrame) -> dict[str, list]:
    variable_types: dict = {"numeric": [], "binary": [], "date": [], "symbolic": []}

    nr_values: Series = df.nunique(axis=0, dropna=True)
    for c in df.columns:
        if 2 == nr_values[c]:
            variable_types["binary"].append(c)
            df[c].astype("bool")
        else:
            try:
                to_numeric(df[c], errors="raise")
                variable_types["numeric"].append(c)
            except ValueError:
                try:
                    df[c] = to_datetime(df[c], errors="raise")
                    variable_types["date"].append(c)
                except ValueError:
                    variable_types["symbolic"].append(c)

    return variable_types

variable_types: dict[str, list] = get_variable_types(data)
print(variable_types)
counts: dict[str, int] = {}
for tp in variable_types.keys():
    counts[tp] = len(variable_types[tp])

figure(figsize=(4, 2))
plot_bar_chart(
    list(counts.keys()), list(counts.values()), title="Nr of variables per type"
)
savefig(f"images/{file_tag}_variable_types.png")
show()
'''


################################################## DISTRIBUTION #############################################


from pandas import DataFrame, read_csv


file_tag = "amostra"
data: DataFrame = read_csv("amostra_ids.csv", index_col="id", na_values="")
summary5: DataFrame = data.describe(include="all")
print(summary5)

var: str = "infant deaths"
print(f"Summary for {var} variable:")
print("\tCount: ", summary5[var]["count"])
print("\tMean: ", summary5[var]["mean"])
print("\tStDev: ", summary5[var]["std"])
print("\tMin: ", summary5[var]["min"])
print("\tQ1: ", summary5[var]["25%"])
print("\tMedian: ", summary5[var]["50%"])
print("\tQ3: ", summary5[var]["75%"])
print("\tMax: ", summary5[var]["max"])

var = "Alcohol"
print(f"Summary for {var} variable:")
print("\tCount: ", summary5[var]["count"])
print("\tUnique: ", summary5[var]["unique"])
print("\tTop: ", summary5[var]["top"])
print("\tFreq: ", summary5[var]["freq"])