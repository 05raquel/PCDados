from pandas import read_csv, DataFrame
from matplotlib.pyplot import figure, show
from dslabs_functions import plot_bar_chart
import pandas


df = pandas.read_csv('life_expectancy_ids.csv')

# Replace 'Developing' with 0 and 'Developed' with 1 in the 'Status' column
df['Status'] = df['Status'].replace({'Developing': 0, 'Developed': 1})
df.to_csv('life_expectancy_ids_copy.csv', index=False)

filename = "life_expectancy_ids_copy.csv"
data: DataFrame = read_csv(filename, index_col="id", na_values="")
#print(f"Dataset nr records={data.shape[0]}", f"nr variables={data.shape[1]}")

mv: dict[str, int] = {}
figure()
for var in data:
    nr: int = data[var].isna().sum()
    if nr > 0:
        mv[var] = nr


plot_bar_chart(
    list(mv.keys()),
    list(mv.values()),
    title="Missing values per variable",
    xlabel="variables",
    ylabel="nr missing values",
)
show()

'''
# apagado quando pelo menos 1 variavel tem NA
df_remove: DataFrame = data.dropna(how="any", inplace=False)
print("\nV1")
print(df_remove.shape)



# apagado quando todas os valores das variaveis da linha sao NA
df_v2: DataFrame = data.dropna(how="all", inplace=False)
print("\nV2")
print(df_v2.shape)

# retirar colunas com NA
df_v3: DataFrame = data.dropna(axis=1, how="any", inplace=False)
print("\nV3")
print(df_v3.shape)
'''
## Vamos usar df_v1 ##

def mvi_by_dropping(
    data: DataFrame, min_pct_per_var: float = 0.1, min_pct_per_rec: float = 0.0
) -> DataFrame:
    # Deleting variables
    df: DataFrame = data.dropna(
        axis=1, thresh=data.shape[0] * min_pct_per_var, inplace=False
    )
    # Deleting records
    df.dropna(axis=0, thresh=data.shape[1] * min_pct_per_rec, inplace=True)

    return df


df_remove_col: DataFrame = mvi_by_dropping(data, min_pct_per_var=0.9, min_pct_per_rec=0)
df_remove: DataFrame = df_remove_col.dropna(how="any", inplace=False)
print("\nV4")
print(df_remove.shape)

df_remove.to_csv(f"remove_col_lin_dataset.csv", index=True)

mv1: dict[str, int] = {}
figure()
for var in df_remove:
    nr: int = df_remove[var].isna().sum()
    if nr > 0:
        mv1[var] = nr

plot_bar_chart(
    list(mv1.keys()),
    list(mv1.values()),
    title="Missing values per variable",
    xlabel="variables",
    ylabel="nr missing values",
)
show()


'''### Abordagem de preencher missing values

from numpy import ndarray
from pandas import concat
from sklearn.impute import SimpleImputer, KNNImputer
from dslabs_functions import get_variable_types, mvi_by_filling


def mvi_by_filling(data: DataFrame, strategy: str = "frequent") -> DataFrame:
    df: DataFrame
    variables: dict = get_variable_types(data)
    stg_num, v_num = "mean", -1
    stg_sym, v_sym = "most_frequent", "NA"
    stg_bool, v_bool = "most_frequent", False
    if strategy != "knn":
        lst_dfs: list = []
        # se a estratégia for constante, são usados os valores -1, NA e False
        if strategy == "constant":
            stg_num, stg_sym, stg_bool = "constant", "constant", "constant"

        if len(variables["numeric"]) > 0:
            imp = SimpleImputer(strategy=stg_num, fill_value=v_num, copy=True)
            tmp_nr = DataFrame(
                imp.fit_transform(data[variables["numeric"]]),
                columns=variables["numeric"],
            )
            lst_dfs.append(tmp_nr)
        if len(variables["symbolic"]) > 0:
            imp = SimpleImputer(strategy=stg_sym, fill_value=v_sym, copy=True)
            tmp_sb = DataFrame(
                imp.fit_transform(data[variables["symbolic"]]),
                columns=variables["symbolic"],
            )
            lst_dfs.append(tmp_sb)
        if len(variables["binary"]) > 0:
            imp = SimpleImputer(strategy=stg_bool, fill_value=v_bool, copy=True)
            tmp_bool = DataFrame(
                imp.fit_transform(data[variables["binary"]]),
                columns=variables["binary"],
            )
            lst_dfs.append(tmp_bool)
        df = concat(lst_dfs, axis=1)
    else:
        imp = KNNImputer(n_neighbors=5)
        imp.fit(data)
        ar: ndarray = imp.transform(data)
        df = DataFrame(ar, columns=data.columns, index=data.index)
    return df


df_remove_col_fil: DataFrame = mvi_by_filling(df_remove_col, strategy="frequent")
#print(df_fill.head(10))

'''

data = df_remove

from numpy import array, ndarray
from pandas import read_csv, DataFrame

file_tag = "life_expectancy_ids_mv_remove_col_lin"
index_col = "id"
target = "Status"
labels: list = list(data[target].unique())
labels.sort()
print(f"Labels={labels}")

positive: int = 1 #Developed
negative: int = 0 #Developing
values: dict[str, list[int]] = {
    "Original": [
        len(data[data[target] == negative]),
        len(data[data[target] == positive]),
    ]
}

y: array = data.pop(target).to_list()
X: ndarray = data.values

print(values)



from pandas import concat
from matplotlib.pyplot import figure, show
from sklearn.model_selection import train_test_split
from dslabs_functions import plot_multibar_chart

target = "Status"

trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

train: DataFrame = concat(
    [DataFrame(trnX, columns=data.columns), DataFrame(trnY, columns=[target])], axis=1
)
train.to_csv(f"data/{file_tag}_train.csv", index=False)

test: DataFrame = concat(
    [DataFrame(tstX, columns=data.columns), DataFrame(tstY, columns=[target])], axis=1
)
test.to_csv(f"data/{file_tag}_test.csv", index=False)

values["Train"] = [
    len(train[train[target] == negative]),
    len(train[train[target] == positive]),
]
values["Test"] = [
    len(test[test[target] == negative]),
    len(test[test[target] == positive]),
]

figure(figsize=(6, 4))
plot_multibar_chart(labels, values, title="Data distribution per dataset")
show()
