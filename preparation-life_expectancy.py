

############################################# Missing Values imputation #############################################
from pandas import read_csv, DataFrame
from matplotlib.pyplot import figure, show
from dslabs_functions import plot_bar_chart

filename = "life_expectancy_ids.csv"
data: DataFrame = read_csv(filename, index_col="id", na_values="")
#print(f"Dataset nr records={data.shape[0]}", f"nr variables={data.shape[1]}")

'''mv: dict[str, int] = {}
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
df: DataFrame = data.dropna(how="any", inplace=False)
#print(df.shape)

# apagado quando todas os valores das variaveis da linha sao NA
df: DataFrame = data.dropna(how="all", inplace=False)
#print(df.shape)

# retirar colunas com NA
df: DataFrame = data.dropna(axis=1, how="any", inplace=False)
#print(df.shape)

##
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


#df: DataFrame = mvi_by_dropping(data, min_pct_per_var=0.7, min_pct_per_rec=0.9)
#print(df.shape)

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


df: DataFrame = mvi_by_filling(data, strategy="frequent")
#print(df.head(10))

#life_expectancy_ids.csv

data: DataFrame = read_csv("life_expectancy_ids.csv", index_col="id", dayfirst=True)

'''print("Algae dataset:", data.shape)
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
show()'''

numeric_vars: list[str] = get_variable_types(data)["numeric"]
df: DataFrame = mvi_by_filling(data[numeric_vars], strategy="frequent")
#print("MVI frequent strategy", df.describe())
# fazer excel?! para comparar distribuição de ambos os casos - sem alterações, frequente e knn


'''df: DataFrame = mvi_by_filling(data[numeric_vars], strategy="knn")
print("MVI KNN strategy", df.describe())'''

############################################# Scaling #############################################

from pandas import read_csv, DataFrame, Series
from sklearn.preprocessing import StandardScaler

file = "life_expectancy"
data: DataFrame = read_csv("life_expectancy_ids.csv", index_col="id", na_values="")
target ="Status"  #"Life expectancy "
vars: list[str] = data.columns.to_list()
target_data: Series = data.pop(target)

transf: StandardScaler = StandardScaler(with_mean=True, with_std=True, copy=True).fit(
    data
)
df_zscore = DataFrame(transf.transform(data), index=data.index)
df_zscore[target] = target_data
df_zscore.columns = vars
df_zscore.to_csv(f"data/{file}_scaled_zscore.csv", index="id")


from sklearn.preprocessing import MinMaxScaler

'''transf: MinMaxScaler = MinMaxScaler(feature_range=(0, 1), copy=True).fit(data)
df_minmax = DataFrame(transf.transform(data), index=data.index)
df_minmax[target] = target_data
df_minmax.columns = vars
df_minmax.to_csv(f"data/{file}_scaled_minmax.csv", index="id")
'''
from matplotlib.pyplot import subplots, show
from matplotlib.pyplot import figure, savefig, show
'''
file_tag = "life_expectancy"
fig, axs = subplots(1, 3, figsize=(20, 10), squeeze=False)
axs[0, 1].set_title("Original data")
data.boxplot(ax=axs[0, 0])
axs[0, 0].set_title("Z-score normalization")
df_zscore.boxplot(ax=axs[0, 1])
axs[0, 2].set_title("MinMax normalization")
df_minmax.boxplot(ax=axs[0, 2])
savefig(f"images/{file_tag}_scaling.png")
show()
'''

from pandas import read_csv, concat, DataFrame, Series
from matplotlib.pyplot import figure, show
from dslabs_functions import plot_bar_chart

# criar ficheiro sem NaN
file = "life_expectancy_ids"
#data: DataFrame = read_csv("life_expectancy_ids.csv", index_col="id", na_values="")
data: DataFrame = read_csv(filename, index_col="id", na_values="")
df1: DataFrame = mvi_by_filling(data, strategy="frequent")
df1.to_csv(f"data/{file}_filling.csv", index="id")
data.to_csv(f"data/{file}_teste.csv", index="id")

#file = "life_expectancy_train"
file = "life_expectancy_ids"
target = "Status"
original: DataFrame = read_csv(f"{file}.csv", sep=",", decimal=".")


target_count: Series = original[target].value_counts()
positive_class = target_count.idxmin()
negative_class = target_count.idxmax()

'''print("Minority class=", positive_class, ":", target_count[positive_class])
print("Majority class=", negative_class, ":", target_count[negative_class])
print(
    "Proportion:",
    round(target_count[positive_class] / target_count[negative_class], 2),
    ": 1",
)'''
values: dict[str, list] = {
    "Original": [target_count[positive_class], target_count[negative_class]]
}

figure()
plot_bar_chart(
    target_count.index.to_list(), target_count.to_list(), title="Class balance"
)
show()

df_positives: Series = original[original[target] == positive_class]
df_negatives: Series = original[original[target] == negative_class]

# UNDERSAMPLING --> 512
'''
df_neg_sample: DataFrame = DataFrame(df_negatives.sample(len(df_positives)))
df_under: DataFrame = concat([df_positives, df_neg_sample], axis=0)
df_under.to_csv(f"data/{file}_under.csv", index=False)

print("Minority class=", positive_class, ":", len(df_positives))
print("Majority class=", negative_class, ":", len(df_neg_sample))
print("Proportion:", round(len(df_positives) / len(df_neg_sample), 2), ": 1")'''

# OVERSAMPLING --> 2426

'''df_pos_sample: DataFrame = DataFrame(
    df_positives.sample(len(df_negatives), replace=True)
)
df_over: DataFrame = concat([df_pos_sample, df_negatives], axis=0)
df_over.to_csv(f"data/{file}_over.csv", index=False)

print("Minority class=", positive_class, ":", len(df_pos_sample))
print("Majority class=", negative_class, ":", len(df_negatives))
print("Proportion:", round(len(df_pos_sample) / len(df_negatives), 2), ": 1")
'''

# SMOTEs

from numpy import ndarray
from pandas import Series
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42
original: DataFrame = df1


smote: SMOTE = SMOTE(sampling_strategy="minority", random_state=RANDOM_STATE)
y = original.pop(target).values
X: ndarray = original.values
smote_X, smote_y = smote.fit_resample(X, y)
df_smote: DataFrame = concat([DataFrame(smote_X), DataFrame(smote_y)], axis=1)
df_smote.columns = list(original.columns) + [target]
df_smote.to_csv(f"data/{file}_smote.csv", index=False)

smote_target_count: Series = Series(smote_y).value_counts()
print("Minority class=", positive_class, ":", smote_target_count[positive_class])
print("Majority class=", negative_class, ":", smote_target_count[negative_class])
print(
    "Proportion:",
    round(smote_target_count[positive_class] / smote_target_count[negative_class], 2),
    ": 1",
)
print(df_smote.shape)
