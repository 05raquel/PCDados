from numpy import ndarray
from pandas import DataFrame, read_csv
from matplotlib.pyplot import savefig, show, figure
from dslabs_functions import plot_multibar_chart, CLASS_EVAL_METRICS, run_NB, run_KNN


def evaluate_approach(
    train: DataFrame, test: DataFrame, target: str = "class", metric: str = "accuracy"
) -> dict[str, list]:
    trnY = train.pop(target).values
    trnX: ndarray = train.values
    tstY = test.pop(target).values
    tstX: ndarray = test.values
    eval: dict[str, list] = {}

    eval_NB: dict[str, float] = run_NB(trnX, trnY, tstX, tstY, metric=metric)
    eval_KNN: dict[str, float] = run_KNN(trnX, trnY, tstX, tstY, metric=metric)
    if eval_NB != {} and eval_KNN != {}:
        for met in CLASS_EVAL_METRICS:
            eval[met] = [eval_NB[met], eval_KNN[met]]

    #if eval_KNN != {}:
    #    for met in CLASS_EVAL_METRICS:
    #        eval[met] = [ eval_KNN[met]]
    return eval


'''target = "Status"
file_tag = "life_expectancy_ids_minmax"
train: DataFrame = read_csv("data/life_expectancy_ids_df_minmax_train.csv")
test: DataFrame = read_csv("data/life_expectancy_ids_df_minmax_test.csv")'''
target = "Status"
file_tag = "life_expectancy_ids_zscore-over"
train: DataFrame = read_csv("data/life_expectancy_ids_df_zscore_train.csv")
test: DataFrame = read_csv("data/life_expectancy_ids_df_zscore_test.csv")

figure()
eval: dict[str, list] = evaluate_approach(train, test, target=target, metric="f1")
plot_multibar_chart(
    #["KNN"], eval, title=f"{file_tag} evaluation", percentage=True
    ["NB", "KNN"], eval, title=f"{file_tag} evaluation", percentage=True
)
#savefig(f"images/{file_tag}_eval.png")
show()
