import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from causalimpact import CausalImpact
import dask 
from dask.diagnostics import ProgressBar
import pickle

data_lai = pd.read_pickle(r"/data/home/hamiddashti/hamid/nasa_above/greeness/working/data_lai")
t = pd.date_range(start="1984", end="2014", freq="A-Dec").year
t2 = pd.date_range(start=pd.datetime(1984, 1, 1), periods=len(t), freq="A")

CI = []
# p_value = []
# actual_average = []
# predicted_average = []
# abs_effect_average = []
# rel_effect_average = []
# actual_cumulative = []
# predicted_cumulative = []
# abs_effect_cumulative = []
# rel_effect_cumulative = []


@dask.delayed
def casual_inference(k):
    m = data_lai[k]
    imp = IterativeImputer(random_state=0)
    imp.fit(m)
    X = imp.transform(m)
    df_tmp = pd.DataFrame(data=X, index=t2)
    pre_priod = ["19841231", "19931231"]
    post_priod = ["19941231", "20131231"]
    # pre_priod = [pd.to_datetime(date) for date in ["1984", "1994"]]
    # pre_priod = [pd.to_datetime(date) for date in ["1995", "2013"]]
    # pre_priod = [0, 10]
    # post_priod = [11, len(t) - 1]
    ci = CausalImpact(
        df_tmp,
        pre_priod,
        post_priod,
        model_args={
            # "fit_method": "hmc",
            "standardize": False,
            # "prior_level_sd": None,
        },
    )
    CI.append(ci)
    # # ci = CausalImpact(Data_lai, pre_period, post_period)
    # print(ci.summary())
    # print(ci.summary("report"))
    # ci.plot()
    # p_value.append(ci.p_value)
    # actual_average.append(ci.summary_data.loc["actual"]["average"])
    # predicted_average.append(ci.summary_data.loc["predicted"]["average"])
    # abs_effect_average.append(ci.summary_data.loc["abs_effect"]["average"])
    # rel_effect_average.append(ci.summary_data.loc["rel_effect"]["average"])
    # actual_cumulative.append(ci.summary_data.loc["actual"]["cumulative"])
    # predicted_cumulative.append(ci.summary_data.loc["predicted"]["cumulative"])
    # abs_effect_cumulative.append(ci.summary_data.loc["abs_effect"]["cumulative"])
    # rel_effect_cumulative.append(ci.summary_data.loc["rel_effect"]["cumulative"])

tasks = []
for k in range(len(data_lai)):
    tmp = casual_inference(k)
    tasks.append(tmp)

with ProgressBar():
    dask.compute(*tasks)

# casual_inference_outs = []
# casual_inference_outs.append(p_value)
# casual_inference_outs.append(actual_average)
# casual_inference_outs.append(predicted_average)
# casual_inference_outs.append(abs_effect_average)
# casual_inference_outs.append(rel_effect_average)
# casual_inference_outs.append(actual_cumulative)
# casual_inference_outs.append(predicted_cumulative)
# casual_inference_outs.append(abs_effect_cumulative)
# casual_inference_outs.append(rel_effect_cumulative)

with open("/data/home/hamiddashti/hamid/nasa_above/greeness/working/CI", "wb") as fp:
    pickle.dump(CI, fp)

print("Done!--------------------------")