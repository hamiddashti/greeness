import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from joblib import Parallel, delayed
import pickle
import random
import asyncio
import multiprocessing as mp
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
p = mp.Pool(processes=4)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def causal_inference(k):
    print("-----------------------------------\n")
    print("Pixel number:" + str(k))
    print("-----------------------------------\n")
    try:
        m = data[k]
        mask = np.isnan(m)
        m[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask),
                            m[~mask])
        df_tmp = pd.DataFrame(data=m, index=t2)
        pre_priod = ["19840701", "19930701"]
        post_priod = ["19940701", "20130701"]
        ci = CausalImpact(
            df_tmp,
            pre_priod,
            post_priod,
            model_args={
                "fit_method": "hmc",
                "standardize": True,
                # "prior_level_sd": None,
            },
        )
        return [
            ci.p_value,
            ci.summary_data.loc["actual"]["average"],
            ci.summary_data.loc["predicted"]["average"],
            ci.summary_data.loc["abs_effect"]["average"],
            ci.summary_data.loc["rel_effect"]["average"],
            ci.summary_data.loc["actual"]["cumulative"],
            ci.summary_data.loc["predicted"]["cumulative"],
            ci.summary_data.loc["abs_effect"]["cumulative"],
            ci.summary_data.loc["rel_effect"]["cumulative"],
        ]
    except:
        return [np.repeat(np.nan, 9)]


data = pd.read_pickle(r"../../working/ndvi_1994_data")

t = pd.date_range(start="1984", end="2014", freq="A").year
t2 = pd.date_range(start=pd.datetime(1984, 1, 1),
                   periods=len(t),
                   freq="AS-JUL")

# results = []
# if __name__ == '__main__':
#     N = mp.cpu_count()
#     with mp.Pool(processes=N) as p:
#         results.append(p.map(causal_inference, [k for k in range(4)]))

# print(results)
results = Parallel(n_jobs=16)(delayed(causal_inference)(i) for i in range(1))

with open("../../working/CI_out", "wb") as fp:
    pickle.dump(results, fp)
print("Done!--------------------------")
