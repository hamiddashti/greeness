# import numpy as np
# import pandas as pd
# import tensorflow as tf
# import tensorflow_probability as tfp
# from causalimpact import CausalImpact
from joblib import Parallel, delayed
# import pickle
# import datetime

import sys
import os
import warnings
import logging

logging.getLogger('tensorflow').setLevel(logging.FATAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath('../'))

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from causalimpact import CausalImpact

tfd = tfp.distributions

observed_stddev, observed_initial = (tf.convert_to_tensor(value=1,
                                                          dtype=tf.float32),
                                     tf.convert_to_tensor(value=0.,
                                                          dtype=tf.float32))
level_scale_prior = tfd.LogNormal(loc=tf.math.log(0.05 * observed_stddev),
                                  scale=1,
                                  name='level_scale_prior')
initial_state_prior = tfd.MultivariateNormalDiag(
    loc=observed_initial[..., tf.newaxis],
    scale_diag=(tf.abs(observed_initial) + observed_stddev)[..., tf.newaxis],
    name='initial_level_prior')
ll_ssm = tfp.sts.LocalLevelStateSpaceModel(
    100,
    initial_state_prior=initial_state_prior,
    level_scale=level_scale_prior.sample())
ll_ssm_sample = np.squeeze(ll_ssm.sample().numpy())

x0 = 100 * np.random.rand(100)
x1 = 90 * np.random.rand(100)
y = 1.2 * x0 + 0.9 * x1 + ll_ssm_sample
y[70:] += 10
data = pd.DataFrame({'x0': x0, 'x1': x1, 'y': y}, columns=['y', 'x0', 'x1'])


def causal_inference(k):
    pre_period = [0, 69]
    post_period = [70, 99]
    ci = CausalImpact(
        data,
        pre_period,
        post_period,
        model_args={
            "fit_method": "vi",
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


causal_inference(0)
# results = Parallel(n_jobs=-1)(delayed(causal_inference)(i) for i in range(2))

# def causal_inference(k):
#     print("-----------------------------------\n")
#     print("Pixel number:" + str(k))
#     print("-----------------------------------\n")
#     try:
#         df_tmp = pd.DataFrame(data=data_ndvi, index=t2)
#         pre_priod = ["19840701", "19930701"]
#         post_priod = ["19940701", "20130701"]
#         ci = CausalImpact(
#             df_tmp,
#             pre_priod,
#             post_priod,
#             model_args={
#                 "fit_method": "hmc",
#                 "standardize": True,
#                 # "prior_level_sd": None,
#             },
#         )
#         return [
#             ci.p_value,
#             ci.summary_data.loc["actual"]["average"],
#             ci.summary_data.loc["predicted"]["average"],
#             ci.summary_data.loc["abs_effect"]["average"],
#             ci.summary_data.loc["rel_effect"]["average"],
#             ci.summary_data.loc["actual"]["cumulative"],
#             ci.summary_data.loc["predicted"]["cumulative"],
#             ci.summary_data.loc["abs_effect"]["cumulative"],
#             ci.summary_data.loc["rel_effect"]["cumulative"],
#         ]
#     except:
#         return [np.repeat(np.nan, 9)]

# data = pd.read_pickle(r"../../working/ndvi_1994_data")
# data_ndvi = data[0]
# mask = np.isnan(data_ndvi)
# data_ndvi[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask),
#                             data_ndvi[~mask])

# t = pd.date_range(start="1984", end="2014", freq="A").year
# t2 = pd.date_range(start=datetime.datetime(1984, 1, 1),
#                    periods=len(t),
#                    freq="AS-JUL")

# results = Parallel(n_jobs=-1)(delayed(causal_inference)(i) for i in range(2))

# with open("../../working/CI_out", "wb") as fp:
#     pickle.dump(results, fp)
# print("Done!--------------------------")
