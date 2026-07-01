import pandas as pd
import argparse
import warnings
import numpy as np

import plotly.graph_objs as go
import plotly.offline as offline
import pandas as pd
from scipy import stats

parser = argparse.ArgumentParser()
parser.add_argument("study")
parser.add_argument("experiments")
parser.add_argument("tfs")
parser.add_argument("watchruns")
parser.add_argument("generations")
parser.add_argument("mainpath")
args = parser.parse_args()

study = args.study
experiments_name = args.experiments.split(',')
tfs = list(args.tfs.split(','))
generations = [0, 100]
mainpath = args.mainpath

path = f'{mainpath}/{study}/analysis/knockouts/data'


def calculate_general():
    origin_file = f'{path}/knockouts_measures.csv'
    df_ori = pd.read_csv(origin_file)
    original = 'o'  # original phenotype, without knockout

    keys = ['experiment_name', 'run', 'gen', 'ranking', 'individual_id']
    traits = ['disp_y', 'distance', 'symmetry', 'extremities_prop']

    traits = [ 'proportion', 'coverage', 'extensiveness_prop', 'branching_prop', 'brick_prop', 'modules_count', 'hinge_ratio', 'hinge_prop', 'head_balance']
    others = ['knockout']
    df = df_ori.filter(items=keys + others + traits)

    # df = df[   ( ( df['experiment_name'] == 'reg2m2') & (df['run'] == 1) &  (df['gen'] == 0 )  ) ] # quick test

    df = df[((df['gen'] == generations[0]) | (df['gen'] == generations[-1]))]

    for trait in traits:
        # sends trait values of each knockout to columns
        pivot_df = df.pivot_table(index=keys,
                                  columns='knockout', values=trait,
                                  # for distance variable, which is not a trait,
                                  # the calculation is idle (compared to 0)
                                  aggfunc='first')

        all_columns = pivot_df.columns
        knock_columns = [col for col in all_columns if col not in keys and col != original]
        # Subtract each knock_columns by the original
        # (positive values mean the mutant had an increase in the trait or growth)
        df_delta = pivot_df.drop(columns=original).sub(pivot_df[original], axis=0)

        double_knocks = [col for col in knock_columns if '.' in col]
        for double_knock in double_knocks:
            genes = double_knock.split('.')

            additive = df_delta[genes[0]] + df_delta[genes[1]]
            df_delta[f'{genes[0]}add{genes[1]}'] = additive
            df_delta[f'{genes[0]}int{genes[1]}'] = df_delta[double_knock] - additive

        int_columns = [col for col in df_delta.columns if 'int' in col]
        positive = df_delta[int_columns] > 0
        neutral = df_delta[int_columns] == 0
        negative = df_delta[int_columns] < 0

        is_finite = np.isfinite(df_delta[int_columns])

        positive = positive & is_finite
        neutral = neutral & is_finite
        negative = negative & is_finite

        count_positive = positive.sum(axis=1)
        count_neutral = neutral.sum(axis=1)
        count_negative = negative.sum(axis=1)

        df_delta['positive'] = count_positive
        df_delta['neutral'] = count_neutral
        df_delta['negative'] = count_negative
        df_delta['total'] = count_positive + count_neutral + count_negative  # obsolete: remove
        df_delta['epistasis'] = count_positive + count_negative

        df_delta['positive'] = df_delta['positive'] / df_delta['total']
        df_delta['neutral'] = df_delta['neutral'] / df_delta['total']
        df_delta['negative'] = df_delta['negative'] / df_delta['total']

        positive_values = df_delta[int_columns].where(positive)
        negative_values = df_delta[int_columns].where(negative)

        positive_avg = positive_values.mean(axis=1, skipna=True)
        negative_avg = negative_values.mean(axis=1, skipna=True)
        df_delta['avg_positive'] = positive_avg
        df_delta['avg_negative'] = negative_avg

        df_exp = df_delta.reset_index()[keys+['neutral', 'positive', 'negative', 'epistasis', 'avg_positive', 'avg_negative']]
        df_exp.to_csv(f'{path}/effects_{trait}.csv')

        print(trait)


calculate_general()


