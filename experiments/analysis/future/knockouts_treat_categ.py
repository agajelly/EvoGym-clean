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


def positive_and_zero(a, b):
    return ((a > 0) & (b == 0)) | ((a == 0) & (b > 0))

def negative_and_zero(a, b):
    return ((a < 0) & (b == 0)) | ((a == 0) & (b < 0))

def positive_positive(a, b):
    return (a > 0) & (b > 0)

def negative_negative(a, b):
    return (a < 0) & (b < 0)

def positive_negative(a, b):
    return ((a > 0) & (b < 0)) | ((a < 0) & (b > 0))

def calculate_general():
    origin_file = f'{path}/knockouts_measures.csv'
    df_ori = pd.read_csv(origin_file)
    original = 'o'  # original phenotype, without knockout

    keys = ['experiment_name', 'run', 'gen', 'ranking', 'individual_id']
    traits = ['disp_y', 'distance', 'symmetry', 'extremities_prop']
    traits = ['proportion', 'coverage', 'extensiveness_prop', 'branching_prop', 'brick_prop', 'modules_count',
             'hinge_ratio', 'hinge_prop', 'head_balance']

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
        for mxy in double_knocks:
            genes = mxy.split('.')

            mx = genes[0]
            my = genes[1]

            df_delta[f'{mx}categ{my}'] = ''

            # Condition 1: 'buffering'
            buffering_conditions = [
                (df_delta[mx] == 0) & (df_delta[my] == 0) & (df_delta[mxy] > 0),
                (df_delta[mx] == 0) & (df_delta[my] == 0) & (df_delta[mxy] < 0)
            ]
            df_delta[f'{mx}categ{my}'] = np.select(buffering_conditions, ['buffering'] * len(buffering_conditions),
                                                   default=df_delta[f'{mx}categ{my}'])

            # Condition 2: 'suppression'
            suppression_conditions = [
                (positive_positive(df_delta[mx], df_delta[my]) & (df_delta[mxy] == 0)),
                (positive_and_zero(df_delta[mx], df_delta[my]) & (df_delta[mxy] == 0)),
                (negative_negative(df_delta[mx], df_delta[my]) & (df_delta[mxy] == 0)),
                (negative_and_zero(df_delta[mx], df_delta[my]) & (df_delta[mxy] == 0))
            ]
            df_delta[f'{mx}categ{my}'] = np.select(suppression_conditions,
                                                   ['suppression'] * len(suppression_conditions),
                                                   default=df_delta[f'{mx}categ{my}'])

            # Condition 3: 'quantitative_buffering'
            quant_buffering_conditions = [
                (positive_positive(df_delta[mx], df_delta[my]) & (df_delta[mxy] > df_delta[mx]) & (
                            df_delta[mxy] > df_delta[my])),
                (positive_and_zero(df_delta[mx], df_delta[my]) & (df_delta[mxy] > df_delta[mx]) & (
                            df_delta[mxy] > df_delta[my])),
                (negative_negative(df_delta[mx], df_delta[my]) & (df_delta[mxy] < df_delta[mx]) & (
                            df_delta[mxy] < df_delta[my])),
                (negative_and_zero(df_delta[mx], df_delta[my]) & (df_delta[mxy] < df_delta[mx]) & (
                            df_delta[mxy] < df_delta[my]))
            ]
            df_delta[f'{mx}categ{my}'] = np.select(quant_buffering_conditions,
                                                   ['quantitative_buffering'] * len(quant_buffering_conditions),
                                                   default=df_delta[f'{mx}categ{my}'])

            # Condition 4: 'quantitative_suppression'
            quant_suppression_conditions = [
                (positive_positive(df_delta[mx], df_delta[my]) & (df_delta[mxy] < df_delta[mx]) & (
                            df_delta[mxy] < df_delta[my]) & (df_delta[mxy] > 0)),
                (positive_and_zero(df_delta[mx], df_delta[my]) & (df_delta[mxy] < df_delta[mx]) & (
                            df_delta[mxy] < df_delta[my]) & (df_delta[mxy] > 0)),
                (negative_negative(df_delta[mx], df_delta[my]) & (df_delta[mxy] > df_delta[mx]) & (
                            df_delta[mxy] > df_delta[my]) & (df_delta[mxy] < 0)),
                (negative_and_zero(df_delta[mx], df_delta[my]) & (df_delta[mxy] > df_delta[mx]) & (
                            df_delta[mxy] > df_delta[my]) & (df_delta[mxy] < 0))
            ]
            df_delta[f'{mx}categ{my}'] = np.select(quant_suppression_conditions,
                                                   ['quantitative_suppression'] * len(quant_suppression_conditions),
                                                   default=df_delta[f'{mx}categ{my}'])

            # Condition 5: 'masking'

            # partial masking
            # masking_conditions = [
            #     (positive_negative(df_delta[mx], df_delta[my]) & ( df_delta[mxy] > 0)),
            #     (positive_negative(df_delta[mx], df_delta[my]) & ( df_delta[mxy] < 0))
            # ]
            
            # complete masking
            masking_conditions = [
                 (positive_negative(df_delta[mx], df_delta[my]) & (df_delta[mxy] == df_delta[mx]) ),
                 (positive_negative(df_delta[mx], df_delta[my]) & (df_delta[mxy] == df_delta[my]) )
            ]

            df_delta[f'{mx}categ{my}'] = np.select(masking_conditions, ['masking'] * len(masking_conditions),
                                                   default=df_delta[f'{mx}categ{my}'])

            # Condition 6: 'inversion'
            inversion_conditions = [
                (positive_positive(df_delta[mx], df_delta[my]) & (df_delta[mxy] < 0)),
                (positive_and_zero(df_delta[mx], df_delta[my]) & (df_delta[mxy] < 0)),
                (negative_negative(df_delta[mx], df_delta[my]) & (df_delta[mxy] > 0)),
                (negative_and_zero(df_delta[mx], df_delta[my]) & (df_delta[mxy] > 0))
            ]
            df_delta[f'{mx}categ{my}'] = np.select(inversion_conditions, ['inversion'] * len(inversion_conditions),
                                                   default=df_delta[f'{mx}categ{my}'])

        categ_columns = [col for col in df_delta.columns if 'categ' in col]

        buffering = df_delta[categ_columns] == 'buffering'
        suppression = df_delta[categ_columns] == 'suppression'
        quantitative_buffering = df_delta[categ_columns] == 'quantitative_buffering'
        quantitative_suppression = df_delta[categ_columns] == 'quantitative_suppression'
        masking = df_delta[categ_columns] == 'masking'
        inversion = df_delta[categ_columns] == 'inversion'

        count_buffering = buffering.sum(axis=1)
        count_suppression = suppression.sum(axis=1)
        count_quantitative_buffering = quantitative_buffering.sum(axis=1)
        count_quantitative_suppression = quantitative_suppression.sum(axis=1)
        count_masking = masking.sum(axis=1)
        count_inversion = inversion.sum(axis=1)

        df_delta['buffering'] = count_buffering
        df_delta['suppression'] = count_suppression
        df_delta['quantitative_buffering'] = count_quantitative_buffering
        df_delta['quantitative_suppression'] = count_quantitative_suppression
        df_delta['masking'] = count_masking
        df_delta['inversion'] = count_inversion

        df_delta['epistasis'] = df_delta['buffering'] + df_delta['suppression'] + \
                                df_delta['quantitative_buffering'] + df_delta['quantitative_suppression'] + \
                                df_delta['masking'] + df_delta['inversion']

        df_delta['buffering'] = df_delta['buffering'] / df_delta['epistasis']
        df_delta['suppression'] = df_delta['suppression'] / df_delta['epistasis']
        df_delta['quantitative_buffering'] = df_delta['quantitative_buffering'] / df_delta['epistasis']
        df_delta['quantitative_suppression'] = df_delta['quantitative_suppression'] / df_delta['epistasis']
        df_delta['masking'] = df_delta['masking'] / df_delta['epistasis']
        df_delta['inversion'] = df_delta['inversion'] / df_delta['epistasis']

        df_exp = df_delta.reset_index()[
            keys + ['buffering', 'suppression', 'quantitative_buffering', 'quantitative_suppression', 'masking',
                    'inversion', 'epistasis']]

        df_exp.to_csv(f'{path}/effectscateg_{trait}.csv')

        print(trait)


calculate_general()



########### CATEGS TESTS #########


# #buffeering
# mx=0
# my=0
# mxy=1

# mx=0
# my=0
# mxy=-1

# #supression
# mx=1
# my=1
# mxy=0

# mx=1
# my=1
# mxy=0

# mx=-1
# my=-1
# mxy=0

# mx=-1
# my=0
# mxy=0

# wuant buff
# mx=1
# my=1
# mxy=2

# mx=1
# my=0
# mxy=2

# mx=-1
# my=0
# mxy=-2

# # masking
# mx=1
# my=-1
# mxy=1

# mx=1
# my=-1
# mxy=-1

# # inversion
# mx=1
# my=1
# mxy=-1

# mx=-1
# my=0
# mxy=1

# mx=-1
# my=-1
# mxy=1

#
# def positive_and_zero(a, b):
#     return (a > 0 and b == 0) or (a == 0 and b > 0)
#
#
# def negative_and_zero(a, b):
#     return (a < 0 and b == 0) or (a == 0 and b < 0)
#
#
# def positive_positive(a, b):
#     return a > 0 and b > 0
#
#
# def negative_negative(a, b):
#     return a < 0 and b < 0
#
#
# if (mx == 0 and my == 0 and mxy > 0) or \
#         (mx == 0 and my == 0 and mxy < 0):
#     print('beffering')
#
# if (positive_positive(mx, my) and mxy == 0) or \
#         (positive_and_zero(mx, my) and mxy == 0) or \
#         (negative_negative(mx, my) and mxy == 0) or \
#         (negative_and_zero(mx, my) and mxy == 0):
#     print('supreesion')
#
# if (positive_positive(mx, my) and mxy > mx and mxy > my) or \
#         (positive_and_zero(mx, my) and mxy > mx and mxy > my) or \
#         (negative_negative(mx, my) and mxy < mx and mxy < my) or \
#         (negative_and_zero(mx, my) and mxy < mx and mxy < my):
#     print('quantitative_buffering')
#
# if (positive_positive(mx, my) and mxy < mx and mxy < my and mxy > 0) or \
#         (positive_and_zero(mx, my) and mxy < mx and mxy < my and mxy > 0) or \
#         (negative_negative(mx, my) and mxy > mx and mxy > my and mxy < 0) or \
#         (negative_and_zero(mx, my) and mxy > mx and mxy > my and mxy < 0):
#     print('quantitative_supression')
#
# if (((mx > 0 and my < 0) or (mx < 0 and my > 0)) \
#     and mxy > 0) or \
#         (((mx > 0 and my < 0) or (mx < 0 and my > 0)) \
#          and mxy < 0):
#     print('masking')
#
# if (positive_positive(mx, my) and mxy < 0) or \
#         (positive_and_zero(mx, my) and mxy < 0) or \
#         (negative_negative(mx, my) and mxy > 0) or \
#         (negative_and_zero(mx, my) and mxy > 0):
#     print('inversion')




