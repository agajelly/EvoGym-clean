import argparse
import pandas
import matplotlib.pyplot as plt
import seaborn as sb
from statannot import add_stat_annotation

from scipy.stats import  ttest_ind
from scipy.stats import mannwhitneyu
import os
import warnings
warnings.filterwarnings("ignore", message="Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0")


parser = argparse.ArgumentParser()
parser.add_argument("study")
parser.add_argument("experiments")
parser.add_argument("runs")
parser.add_argument("generations")
parser.add_argument("comparison")
parser.add_argument("mainpath")
args = parser.parse_args()

study = args.study
experiments_name = args.experiments.split(',')
#runs = list(map(int, args.runs.split(',')))
generations = list(map(int, args.generations.split(',')))
comparison = args.comparison
mainpath = args.mainpath

experiments = experiments_name
inner_metrics = ['mean', 'max']
include_max = False
merge_lines = True
gens_boxes = generations
clrs = ['#ADD8E6',
        '#F6B970' ]
#,
        # '#434699',
        # '#95fc7a',
        # '#221210',
        # '#87ac65']
path = f'{mainpath}/{study}'

measures = {
    'pop_diversity': ['Diversity', 0, 1],
   # 'novelty': ['Novelty', 0, 1],
    'seasonal_dominated': ['Seasonal Dominated', 0, 1],
    'age': ['Age', 0, 1],
    'speed_y': ['Speed (cm/s)', 0, 1],
    'disp_y': ['fitness (displacement)', 0, 100], #Fitness
    'relative_speed_y': ['Relative speed (cm/s)', 0, 1],
    'displacement': ['Total displacement (m)', 0, 1],
    'average_z': ['Z', 0, 1],
    'head_balance': ['Balance', 0, 1],
    'modules_count': ['size', 0, 1], #Size
    'hinge_count': ['Hinge count', 0, 1],
    'brick_count': ['Brick count', 0, 1],
    'hinge_prop': ['Joints', 0, 1],
    'hinge_ratio': ['Joints ratio', 0, 1],
    'brick_prop': ['Bricks', 0, 1],
    'branching_count': ['Branching count', 0, 1],
    'branching_prop': ['Branching', 0, 1],
    'extremities': ['Limbs count', 0, 1],
    'extensiveness': ['Length Limbs count', 0, 1],
    'extremities_prop': ['Limbs', 0, 1],
    'extensiveness_prop': ['Length Limbs', 0, 1],
    'width': ['Width', 0, 1],
    'height': ['Height', 0, 1],
    'coverage': ['Coverage', 0, 1],
    'proportion': ['Proportion', 0, 1],
    'symmetry': ['Symmetry', 0, 1]}


def plots():
    if not os.path.exists(f'{path}/analysis/{comparison}'):
        os.makedirs(f'{path}/analysis/{comparison}')

    df_inner = pandas.read_csv(f'{path}/analysis/{comparison}/df_inner.csv')
    df_outer = pandas.read_csv(f'{path}/analysis/{comparison}/df_outer.csv')

    plot_lines(df_outer)
    plot_boxes(df_inner)
   

def plot_lines(df_outer):

    print('plotting lines...')

    for measure in measures.keys():

        font = {'font.size': 20}
        plt.rcParams.update(font)
        fig, ax = plt.subplots()
        ax.spines['top'].set_color('grey')

        plt.xlabel('')
        plt.ylabel(f'{measures[measure][0]}')
        ax.grid(False)
        ax.spines['top'].set_color('grey')
        ax.spines['top'].set_linewidth(0.5)
        ax.spines['right'].set_color('grey')
        ax.spines['right'].set_linewidth(0.5)
        ax.spines['bottom'].set_color('grey')
        ax.spines['bottom'].set_linewidth(0.5)
        ax.spines['left'].set_color('grey')
        ax.spines['left'].set_linewidth(0.5)
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)
        ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=True)

        for idx_experiment, experiment in enumerate(experiments):
            data = df_outer[(df_outer['experiment'] == experiment)]

            ax.plot(data['generation_index'], data[f'{measure}_{inner_metrics[0]}_median'],
                    label=f'{experiment}_{inner_metrics[0]}', c=clrs[idx_experiment], linewidth=3)

            ax.fill_between(data['generation_index'],
                            data[f'{measure}_{inner_metrics[0]}_q25'],
                            data[f'{measure}_{inner_metrics[0]}_q75'],
                            alpha=0.4, facecolor=clrs[idx_experiment])

            ax.plot(data['generation_index'],
                    data[f'{measure}_{inner_metrics[0]}_q25'],
                    linestyle='-', color=clrs[idx_experiment], alpha=0.5, linewidth=1.5)  # Contour line for Q1

            ax.plot(data['generation_index'],
                    data[f'{measure}_{inner_metrics[0]}_q75'],
                    linestyle='-', color=clrs[idx_experiment], alpha=0.5, linewidth=1.5)  # Contour line for Q3

            if include_max:
                ax.plot(data['generation_index'], data[f'{measure}_{inner_metrics[1]}_median'],
                        'b--', label=f'{experiment}_{inner_metrics[1]}', c=clrs[idx_experiment])
                ax.fill_between(data['generation_index'],
                                data[f'{measure}_{inner_metrics[1]}_q25'],
                                data[f'{measure}_{inner_metrics[1]}_q75'],
                                alpha=0.3, facecolor=clrs[idx_experiment])

            # if measures[measure][1] != -math.inf and measures[measure][2] != -math.inf:
            #     ax.set_ylim(measures[measure][1], measures[measure][2])

            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),  fancybox=True, shadow=True, ncol=5, fontsize=10)
            if not merge_lines:
                plt.savefig(f'{path}/analysis/{comparison}/line_{experiment}_{measure}.png', bbox_inches='tight')
                plt.clf()
                plt.close(fig)
                plt.rcParams.update(font)
                fig, ax = plt.subplots()

        if merge_lines:
            plt.savefig(f'{path}/analysis/{comparison}/line_{measure}.png', bbox_inches='tight')
            plt.clf()
            plt.close(fig)

    print('plotted lines!')


def plot_boxes(df_inner):
    print('plotting boxes...')

    for gen_boxes in gens_boxes:

        df_inner2 = df_inner[(df_inner['generation_index'] == gen_boxes)]
        #TODO: move this to otherstudies as an alterne copy. also include the tiny heads exclusion
        # df_inner2 = df_inner2[(df_inner2['experiment'] == "reg2m2")
        #                      | ((df_inner2['experiment'] == "reg10m2")
        #                         & (~df_inner2['run'].isin([13, 17, 23, 29, 18,  6,26, 16]) ) )]

        plt.clf()
        tests_combinations = [(experiments[i], experiments[j]) \
                              for i in range(len(experiments)) for j in range(i+1, len(experiments))]
        for idx_measure, measure in enumerate(measures.keys()):
            sb.set(rc={"axes.titlesize": 23, "axes.labelsize": 23, 'ytick.labelsize': 21, 'xtick.labelsize': 21})
            sb.set_style("whitegrid")

            plot = sb.boxplot(x='experiment', y=f'{measure}_{inner_metrics[0]}', data=df_inner2,
                              palette=clrs, width=0.4, showmeans=True, linewidth=2, fliersize=6,
                              meanprops={"marker": "o", "markerfacecolor": "yellow", "markersize": "12"})
            plot.tick_params(axis='x', labelrotation=10)

            # TODO: fix this later / test for normality and then choose test. show test name and p value at the top
            # try:
            #     if len(tests_combinations) > 0:
            #
            #         add_stat_annotation(plot, data=df_inner2, x='experiment', y=f'{measure}_{inner_metrics[0]}',
            #                             box_pairs=tests_combinations,
            #                             comparisons_correction=None,
            #                             test='Wilcoxon',
            #                             text_format = 'star', fontsize = 'xx-large', loc = 'inside',
            #                             verbose=1)
            # except AttributeError as error:
            #     print('stat test fail',error)

            # Calculate Wilcoxon test statistics for each box pair
            print('\n\n',measure, tests_combinations)
            for pair in tests_combinations:

                group1_data = df_inner2[df_inner2['experiment'] == pair[0]][f'{measure}_{inner_metrics[0]}']
                group2_data = df_inner2[df_inner2['experiment'] == pair[1]][f'{measure}_{inner_metrics[0]}']
                _, p_value = mannwhitneyu(group1_data, group2_data, alternative='two-sided')

                print(group1_data.median(),group2_data.median(),p_value)

                # Add text annotation for p-value
                x_pos = tests_combinations.index(pair)
                plot.text(x_pos, max(group1_data.max(), group2_data.max()) + 0.1, f'p={p_value:.4f}', ha='center')

            # if measures[measure][1] != -math.inf and measures[measure][2] != -math.inf:
            #     plot.set_ylim(measures[measure][1], measures[measure][2])
            plt.xlabel('')
            plt.ylabel(f'{measures[measure][0]}')
            plot.get_figure().savefig(f'{path}/analysis/{comparison}/box_{measure}_{gen_boxes}.png', bbox_inches='tight')
            plt.clf()
            plt.close()

    print('plotted boxes!')


plots()




