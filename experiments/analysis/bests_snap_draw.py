import os
import cv2
import pprint
import math
import argparse
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
from utils.config import Config


def main():
    args = Config()._get_params()

    study = args.study_name
    experiments_name = args.experiments.split(',')
    runs = list(map(int, args.runs.split(',')))
    generations = list(map(int, args.generations.split(',')))
    out_path = args.out_path

    bests = 1
    sort = True
    path_out = f'{out_path}/{study}/analysis/snapshots'

    for gen in generations:
        # TODO: change black background to white
        for experiment_name in experiments_name:
            print(experiment_name)

            horizontal = []
            fit_horizontal = []

            for run in runs:
                print('  run: ', run)
                print('   gen: ', gen)

                path_in = f'{path_out}/{experiment_name}/run_{run}/gen_{gen}'
                lst = os.listdir(path_in)
                lst.sort(key=lambda x: int(x.split('_')[0]))
                lst = lst[0:bests]

                fit_horizontal.append(lst[0]) # best of each run
                for_concats = [cv2.imread(f'{path_in}/{robot}') for robot in lst]
                for l in lst:
                    fit = l.split("_")[1]
                    id = l.split("_")[2]
                    with open(f'{out_path}/{study}/analysis/snapshots/bests.txt', 'a') as f:
                        f.write(f'{experiment_name} {run} {gen} {id}: {fit}\n')

                heights = [o.shape[0] for o in for_concats]
                max_height = max(heights)
                margin = 100

                for idx, c in enumerate(for_concats):
                    if for_concats[idx].shape[0] < max_height:
                        bottom = max_height - for_concats[idx].shape[0] + margin
                    else:
                        bottom = margin

                    for_concats[idx] = cv2.copyMakeBorder(for_concats[idx], margin, math.ceil(bottom), margin,\
                                                           margin, cv2.BORDER_CONSTANT, None, value=[255, 255, 255])

                concats = cv2.hconcat(for_concats)
                horizontal.append(concats)

            # sort by best runs
            sort_aux = ''
            if sort:
                sorted_indices = np.argsort(fit_horizontal)[::-1]
                horizontal = [horizontal[i] for i in sorted_indices]
                sort_aux = 'sort'

            widths = [o.shape[1] for o in horizontal]
            max_width = max(widths)
            for idx, img in enumerate(horizontal):
                if horizontal[idx].shape[1] < max_width:
                    right = max_width - horizontal[idx].shape[1]
                else:
                    right = 0

                horizontal[idx] = cv2.copyMakeBorder(horizontal[idx], 0, margin*3, 0,\
                                                       math.ceil(right), cv2.BORDER_CONSTANT, None, value=(255, 255, 255))

            vertical = cv2.vconcat(horizontal)

            cv2.imwrite(f'{path_out}/bests{sort_aux}_{experiment_name}_gen{gen}.png', vertical)


if __name__ == "__main__":
    main()