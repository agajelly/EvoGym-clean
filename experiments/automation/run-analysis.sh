#!/bin/bash

# run this script from the repo root:
# ./experiments/automation/run-analysis.sh path/to/PARAMSFILE.sh

if [ $# -eq 0 ]
  then
    params_file="experiments/locomotion.sh"
  else
    params_file=$1
fi

set -a
source "$params_file"
set +a

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

python3 "${REPO_ROOT}/experiments/analysis/consolidate.py" \
 --study_name "$study_name" \
 --experiments "$experiments" \
 --runs "$runs" \
 --out_path "$out_path" \
 --final_gen "$final_gen";



papermill "experiments/analysis/analysis.ipynb" \
          "experiments/analysis/analysis-executed.ipynb" \
          -p study_name  "$study_name" \
          -p experiments "$experiments" \
          -p runs "$runs" \
          -p voxel_types "$voxel_types" \
          -p generations "$generations" \
          -p final_gen "$final_gen" \
          -p out_path "$out_path"


python3 "${REPO_ROOT}/experiments/analysis/snapshots_bests.py" \
  --study_name "$study_name" \
  --experiments "$experiments" \
  --voxel_types "$voxel_types" \
  --runs "$runs" \
  --generations "$generations" \
  --out_path "$out_path" \
  --max_voxels "$max_voxels" \
  --cube_face_size "$cube_face_size" \
  --env_conditions "$env_conditions" \
  --algorithm "$algorithm" \
  --plastic "$plastic"

#
#python3 "${REPO_ROOT}/experiments/analysis/bests_snap_draw.py" \
#  --study_name "$study_name" \
#  --experiments "$experiments" \
#  --runs "$runs" \
#  --generations "$generations" \
#  --out_path "$out_path"


#python3 "${REPO_ROOT}/experiments/analysis/family_tree.py" \
#  --study_name "$study_name" \
#  --experiments "$experiments" \
#  --voxel_types "$voxel_types" \
#  --runs "$runs" \
#  --generations "$generations" \
#  --out_path "$out_path" \
#  --max_voxels "$max_voxels" \
#  --cube_face_size "$cube_face_size" \
#  --env_conditions "$env_conditions" \
#  --algorithm "$algorithm" \
#  --plastic "$plastic"
