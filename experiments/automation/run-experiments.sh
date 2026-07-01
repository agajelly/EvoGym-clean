#!/usr/bin/env bash
# Run from repo root:
#   ./experiments/automation/run-experiments.sh path/to/PARAMS.sh
set -euo pipefail

params_file=${1:-experiments/locomotion.sh}
source "$params_file"

# Defaults
: "${evogym_steps:=500}"
: "${evogym_num_workers:=0}"
: "${evogym_headless:=1}"
: "${evogym_render_mode:=screen}"
: "${RUN_ANALYSIS:=1}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

mkdir -p "${out_path}/${study_name}" "${out_path}/${study_name}/analysis"

IFS=',' read -r -a EXP_LIST <<< "${experiments}"
IFS=',' read -r -a voxel_types_LIST <<< "${voxel_types}"
IFS=',' read -r -a ustatic_LIST <<< "${ustatic}"
IFS=',' read -r -a udynamic_LIST <<< "${udynamic}"
IFS=',' read -r -a COND_LIST <<< "${env_conditions}"
IFS=',' read -r -a RUN_LIST <<< "${runs}"

if [[ ${#EXP_LIST[@]} -ne ${#voxel_types_LIST[@]} || \
      ${#EXP_LIST[@]} -ne ${#COND_LIST[@]} || \
      ${#EXP_LIST[@]} -ne ${#ustatic_LIST[@]} || \
      ${#EXP_LIST[@]} -ne ${#udynamic_LIST[@]} ]]; then
  echo "Error: experiments, voxel_types, env_conditions, ustatic, udynamic must have same length."
  exit 1
fi

echo ">> Parallelism policy: single run at a time; intra-run CPU parallelism via --evogym_num_workers"

for idx in "${!EXP_LIST[@]}"; do
  exp="${EXP_LIST[$idx]}"
  voxel_type="${voxel_types_LIST[$idx]}"
  ustatic_v="${ustatic_LIST[$idx]}"
  udynamic_v="${udynamic_LIST[$idx]}"
  cond="${COND_LIST[$idx]}"

  for run in "${RUN_LIST[@]}"; do
    logfile="${out_path}/${study_name}/${exp}_${run}.log"
    echo ">> Running experiment=${exp} run=${run}  (log: ${logfile})"

    cmd=(
      python3 -u "${REPO_ROOT}/algorithms/${algorithm}.py"
      --out_path "${out_path}"
      --experiment_name "${exp}"
      --env_conditions "${cond}"
      --run "${run}"
      --study_name "${study_name}"
      --algorithm "${algorithm}"
      --fitness_metric "${fitness_metric}"
      --num_generations "${num_generations}"
      --population_size "${population_size}"
      --offspring_size "${offspring_size}"
      --evogym_steps "${evogym_steps}"
      --evogym_num_workers "${evogym_num_workers}"
      --evogym_headless "${evogym_headless}"
      --evogym_render_mode "${evogym_render_mode}"
      --plastic "${plastic}"
      --crossover_prob "${crossover_prob}"
      --mutation_prob "${mutation_prob}"
      --max_voxels "${max_voxels}"
      --voxel_types "${voxel_type}"
      --udynamic "${udynamic_v}"
      --ustatic "${ustatic_v}"
      --cube_face_size "${cube_face_size}"
      --run_simulation "${run_simulation}"
    )

    mkdir -p "${out_path}/${study_name}"
    "${cmd[@]}" >>"${logfile}" 2>&1
  done
done

if [[ "${RUN_ANALYSIS}" -eq 1 ]]; then
  echo ">> All runs finished. Starting analysis..."
  "${REPO_ROOT}/experiments/automation/run-analysis.sh" "$params_file"
fi
