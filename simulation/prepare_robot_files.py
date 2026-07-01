import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from evogym import get_full_connectivity


def trim_phenotype_materials(phenotype):
    """
    Trim empty borders from a phenotype and return a 2D grid.
    """
    body = np.asarray(phenotype, dtype=int)

    if body.ndim != 2:
        raise ValueError(f"Expected 2D phenotype, got {body.shape}")

    x_mask = np.any(body != 0, axis=1)
    body = body[x_mask]
    y_mask = np.any(body != 0, axis=0)
    body = body[:, y_mask]
    return body


def _material_maps(voxel_types, args=None):
    """
    Map GRN material IDs -> EvoGym voxel IDs.
    We intentionally map both muscle classes to H_ACT so mechanics are equal;
    phase differences are carried by controller phase offsets.
    """
    EVOGYM = {
        "EMPTY": 0,
        "RIGID": 1,
        "SOFT": 2,
        "H_ACT": 3,
    }

    if voxel_types == "withbone":
        material_to_evogym = {
            0: EVOGYM["EMPTY"],
            1: EVOGYM["RIGID"],
            2: EVOGYM["SOFT"],
            3: EVOGYM["H_ACT"],
            4: EVOGYM["H_ACT"],
        }
    elif voxel_types == "nobone":
        material_to_evogym = {
            0: EVOGYM["EMPTY"],
            1: EVOGYM["SOFT"],
            2: EVOGYM["SOFT"],
            3: EVOGYM["H_ACT"],
            4: EVOGYM["H_ACT"],
        }
    else:
        raise ValueError(f"Unsupported voxel_types: {voxel_types}")

    amp_m3 = float(getattr(args, "evogym_amplitude_m3", 0.4))
    amp_m4 = float(getattr(args, "evogym_amplitude_m4", 0.2))

    period_m3 = int(getattr(args, "evogym_period_m3", 20))
    period_m4 = int(getattr(args, "evogym_period_m4", 40))

    amplitudes = {
        3: amp_m3,
        4: amp_m4,
    }

    periods = {
        3: period_m3,
        4: period_m4,
    }

    return material_to_evogym, amplitudes, periods


def _build_evogym_robot_data(untrimmed_phenotype, args):
    """
    Builds the structure and control maps using global untrimmed genetic coordinates
    first, and then trims them down to size simultaneously to preserve tracking shapes.
    """
    material_to_evogym, amplitudes_by_material, periods_by_material = _material_maps(args.voxel_types, args)

    full_structure = np.vectorize(lambda m: material_to_evogym.get(int(m), 0), otypes=[int])(untrimmed_phenotype)

    rows, cols = full_structure.shape
    wave_freq_x = float(getattr(args, "wave_frequency_x", 0.5))
    wave_freq_y = float(getattr(args, "wave_frequency_y", 0.0))

    full_phase_offsets = np.zeros_like(full_structure, dtype=np.float32)
    full_amplitudes = np.zeros_like(full_structure, dtype=np.float32)
    full_periods = np.ones_like(full_structure, dtype=np.float32)

    # Calculate properties locked tightly to absolute genomic coordinates
    for row in range(rows):
        for col in range(cols):
            if full_structure[row, col] == 3:  # Actuator voxel found
                raw_mat = int(untrimmed_phenotype[row, col])
                base_material_phase = 0.0 if raw_mat == 3 else np.pi

                full_phase_offsets[row, col] = base_material_phase + (col * wave_freq_x) + (row * wave_freq_y)
                full_amplitudes[row, col] = amplitudes_by_material.get(raw_mat, 0.4)
                full_periods[row, col] = periods_by_material.get(raw_mat, 20)


    x_mask = np.any(full_structure != 0, axis=1)
    if not np.any(x_mask):  # Edge-case for entirely empty phenotype
        return np.zeros((1, 1), dtype=np.int32), np.zeros((1, 1), dtype=np.int32), np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1)), {}

    y_mask = np.any(full_structure != 0, axis=0)

    # Crop structural and tracking layouts simultaneously using the same masks
    structure = full_structure[x_mask][:, y_mask].astype(np.int32)
    dense_phase_offsets = full_phase_offsets[x_mask][:, y_mask]
    dense_amplitudes = full_amplitudes[x_mask][:, y_mask]
    dense_periods = full_periods[x_mask][:, y_mask]

    connections = get_full_connectivity(structure).astype(np.int32)

    # 3. Compile clean 1D actuator control vectors from the synchronized trimmed grids
    actuator_phase_offsets = []
    actuator_amplitudes = []
    actuator_periods = []

    t_rows, t_cols = structure.shape
    for r in range(t_rows):
        for c in range(t_cols):
            if structure[r, c] == 3:
                actuator_phase_offsets.append(dense_phase_offsets[r, c])
                actuator_amplitudes.append(dense_amplitudes[r, c])
                actuator_periods.append(dense_periods[r, c])

    controller = {
        "action_bias": float(getattr(args, "evogym_action_bias", 1.0)),
        "action_amplitude": np.array(actuator_amplitudes, dtype=np.float32),
        "period_steps": np.array(actuator_periods, dtype=np.int32),
        "phase_offsets": np.array(actuator_phase_offsets, dtype=np.float32),
    }

    return structure, connections, dense_phase_offsets, dense_amplitudes, dense_periods, controller


def prepare_robot_files(individual, args):
    """
    Main gateway function called by your evolutionary loop.
    Directly targets the untrimmed individual.phenotype to construct assets.
    """
    structure, connections, dense_phases, dense_amps, dense_pers, controller = _build_evogym_robot_data(individual.phenotype, args)

    individual.evogym_structure = structure
    individual.evogym_connections = connections
    individual.evogym_controller = controller

    # Pack the final dense 2D sub-grids tracking metrics back into individual properties
    individual.evogym_phase_offsets = dense_phases
    individual.evogym_amplitudes = dense_amps
    individual.evogym_periods = dense_pers
