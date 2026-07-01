VOXEL_TYPES = {
    'bone': 1,
    'fat': 2,
    'phase_muscle': 3,
    'offphase_muscle': 4,
} #  matches materials order in prepare_robot_files.py

TF_WEIGHTS = {
    'bone': 4,
    'fat': 4,
    'phase_muscle': 6.0,
    'regulatory': 0.4,   # applies per regulatory TF
}

VOXEL_TYPES_COLORS = {
    'bone': (240, 235, 220), # ice
    'fat': (250, 220, 125), # yellow
    'phase_muscle': (120, 20, 20), # dark red
    'offphase_muscle': (240, 120, 120), # light red
}

####

VOXEL_TYPES_NOBONE = {
    'fat': 1,
    'fat2': 2,
    'phase_muscle': 3,
    'offphase_muscle': 4,
} #  matches materials order in prepare_robot_files.py

TF_WEIGHTS_NOBONE = {
    'fat': 4,
    'fat2': 4,
    'phase_muscle': 6.0,
    'regulatory': 0.4,   # applies per regulatory TF
}

VOXEL_TYPES_COLORS_NOBONE = {
    'fat': (250, 220, 125), # ice
    'fat2': (250, 220, 125), # yellow
    'phase_muscle': (120, 20, 20), # dark red
    'offphase_muscle': (240, 120, 120), # light red
}