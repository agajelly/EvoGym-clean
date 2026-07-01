import numpy as np
import sys
from pathlib import Path
import math
from sklearn.neighbors import KDTree
from scipy.ndimage import label, generic_filter # for tracking new components in the body for the morpological metrics

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from algorithms.voxel_types import VOXEL_TYPES, VOXEL_TYPES_NOBONE

METRICS_ABS = [
    # genotypic
    "genome_size",

    # behavioral
    "displacement",

    # phenotypic
    "num_voxels",
    "bone_count",
    "bone_prop",
    "fat_count",
    "fat_prop",
    "fat2_count",
    "fat2_prop",
    "phase_muscle_count",
    "phase_muscle_prop",
    "offphase_muscle_count",
    "offphase_muscle_prop",

    # morphological metrics added 
    "aspect_ratio",
    "material_heterogeneity",
    "max_limb_length",
    "symmetry_score",
]

METRICS_REL = [
                "uniqueness",
                "fitness",
                "age",
                # "dominated_disp_age",
                "dominated_disp_nov",
                "novelty",
                "novelty_weighted"
               ]

# metrics relative to other individuals or factors like time
def relative_metrics(population, args, generation, novelty_archive=None):
    uniqueness(population)
    novelty(population, novelty_archive)
    novelty_weighted(population)
    age(population, generation)
    # pareto_dominance_count( population,
    #                         objectives=(("age", "min"), ("displacement", "max")), out_attr="dominated_disp_age")
    pareto_dominance_count(population,
                           objectives=(("novelty", "max"), ("displacement", "max")), out_attr="dominated_disp_nov")
    set_fitness(population, args.fitness_metric)


def genopheno_abs_metrics(individual, args):

    # genome
    genome_size(individual)

    # phenotype
    num_voxels(individual)
    update_material_metrics(individual, args)
    update_morphological_metrics(individual) # new morphological metrics added
    test_validity(individual)


def behavior_abs_metrics(population):
    # as center-of-mass displacement in meters
    # TODO: implement others and treat for -inf
    pass

def update_material_metrics(individual, args):
    if args.voxel_types == 'withbone':
        voxel_types = VOXEL_TYPES
    if args.voxel_types == 'nobone':
        voxel_types = VOXEL_TYPES_NOBONE

    grid = np.asarray(individual.phenotype, dtype=int)
    filled_total = int((grid != 0).sum())
    individual.filled_total = filled_total

    for name, mid in voxel_types.items():

        count = int((grid == mid).sum())
        prop = (count / filled_total) if filled_total > 0 else 0.0

        setattr(individual, f"{name}_count", count)
        setattr(individual, f"{name}_prop", round(prop,2))

def update_morphological_metrics(individual):
    grid = np.asarray(individual.phenotype, dtype=int)
    solid_mask = (grid > 0)

    if not np.any(solid_mask):
        # Empty structure; set default metrics
        individual.aspect_ratio = 1.0
        individual.material_heterogeneity = 0.0
        individual.max_limb_length = 0.0
        individual.symmetry_score = 1.0
        return
    
    #1 = aspect ratio
    rows, cols = grid.shape
    individual.aspect_ratio = round(float(cols / rows), 3) if rows > 0 else 1.0

    #2 = material heterogeneity
    total_voxels = int(grid.size)  # Changed from .shape to .size to return an integer count
    unique_materials = np.unique(grid)
    individual.material_heterogeneity = round(float(len(unique_materials) / float(total_voxels)), 3) if total_voxels > 0 else 0.0

    #3 = leg length (max distance from center of mass to any voxel)
    row_counts = np.sum(solid_mask, axis=1)
    torso_row = int(np.argmax(row_counts))  # Row with the most solid voxels

    legs_zone = solid_mask[torso_row + 1:, :]

    if np.any(legs_zone):
        labeled_legs, num_legs = label(legs_zone)

        if num_legs > 0:
            leg_lengths = []
            for limb_label in range(1, num_legs + 1):
                limb_coords = np.argwhere(labeled_legs == limb_label)
                limb_length = np.max(limb_coords[:, 0]) + 1  # +1 to convert from index to length
                leg_lengths.append(limb_length)

            individual.max_limb_length = float(np.max(leg_lengths))
        else:
            individual.max_limb_length = 0.0
    else:
        individual.max_limb_length = 0.0

    #4 = symmetry score (proportion of voxels that match when flipped horizontally)
    flipped = np.fliplr(grid)
    matching = np.sum(grid == flipped)
    individual.symmetry_score = round(float(matching) / float(grid.size), 3) if grid.size > 0 else 1.0

def set_fitness(population, fitness_metric):
    
    for ind in population:
        displacement = float(getattr(ind, "displacement", None))

        if fitness_metric == "displacement":
            ind.fitness = displacement
        
        elif fitness_metric == "novelty":
            novelty = float(getattr(ind, "novelty", 0.0))
            ind.fitness = displacement * novelty

        elif fitness_metric == "symmetry":
            symmetry = float(getattr(ind, "symmetry_score", 1.0))
            ind.fitness = displacement * symmetry

        # fallback to default (displacement) if metric not found
        else:
            val = getattr(ind, fitness_metric, None)
            ind.fitness = float(val) if val is not None else displacement 

def test_validity(individual):
    # 1. Existing muscle checks
    has_any_muscle = (individual.phase_muscle_count + individual.offphase_muscle_count) >= 1
    
    # 2. NEW: Check for disconnected floating components
    grid = np.asarray(individual.phenotype, dtype=int)
    solid_mask = (grid > 0)
    
    if np.any(solid_mask):
        # label() counts isolated structural islands
        _, num_features = label(solid_mask)
        is_fully_connected = (num_features == 1)
    else:
        is_fully_connected = False

    # The robot is only valid if it has muscles AND is a single connected piece!
    individual.valid = has_any_muscle and is_fully_connected

def age(population, generation):
    for ind in population:
        age = generation - ind.born_generation + 1
        ind.age = age

def genome_size(individual):
    individual.genome_size = len(individual.genome)


def num_voxels(individual):  # size / mass proxy
    individual.num_voxels = int((individual.phenotype != 0).sum())

def distance(g1, g2):
   #similar to hamming
    a = np.asarray(g1)
    b = np.asarray(g2)

    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    one_zero = (a == 0) ^ (b == 0)  # 0 vs non-zero → 1.0 (different shape)
    both_nonzero_diff = (a != 0) & (b != 0) & (a != b)  # non-zero vs different non-zero → 0.5 (different material)

    return float(one_zero.sum() + 0.5 * both_nonzero_diff.sum())


def uniqueness(population):
    # average morphological distance to all current pop using Hamming distance
    for i, ind in enumerate(population):
        distances = []
        for j, other in enumerate(population):
            if i != j:
                d = distance(ind.phenotype, other.phenotype)
                distances.append(d / max(ind.num_voxels, other.num_voxels))
        ind.uniqueness = np.mean(distances)

# def novelty_weighted(population):
#     for ind in population:
#         novelty_weighted = ind.displacement * ind.novelty
#         ind.novelty_weighted = novelty_weighted

def novelty_weighted(population):
    beta = 0.05
    for ind in population:
        novelty_weighted = ind.displacement * ind.novelty + beta * ind.displacement
        ind.novelty_weighted = novelty_weighted

def novelty(population, novelty_archive, k=5, M=50, embed_fn=None):
    pool = list(population) + list(novelty_archive or [])

    if embed_fn is None:
        # minimal embedding: 1D vector
        embed_fn = lambda ind: np.array([ind.num_voxels], dtype=np.float32)

    X = np.vstack([embed_fn(ind) for ind in pool]).astype(np.float32)
    tree = KDTree(X)

    for ind in population:
        qi = embed_fn(ind).reshape(1, -1)
        _, idxs = tree.query(qi, k=min(M + 1, len(pool)))
        idxs = idxs[0]

        dists = []
        for j in idxs:
            other = pool[j]
            if other is ind:
                continue
            d = distance(ind.phenotype, other.phenotype)
            dists.append(d / max(ind.num_voxels, other.num_voxels))

        kk = min(k, len(dists))
        ind.novelty = float(np.partition(np.asarray(dists, dtype=np.float32), kk - 1)[:kk].mean()) if kk else 0.0


def pareto_dominance_count(
    population,
    objectives=(("age", "min"), ("displacement", "max")),
    out_attr="dominates_count",
):
    """
    For each individual, count how many others it Pareto-dominates
    Dominance rule:
      A dominates B iff
        - A is no worse than B in all objectives, AND
        - A is strictly better in at least one objective.
    """
    # Normalize directions and validate
    obj_specs = []
    for attr, direction in objectives:
        d = direction.strip().lower()
        obj_specs.append((attr, d))

    def dominates(a, b) -> bool:
        no_worse_all = True
        strictly_better_any = False

        for attr, d in obj_specs:
            av = getattr(a, attr)
            bv = getattr(b, attr)

            if d == "min":
                if av > bv:
                    no_worse_all = False
                    break
                if av < bv:
                    strictly_better_any = True
            else:  # "max"
                if av < bv:
                    no_worse_all = False
                    break
                if av > bv:
                    strictly_better_any = True

        return no_worse_all and strictly_better_any

    # Init output
    for ind in population:
        setattr(ind, out_attr, 0)

    # O(n^2) dominance counting
    n = len(population)
    for i in range(n):
        a = population[i]
        cnt = 0
        for j in range(n):
            if i == j:
                continue
            if dominates(a, population[j]):
                cnt += 1
        setattr(a, out_attr, cnt)



