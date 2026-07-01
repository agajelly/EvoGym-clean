import os
import sys
import numpy as np
from pathlib import Path
import shutil
import time

# make repository folder the root
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from algorithms.experiment import Experiment
from algorithms.EA_classes import Individual
from algorithms.GRN_2D import GRN, initialization, mutation_type1, unequal_crossover_prop
from simulation.simulation_resources import simulate_evogym_batch
from simulation.prepare_robot_files import prepare_robot_files
from utils.metrics import genopheno_abs_metrics, behavior_abs_metrics, relative_metrics
from utils.config import Config


# Simple non-standard EA:
# uses tournaments for parent selection
# creates a pool (m+l) and does survival selection with tournaments
class EA(Experiment):
    def __init__(self, args=None):
        # Allow instantiation-inject args OR fallback to config-inject
        self.args =  Config()._get_params()

        super().__init__(self.args)  # sets out_path, DB, session, rng, id_counter

        # experiment-level params used by EA logic
        self.MAX_GENOME_SIZE = 1000
        self.INI_GENOME_SIZE = 300
        self.PROMOTOR_THRESHOLD = 0.95

        self.novelty_archive = [] # TODO: include in recovery
        self.archive_add_frac = 0.05

        self.cube_face_size = self.args.cube_face_size
        self.max_voxels = self.args.max_voxels
        self.voxel_types = self.args.voxel_types
        self.plastic = self.args.plastic
        self.env_conditions = self.args.env_conditions
        self.population_size = self.args.population_size
        self.offspring_size = self.args.offspring_size
        self.crossover_prob = self.args.crossover_prob
        self.mutation_prob = self.args.mutation_prob
        self.tournament_k = self.args.tournament_k
        self.num_generations = self.args.num_generations
        self.fitness_metric = self.args.fitness_metric
        # keep top-N by displacement each generation (0 disables)
        self.elitism = getattr(self.args, "elitism", 3)
        self.ustatic = self.args.ustatic
        self.udynamic = self.args.udynamic

    # ---------- EA-specific utilities ----------

    def develop_phenotype(self, genome, voxel_types):
        phenotype = GRN(
            promoter_threshold=self.PROMOTOR_THRESHOLD,
            max_voxels=self.max_voxels,
            cube_face_size=self.cube_face_size,
            voxel_types=voxel_types,
            genotype=genome,
            env_conditions=self.env_conditions,
            plastic=self.plastic,
        ).develop()

        phenotype_materials = np.zeros(phenotype.shape, dtype=int)
        for index, value in np.ndenumerate(phenotype):
            phenotype_materials[index] = value.voxel_type if value != 0 else 0

        return phenotype_materials

    def initialize_population(self, size, generation):
        individuals = []
        for _ in range(size):
            self.id_counter += 1
            ind= Individual(initialization(self.rng, self.INI_GENOME_SIZE), self.id_counter,
                                                         parent1_id=None, parent2_id=None)
            ind.born_generation = generation
            individuals.append(ind)
        return individuals

    def mutate(self, individual):
        if self.rng.uniform(0, 1) <= self.mutation_prob:
            individual.genome = mutation_type1(self.rng, individual.genome)

    def crossover(self, parent1, parent2):
        if self.rng.uniform(0, 1) <= self.crossover_prob:
            child_genome = unequal_crossover_prop(
                self.rng,
                self.PROMOTOR_THRESHOLD,
                self.MAX_GENOME_SIZE,
                parent1,
                parent2,
            )
        else:
            chosen = self.rng.choice((parent1, parent2))
            child_genome = list(chosen.genome)

        self.id_counter += 1
        child = Individual(child_genome, self.id_counter, parent1_id=parent1.id, parent2_id=parent2.id)
        return child

    def tournament_selection(self, population, k):
        return max(self.rng.sample(population, k), key=lambda ind: ind.fitness)

    # ---------- Main run ----------

    def run(self):

        super().recover_db()

        last_gen, recovered_population = self._recover_state()

        if recovered_population is None:
            # Fresh start
            generation = 1
            population = self.initialize_population(self.population_size, generation)
            self.update_novelty_archive(population)

            for ind in population:
                ind.phenotype = self.develop_phenotype(ind.genome, self.voxel_types)
                genopheno_abs_metrics(ind, self.args)

                if self.args.run_simulation:
                    prepare_robot_files(ind, self.args)

            if self.args.run_simulation:
                simulate_evogym_batch(population, self.args)
    
                for ind in population:
                    behavior_abs_metrics(ind)

            relative_metrics(population, self.args, generation, novelty_archive=self.novelty_archive)

            # persist parents as both robots and survivors for gen 1
            self._persist_generation_atomic(generation, population, population)
            start_gen = generation + 1
            print(f"Finished generation {generation}.")

        else:
            # Continue from the next generation after the last completed one
            population = recovered_population
            start_gen = last_gen + 1
            print(
                f"Recovered last completed generation = {last_gen}, "
                f"population size = {len(population)}, next id = {self.id_counter + 1}"
            )

        for generation in range(start_gen, self.num_generations + 1):
            # Generate offspring
            offspring = []
            for _ in range(self.offspring_size):
                parent1 = self.tournament_selection(population, self.tournament_k)
                co_attempts = 0
                while True and co_attempts < 10: # parents should be distinct individuals
                    parent2 = self.tournament_selection(population, self.tournament_k)
                    if parent2.id != parent1.id:
                        break
                    co_attempts += 1

                child = self.crossover(parent1, parent2)
                child.born_generation = generation
                self.mutate(child)
                offspring.append(child)

                child.phenotype = self.develop_phenotype(child.genome, self.voxel_types)
                genopheno_abs_metrics(child, self.args)
                
                if self.args.run_simulation:
                    prepare_robot_files(child, self.args)

            self.update_novelty_archive(offspring)

            if self.args.run_simulation:
                simulate_evogym_batch(offspring, self.args)

                for ind in offspring:
                    behavior_abs_metrics(ind)

            # Combine parents and offspring into a pool
            pool = population + offspring
            relative_metrics(pool, self.args, generation, novelty_archive=self.novelty_archive)

            # Select next generation (unique winners)
            new_population = []
            pool = pool.copy()
            for _ in range(self.population_size):
                k = min(self.tournament_k, len(pool))
                contestants = self.rng.sample(pool, k)
                winner = max(contestants, key=lambda ind: ind.fitness)
                new_population.append(winner)
                pool.remove(winner)  # ensures uniqueness

            # --- Elitism: keep best displacement ---
            if self.elitism:
                # best individual from full evaluated pool
                elite = max(population + offspring, key=lambda ind: ind.fitness)

                # only inject if not already present
                if elite not in new_population:
                    idx = self.rng.randrange(len(new_population))
                    new_population.pop(idx)
                    new_population.append(elite)

            population = new_population
            relative_metrics(population, self.args, generation, novelty_archive=self.novelty_archive)

            # Persist this generation atomically
            self._persist_generation_atomic(generation, offspring, population)
            print(f"Finished generation {generation}.")

        try:
            self.session.close()
        except Exception:
            pass

        path_robots = f"{self.args.out_path}/{self.args.study_name}/{self.args.experiment_name}/run_{self.args.run}/robots"
        if os.path.exists(path_robots):
            shutil.rmtree(path_robots)

        print("Finished optimizing.")

    def update_novelty_archive(self, individuals):
        k = max(1, int(round(self.archive_add_frac * len(individuals))))
        chosen = self.rng.sample(individuals, k)
        self.novelty_archive.extend(chosen)


if __name__ == "__main__":
    start = time.time()
    EA().run()
    end = time.time()

    elapsed = end - start
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = elapsed % 60
    print(f"\n[RUN-TIME]  {hours}h {minutes}m {seconds:.1f}s")






