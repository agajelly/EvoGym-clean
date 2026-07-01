"""
Visualize and run a modular robot using Mujoco.
"""

from pyrr import Quaternion, Vector3
import argparse
from revolve2.actor_controller import ActorController
from revolve2.core.physics.running import ActorControl, Batch, Environment, PosedActor

from sqlalchemy.ext.asyncio.session import AsyncSession
from revolve2.core.database import open_async_database_sqlite
from sqlalchemy.future import select
from revolve2.core.optimization.ea.generic_ea import DbEAOptimizerGeneration, DbEAOptimizerIndividual, DbEAOptimizer, DbEnvconditions
from genotype import GenotypeSerializer, develop_knockout
from optimizer import DbOptimizerState
import sys
from revolve2.core.modular_robot.render.render import Render
from revolve2.core.modular_robot import Measure
from revolve2.core.database.serializers import DbFloat
import pprint
import numpy as np
import os
from ast import literal_eval
import math

from revolve2.runners.isaacgym import LocalRunner as LocalRunnerI


from body_spider import *


class Simulator:
    _controller: ActorController

    async def simulate(self) -> None:

        parser = argparse.ArgumentParser()
        parser.add_argument("study")
        parser.add_argument("experiments")
        parser.add_argument("tfs")
        parser.add_argument("watchruns")
        parser.add_argument("generations")
        parser.add_argument("mainpath")

        args = parser.parse_args()

        self.study = args.study
        self.experiments_name = args.experiments.split(',')
        self.tfs = list(args.tfs.split(','))
        self.runs = args.watchruns.split(',')
        self.generations = [0, 100]
        test_robots = []
        self.mainpath = args.mainpath

        self.bests = 1
        # 'all' selects best from all individuals
        # 'gens' selects best from chosen generations
        self.bests_type = 'gens'
        self.ranking = ['best', 'worst']

        path = f'{self.mainpath}/{self.study}/analysis/knockouts/'
        if not os.path.exists(path):
            os.makedirs(path)

        self.pfile = f'{self.mainpath}/{self.study}/analysis/knockouts/knockouts_measures.csv'
        header = ['experiment_name', 'run', 'gen', 'ranking', 'individual_id', 'geno_size', 'n_genes', 'knockout', 'distance',
                  'birth'  ,  'displacement'  , 'disp_y', 'speed_y'   , 'speed_x'  ,  'average_z'   , 'head_balance' ,   'hinge_count',    'brick_count',
                  'hinge_ratio' ,   'hinge_horizontal'  ,  'hinge_vertical' ,   'modules_count'  ,  'hinge_prop'  ,  'brick_prop',
                  'branching_count'  ,  'branching_prop'   , 'extensiveness'  ,  'extremities' ,   'extremities_prop',
                  'extensiveness_prop'   , 'width'  ,  'height'   , 'coverage'  ,  'proportion',    'symmetry'  ,  'relative_speed_y'
                  ]
        with open(self.pfile, 'w') as file:
            file.write(','.join(map(str, header)))
            file.write('\n')

        for ids, experiment_name in enumerate(self.experiments_name):
            print('\n', experiment_name)
            for run in self.runs:
                print('run: ', run)

                path = f'{self.mainpath}/{self.study}'

                fpath = f'{path}/{experiment_name}/run_{run}'
                db = open_async_database_sqlite(fpath)

                if self.bests_type == 'gens':
                    for gen in self.generations:
                        print('  gen: ', gen)
                        await self.recover(db, gen, path, test_robots, self.tfs[ids], experiment_name, run)
                elif self.bests_type == 'all':
                    pass
                    # TODO: implement

    async def recover(self, db, gen, path, test_robots, tfs, experiment_name, run):
        async with AsyncSession(db) as session:

            rows = (
                (await session.execute(select(DbEAOptimizer))).all()
            )
            max_modules = rows[0].DbEAOptimizer.max_modules
            substrate_radius = rows[0].DbEAOptimizer.substrate_radius
            plastic_body = rows[0].DbEAOptimizer.plastic_body
            plastic_brain = rows[0].DbEAOptimizer.plastic_brain

            rows = (
                (await session.execute(select(DbOptimizerState))).all()
            )
            sampling_frequency = rows[0].DbOptimizerState.sampling_frequency
            control_frequency = rows[0].DbOptimizerState.control_frequency
            simulation_time = rows[0].DbOptimizerState.simulation_time

            rows = ((await session.execute(select(DbEnvconditions))).all())
            env_conditions = {}
            for c_row in rows:
                env_conditions[c_row[0].id] = literal_eval(c_row[0].conditions)

            if self.bests_type == 'all':
                pass

            elif self.bests_type == 'gens':

                for ranking in self.ranking:

                    query = select(DbEAOptimizerGeneration, DbEAOptimizerIndividual, DbFloat) \
                        .filter((DbEAOptimizerGeneration.individual_id == DbEAOptimizerIndividual.individual_id)
                                & (DbEAOptimizerGeneration.env_conditions_id == DbEAOptimizerIndividual.env_conditions_id)
                                & (DbFloat.id == DbEAOptimizerIndividual.float_id)
                                & DbEAOptimizerGeneration.generation_index.in_([gen])
                                )

                    if len(test_robots) > 0:
                        query = query.filter(DbEAOptimizerIndividual.individual_id.in_(test_robots))

                    print(' ', ranking)
                    if ranking == 'best':
                        # if seasonal setup, criteria is seasonal pareto
                        if len(rows) > 1:
                            query = query.order_by(
                                                   # CAN ALSO USE SOME OTHER CRITERIA INSTEAD OF SEASONAL
                                                   DbEAOptimizerGeneration.seasonal_dominated.desc(),
                                                   DbEAOptimizerGeneration.individual_id.asc(),
                                                   DbEAOptimizerGeneration.env_conditions_id.asc())
                        else:
                            query = query.order_by(DbFloat.disp_y.desc())
                    else:
                        if len(rows) > 1:
                            query = query.order_by(
                                                   DbEAOptimizerGeneration.seasonal_dominated.asc(),
                                                   DbEAOptimizerGeneration.individual_id.asc(),
                                                   DbEAOptimizerGeneration.env_conditions_id.asc())
                        else:
                            query = query.order_by(DbFloat.disp_y.asc())

                    rows = ((await session.execute(query)).all())

                    num_lines = self.bests * len(env_conditions)
                    for idx, r in enumerate(rows[0:num_lines]):

                        phenotypes = []
                        data_part1 = []

                        env_conditions_id = r.DbEAOptimizerGeneration.env_conditions_id
                        # print(f'\n  rk:{idx+1} ' \
                        #          f' id:{r.DbEAOptimizerIndividual.individual_id} ' \
                        #          f' birth:{r.DbFloat.birth} ' \
                        #          f' gen:{r.DbEAOptimizerGeneration.generation_index} ' \
                        #          f' cond:{env_conditions_id} ' \
                        #          f' dom:{r.DbEAOptimizerGeneration.seasonal_dominated} ' \
                        #          f' speed_y:{r.DbFloat.speed_y} ' \
                        #          f' disp_y:{r.DbFloat.disp_y} ' \
                        #       )

                        genotype = (
                            await GenotypeSerializer.from_database(
                                session, [r.DbEAOptimizerIndividual.genotype_id]
                            )
                        )[0]
                        geno_size = len(genotype.body.genotype)
                        knockout = None
                        knockstring = 'o'
                        distance = 0

                        original_phenotype, original_substrate, genes = \
                            develop_knockout(genotype, genotype.mapping_seed, max_modules, tfs,
                                             substrate_radius, env_conditions[env_conditions_id],
                                             len(env_conditions), plastic_body, plastic_brain,
                                             knockout)
                        n_genes = len(genes)

                        # original phenotype is added to lists
                        phenotypes.append(original_phenotype)
                        data_part1.append([experiment_name, run, gen, ranking, r.DbEAOptimizerIndividual.individual_id,
                                           geno_size, n_genes, knockstring, distance])

                        # every individual gene and all pairs
                        singles = [[i] for i in range(n_genes)]
                        pairs = [[singles[i][0], singles[j][0]] for i in range(len(singles)) for j in
                                 range(i + 1, len(singles))]
                        knockouts = singles + pairs

                        # all phenotype variations are added to list
                        for knockout in knockouts:
                            knockstring = '.'.join([str(item) for item in knockout])
                            knockout_phenotype, knockout_substrate, genes = \
                                develop_knockout(genotype, genotype.mapping_seed, max_modules, tfs,
                                                 substrate_radius,
                                                 env_conditions[env_conditions_id],
                                                 len(env_conditions), plastic_body, plastic_brain,
                                                 knockout)

                            # # render = Render()
                            # # img_path = f'{self.mainpath}/{self.study}/analysis/knockouts/{experiment_name}_{run}_{gen}_{individual_id}_{knockstring}.png'
                            # # render.render_robot(phenotype.body.core, img_path)

                            distance = self.measure_distance(original_substrate, knockout_substrate)

                            phenotypes.append(knockout_phenotype)
                            data_part1.append([experiment_name, run, gen, ranking, r.DbEAOptimizerIndividual.individual_id,
                                               geno_size, n_genes, knockstring, distance])

                        batch_size = 100
                        num_batches = math.ceil(len(phenotypes)/batch_size)
                        for batch_index in range(0, num_batches):

                            batch_phenotypes = phenotypes[batch_index*batch_size:(batch_index+1)*batch_size]

                            batch = Batch(
                                simulation_time=simulation_time,
                                sampling_frequency=sampling_frequency,
                                control_frequency=control_frequency,
                                control=self._control,
                            )
                            self._controllers = []
                            for phenotype in batch_phenotypes:

                                actor,  controller = phenotype.make_actor_and_controller()
                                bounding_box = actor.calc_aabb()
                                self._controllers.append(controller)

                                env = Environment()
                                x_rotation_degrees = float(env_conditions[env_conditions_id][2])
                                robot_rotation = x_rotation_degrees * np.pi / 180

                                env.actors.append(
                                    PosedActor(
                                        actor,
                                        Vector3(
                                            [
                                                0.0,
                                                0.0,
                                                (bounding_box.size.z / 2.0 - bounding_box.offset.z),
                                            ]
                                        ),
                                        Quaternion.from_eulers([robot_rotation, 0, 0]),
                                        [0.0 for _ in  controller.get_dof_targets()],
                                    )
                                )
                                batch.environments.append(env)

                            runner = LocalRunnerI(LocalRunnerI.SimParams(),
                                headless=True,
                                env_conditions=env_conditions[env_conditions_id],
                                real_time=False,)
                            states = await runner.run_batch(batch)

                            for i, phenotype in enumerate(batch_phenotypes):
                                m = Measure(states=states, genotype_idx=i, phenotype=phenotype,
                                            generation=gen, simulation_time=simulation_time)
                                #pprint.pprint(m.measure_all_non_relative())
                                measures = m.measure_all_non_relative()

                                with open(self.pfile, 'a') as file:
                                    file.write(','.join(map(str, data_part1[batch_index*batch_size+i])))
                                with open(self.pfile, 'a') as file:
                                    for measure in measures:
                                        file.write(',' + str(measures[measure]))
                                with open(self.pfile, 'a') as file:
                                    file.write('\n')

    def _control(self, dt: float, control: ActorControl) -> None:
        for control_i, controller in enumerate(self._controllers):
            controller.step(dt)
            control.set_dof_targets(control_i, 0, controller.get_dof_targets())

    def measure_distance(self, original_substrate, knockout_substrate):

        keys_first = set(original_substrate.keys())
        keys_second = set(knockout_substrate.keys())
        intersection = keys_first & keys_second
        disjunct_first = [a for a in keys_first if a not in intersection]
        disjunct_second = [b for b in keys_second if b not in intersection]
        body_changes = len(disjunct_first) + len(disjunct_second)

        for i in intersection:
            if hasattr(original_substrate[i], '_absolute_rotation'):
                if type(original_substrate[i]) != type(knockout_substrate[i]) or \
                    original_substrate[i]._absolute_rotation != knockout_substrate[i]._absolute_rotation :
                    body_changes += 1

        return body_changes


async def main() -> None:

    sim = Simulator()
    await sim.simulate()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())



