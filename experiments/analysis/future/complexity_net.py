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
from revolve2.genotypes.cppnwin.modular_robot.geno_body_GRN_v3 import GRN
from optimizer import DbOptimizerState
import sys
from revolve2.core.database.serializers import DbFloat
import pprint
import numpy as np
import os
from ast import literal_eval
import math
import asyncio
import networkx as nx
import matplotlib.pyplot as plt


class Complexity:
    async def run(self) -> None:

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
        self.generations = list(range(0, 101))
        test_robots = []
        self.mainpath = args.mainpath

        self.bests = 100
        # 'all' selects best from all individuals
        # 'gens' selects best from chosen generations
        self.bests_type = 'gens'
        self.ranking = ['best'] # obsolete var

        self.path = f'{self.mainpath}/{self.study}/analysis/complexity/'

        if not os.path.exists(self.path):
            os.makedirs(self.path)
        if not os.path.exists(f'{self.path}/comp_nets/'):
            os.makedirs(f'{self.path}/comp_nets/')

        self.pfile = f'{self.path}/complexity_net.csv'
        header = ['experiment_name', 'run', 'gen', 'individual_id', 'complexity_net', 'zero_regulators', 'geno_size', 'n_genes']
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

                        grn = GRN(max_modules, tfs, genotype.body, genotype.mapping_seed,
                                    env_conditions[env_conditions_id], len(env_conditions), plastic_body)

                        connections, numbers_regulators = grn.net_parser()
                        zero_regulators = numbers_regulators.count(0)

                        num_connections = len(connections)
                        n_genes = len(grn.genes)

                        if num_connections <= 50 and gen == 100:

                            G = nx.DiGraph()

                            # Add edges to the graph
                            G.add_edges_from(connections)

                            # Draw the network
                            plt.figure(figsize=(12, 10))
                            pos = nx.spring_layout(G, seed=42)  # Position nodes using the spring layout

                            # Draw the nodes
                            nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='lightblue', edgecolors='black')

                            # Draw the edges with arrows
                            nx.draw_networkx_edges(
                                G, pos,
                                edgelist=connections,
                                arrowstyle='-|>',  # Arrow style
                                arrowsize=50,  # Size of arrows
                                edge_color='gray',  # Color of edges
                                style='dashed',  # Style of edges for better visibility
                                connectionstyle='arc3,rad=0.1'  # Arc curvature
                            )

                            # Draw the labels
                            nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif')

                            # Set title and display the plot
                            plt.title('Gene Regulatory Network')
                            plt.axis('off')  # Turn off the axis

                            plt.savefig(f'{self.path}/comp_nets/{experiment_name}_{run}_{gen}_{r.DbEAOptimizerIndividual.individual_id}_{num_connections}.png')
                            plt.close()

                        with open(f'{self.pfile}', 'a') as f:
                            f.write(
                                f"{experiment_name},{run},{gen},"
                                f"{r.DbEAOptimizerIndividual.individual_id},{num_connections}, {zero_regulators},{geno_size},{n_genes}\n")


asyncio.run(Complexity().run())



