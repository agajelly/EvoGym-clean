import argparse


class Config():

    def _get_params(self):
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "--out_path",
            required=False,
            default="tmp_out",
            type=str,
            help="path for results files"
        )

        parser.add_argument(
            "--study_name",
            required=False,
            default="defaultstudy",
            type=str,
            help="",
        )

        parser.add_argument(
            "--experiment_name",
            required=False,
            default="defaultexperiment",
            type=str,
            help="Name of the experiment.",
        )

        parser.add_argument(
            "--algorithm",
            required=False,
            default="basic_EA",
            type=str,
            help="",
        )

        parser.add_argument(
            "--population_size",
            required=False,
            default=10,
            type=int,
        )

        parser.add_argument(
            "--offspring_size",
            required=False,
            default=10,
            type=int,
        )

        parser.add_argument(
            "--num_generations",
            required=False,
            default=10,
            type=int,
        )

        parser.add_argument(
            "--tournament_k",
            required=False,
            default=4,
            type=int,
        )

        parser.add_argument(
            "--max_voxels",
            required=False,
            default=64,
            type=int,
            help="",
        )

        parser.add_argument(
            "--cube_face_size",
            required=False,
            default=4,
            type=int,
            help="",
        )

        parser.add_argument(
            "--voxel_types",
            required=False,
            default="withbone",
            type=str,
            help="list of voxel_types config",
        )

        parser.add_argument(
            "--plastic",
            required=False,
            default=0,
            type=int,
            help="0 is not plastic, 1 is plastic",
        )

        parser.add_argument(
            "--env_conditions",
            required=False,
            default='',
            type=str,
            help="params that define environmental conditions and/or task",
        )

        parser.add_argument(
            "--crossover_prob",
            required=False,
            default=1,
            type=float,
        )

        parser.add_argument(
            "--mutation_prob",
            required=False,
            default=0.9,
            type=float,
        )

        parser.add_argument(
            "--fitness_metric",
            required=False,
            default="displacement",
            type=str,
        )

        parser.add_argument(
            "--generations",
            required=False,
            default="",
            type=str,
            help="list of generations of be analyzed",
        )

        parser.add_argument(
            "--final_gen",
            required=False,
            default="",
            type=str,
            help="last generation to be analyzed"
        )

        parser.add_argument(
            "--experiments",
            required=False,
            default="",
            type=str,
            help="list of experiment_name",
        )

        parser.add_argument(
            "--ustatic",
            required=False,
            default=1,
            type=float,
            help="static friction"
        )

        parser.add_argument(
            "--udynamic",
            required=False,
            default=0.8,
            type=float,
            help="dynamic friction"
        )

        parser.add_argument(
            "--run",
            required=False,
            default=1,
            type=int,
            help="",
        )

        parser.add_argument(
            "--runs",
            required=False,
            default="",
            type=str,
            help="list of all runs",
        )

        parser.add_argument(
            "--run_simulation",
            required=False,
            default=1,
            type=int,
            help="If 0, runs optimizer without simulating robots, so behavioral measures are none."
        )

        parser.add_argument(
            "--evogym_num_workers",
            required=False,
            default=1,
            type=int,
            help="EvoGym batch workers. 0=auto based on machine CPU."
        )

        parser.add_argument(
            "--evogym_steps",
            required=False,
            default=500,
            type=int,
            help="Physics steps per robot evaluation in EvoGym."
        )

        parser.add_argument(
            "--evogym_init_x",
            required=False,
            default=3,
            type=int,
            help="Initial robot x position in EvoGym world."
        )

        parser.add_argument(
            "--evogym_init_y",
            required=False,
            default=1,
            type=int,
            help="Initial robot y position in EvoGym world."
        )

        parser.add_argument(
            "--evogym_action_bias",
            required=False,
            default=1.0,
            type=float,
            help="Center value for actuator sine controller."
        )

        parser.add_argument(
            "--evogym_action_amplitude",
            required=False,
            default=0.4,
            type=float,
            help="Amplitude for actuator sine controller."
        )

        parser.add_argument(
            "--evogym_period_steps",
            required=False,
            default=20,
            type=int,
            help="Sine period in simulation steps."
        )

        parser.add_argument(
            "--evogym_headless",
            required=False,
            default=1,
            type=int,
            help="1=headless (default), 0=render simulation window for debugging."
        )

        parser.add_argument(
            "--evogym_render_mode",
            required=False,
            default="screen",
            type=str,
            help="Render mode when evogym_headless=0 (screen or human)."
        )

        parser.add_argument('--tfs', type=int, default=1, help='Time-frequency scaling factor or transformation flag')
             
        args = parser.parse_args()

        return args
