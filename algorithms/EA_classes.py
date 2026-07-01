from sqlalchemy.orm import declarative_base
from sqlalchemy import (
    create_engine, Column, Integer, Float, String, JSON, ForeignKey, UniqueConstraint,
    PrimaryKeyConstraint
)
from math import inf
from pathlib import Path
import sys

Base = declarative_base()

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from utils.metrics import METRICS_ABS, METRICS_REL


# DB CLASSES


class ExperimentInfo(Base):
    __tablename__ = "experiment_info"
    id = Column(Integer, primary_key=True, autoincrement=True)
    seed = Column(Integer, nullable=False)


def build_robot_class():
    # Base attributes (non-dynamic)
    attrs = {
        "__tablename__": "all_robots",
        "robot_id": Column(Integer, primary_key=True),
        "born_generation": Column(Integer, nullable=False),
        "genome": Column(JSON, nullable=False),
        "valid": Column(Float, default=0.0),

        "parent1_id": Column(Integer, ForeignKey("all_robots.robot_id"), nullable=True),
        "parent2_id": Column(Integer, ForeignKey("all_robots.robot_id"), nullable=True),
    }
    # Dynamic fields
    for m in METRICS_ABS:
        attrs[m] = Column(Float)
    return type("Robot", (Base,), attrs)
# Instantiate the Robot class
Robot = build_robot_class()


def build_generation_survivor_class():
    attrs = {
        "__tablename__": "generation_survivors",

        # Base attributes (non-dynamic)
        "generation": Column(Integer, nullable=False),
        "robot_id": Column(Integer, ForeignKey("all_robots.robot_id"), nullable=False),

        "__table_args__": (
            PrimaryKeyConstraint("generation", "robot_id", name="pk_generation_robot"),
        ),
    }

    # Dynamic fields
    for m in METRICS_REL:
        attrs[m] = Column(Float, default=0.0)
    return type("GenerationSurvivor", (Base,), attrs)
# Instantiate the GenerationSurvivor class
GenerationSurvivor = build_generation_survivor_class()

# EA CLASSES


class Individual:
    def __init__(self, genome, id_counter, parent1_id=None, parent2_id=None):
        self.id = id_counter
        self.genome = genome
        self.parent1_id = parent1_id
        self.parent2_id = parent2_id
        self.born_generation = None
        self.phenotype = None
        self.valid = 0    # invalid until successfully evaluated

        # EvoGym simulation payload (explicit fields to avoid ad-hoc attributes)
        self.evogym_structure = None
        self.evogym_connections = None
        self.evogym_phase_offsets = None
        self.evogym_controller = None

        # === Dynamically create absolute metrics ======================
        for m in METRICS_ABS:
            # displacement (x) is the only one that previously had -inf default
            if m == "displacement":
                setattr(self, m, float('-inf'))
            else:
                setattr(self, m, None)

        # === Dynamically create relative metrics ======================
        for m in METRICS_REL:
            setattr(self, m, None)
