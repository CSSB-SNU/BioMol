import importlib.util
import multiprocessing
from pathlib import Path

import numpy as np


from biomol.io.processor import Processor
from biomol.io.schema import MappingSpec, FeatureSpec, FeatureKind, FeatureLevel
from biomol.core.biomol import BioMol
from biomol.core.container import AtomContainer, ResidueContainer, ChainContainer
from biomol.core.index import IndexTable


class BioMolFactory:
    """
    Create a Processor instance with the given pipeline configuration.

    Args:
        pipeline: A list of MappingSpec objects defining the parsing stages.

    Returns
    -------
        An instance of Processor initialized with the provided pipeline.

    ***NOTE***: This factory currently supports only single chain small-molecule.
    """

    def __init__(self, blueprint: str, num_workers: int = 1) -> None:
        self.plan = self._load_plan(blueprint)
        self.num_workers = (
            num_workers if num_workers > 0 else multiprocessing.cpu_count()
        )

    def _load_plan(self, blueprint_path: str) -> list[MappingSpec]:
        """
        Dynamically load a processing plan from the plans module.

        This method assumes that the plan is already defined
        """
        blueprint = Path(blueprint_path).resolve()
        if not blueprint.is_file():
            msg = f"Plan file '{blueprint_path}' does not exist."
            raise FileNotFoundError(msg)

        module_name = blueprint.stem
        spec = importlib.util.spec_from_file_location(module_name, blueprint)
        if spec is None or spec.loader is None:
            msg = f"Cannot load module from '{blueprint_path}'"
            raise ImportError(msg)
        blueprint_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(blueprint_module)
        plan = getattr(blueprint_module, "PLAN", None)
        if plan is None:
            msg = f"The plan file '{blueprint_path}' must define a 'PLAN' variable."
            raise AttributeError(msg)

        return plan

    def _produce_module(self, raw_material: dict) -> BioMol:
        if not self.plan:
            msg = "No processing plan loaded. Please load a plan before producing."
            raise ValueError(msg)
        parser = Processor(self.plan)
        intermediate_parts = parser.parse(raw_material)
        # ---- build BioMol ---- #
        atom_container = AtomContainer(
            node_features=intermediate_parts.get_features(
                FeatureSpec(
                    name="", kind=FeatureKind.NODE, level=FeatureLevel.ATOM, dtype=None
                )
            ),
            edge_features=intermediate_parts.get_features(
                FeatureSpec(
                    name="", kind=FeatureKind.EDGE, level=FeatureLevel.ATOM, dtype=None
                )
            ),
        )
        residue_container = ResidueContainer(
            node_features=intermediate_parts.get_features(
                FeatureSpec(
                    name="",
                    kind=FeatureKind.NODE,
                    level=FeatureLevel.RESIDUE,
                    dtype=None,
                )
            ),
            edge_features={},
        )
        chain_container = ChainContainer(
            node_features=intermediate_parts.get_features(
                FeatureSpec(
                    name="",
                    kind=FeatureKind.NODE,
                    level=FeatureLevel.RESIDUE,
                    dtype=None,
                )
            ),
            edge_features={},
        )  # TODO: use FeatureLevel.CHAIN rather than RESIDUE in MMCIF parsing

        num_atoms = len(intermediate_parts.get_feature("atom_id"))
        index_table = IndexTable.from_parents(
            atom_to_res=np.zeros((num_atoms,), dtype=np.int32),
            res_to_chain=np.zeros((1,), dtype=np.int32),
            n_chain=1,
        )  # TODO: make dynamic index mapping for MMCIF parsing

        return BioMol(atom_container, residue_container, chain_container, index_table)

    def produce(self, dataset: list[dict]) -> list[BioMol]:
        """Produce BioMol objects from raw datas."""
        if not self.plan:
            msg = "No processing plan loaded. Please load a plan before producing."
            raise ValueError(msg)
        if self.num_workers == 1:
            return [self._produce_module(data) for data in dataset]

        with multiprocessing.Pool(self.num_workers) as pool:
            results = pool.map(self._produce_module, dataset)
        return results
