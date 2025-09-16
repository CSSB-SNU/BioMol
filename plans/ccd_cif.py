from biomol.io.blueprint import Blueprint

# --- Use the beloved Blueprint DSL to define the plan ---
_blueprint = (
    Blueprint()
    .stage("parse_chem_comp_properties")
    .using("identity")
    .from_fields("_chem_comp.id", "_chem_comp.name", "_chem_comp.formula")
    .to_residue_nodes(id=str, name=str, formula=str)
)
_blueprint = (
    _blueprint.stage("parse_atoms")
    .using("identity")
    .from_fields("_chem_comp_atom.atom_id", "_chem_comp_atom.type_symbol")
    .to_atom_nodes(atom_id=str, atom_symbol=(str, {"?": "X"}))
)
ccd_plan = (
    _blueprint.stage("parse_ideal_coordinates")
    .using("stack")
    .from_fields(
        "_chem_comp_atom.pdbx_model_Cartn_x_ideal",
        "_chem_comp_atom.pdbx_model_Cartn_y_ideal",
        "_chem_comp_atom.pdbx_model_Cartn_z_ideal",
    )
    .to_atom_nodes(ideal_coords=(float, {"?": 0.0}))
)
_blueprint = (
    _blueprint.stage("parse_bonds")
    .using("bond")
    .from_fields(
        "_chem_comp_bond.atom_id_1",
        "_chem_comp_bond.atom_id_2",
        "_chem_comp_bond.value_order",
    )
    .with_context("atom_id")
    .to_atom_edges(bond_type=str)
)

# --- Assign the built plan to a conventional variable name ---
PLAN = _blueprint.build()
