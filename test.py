from biomol import BioMol
import os

cif_dir = '/data/psk6950/PDB_2024Mar18/cif/'
cif_path = os.path.join(cif_dir, 'm1/1m1c.cif.gz')

# load cif
biomol = BioMol(cif=cif_path, cif_config="./cif_configs/protein_only.json")
# choose assembly_id, model_id, alt_id
biomol.choose('1', '1', '.')
# crop and load msa
biomol.crop_and_load_msa({'method_prob': [0.2,0.4,0.4], 'crop_size': 384})
# save cropped structure to cif
biomol.structure.to_mmcif('test.cif')
