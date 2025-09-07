"""Plotting helpers (bit highlight)."""
from rdkit import Chem
from rdkit.Chem import Draw

def highlight_bit_png(mol, bitInfo, bit_id:int, out_png:str, size=(600,450)) -> bool:
    if bit_id not in bitInfo or not bitInfo[bit_id]:
        return False
    atoms_to_highlight = []
    for atom_idx, rad in bitInfo[bit_id]:
        env = Chem.FindAtomEnvironmentOfRadiusN(mol, rad, atom_idx)
        amap = {}
        Chem.PathToSubmol(mol, env, atomMap=amap)
        atoms_to_highlight += list(amap.values())
    Draw.MolToImage(mol, size=size, highlightAtoms=list(set(atoms_to_highlight))).save(out_png)
    return True
