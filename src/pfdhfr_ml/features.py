"""Feature generation (ECFP4)."""
from rdkit.Chem import AllChem

def ecfp4_bitvect(mol, nBits=2048, radius=2, with_bitinfo=False):
    bitInfo = {} if with_bitinfo else None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits, bitInfo=bitInfo)
    return fp, bitInfo
