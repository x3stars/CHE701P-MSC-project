import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors

inp  = r"C:\Users\Finley\OneDrive\Masters code\top5_raw.csv"
outp = r"C:\Users\Finley\OneDrive\Masters code\top5_parent_props.csv"

df = pd.read_csv(inp)
def largest_parent(smiles: str):
    m = Chem.MolFromSmiles(smiles)
    if m is None: return None
    frags = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=True)
    # keep largest by heavy atoms
    parent = max(frags, key=lambda x: x.GetNumHeavyAtoms())
    # re-canonicalize
    return Chem.MolToSmiles(parent, canonical=True)

rows = []
for _, r in df.iterrows():
    smi_all = r["smiles"]
    parent_smi = largest_parent(smi_all)
    if not parent_smi:
        rows.append({**r.to_dict(), "parent_smiles": None})
        continue
    m = Chem.MolFromSmiles(parent_smi)
    props = {
        "parent_smiles": parent_smi,
        "MW": Descriptors.MolWt(m),
        "cLogP": Crippen.MolLogP(m),
        "TPSA": rdMolDescriptors.CalcTPSA(m),
        "HBD": rdMolDescriptors.CalcNumHBD(m),
        "HBA": rdMolDescriptors.CalcNumHBA(m),
        "RotBonds": rdMolDescriptors.CalcNumRotatableBonds(m),
        "FormalCharge": sum(a.GetFormalCharge() for a in m.GetAtoms()),
        "HeavyAtoms": m.GetNumHeavyAtoms(),
    }
    rows.append({**r.to_dict(), **props})

out = pd.DataFrame(rows)
out.to_csv(outp, index=False)
print(f"Wrote {outp} (n={len(out)})")
