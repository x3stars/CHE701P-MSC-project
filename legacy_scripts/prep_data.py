import pandas as pd
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import Descriptors

#Set paths
input_path = r"C:\Users\Finley\OneDrive\Masters code\Data\pfDHFR_chembl_raw.csv"
output_path = r"C:\Users\Finley\OneDrive\Masters code\Data\pfDHFR_cleaned_with_potency_descriptors.csv"

#Load data
print(f"Loading data from: {input_path}")
df = pd.read_csv(input_path, sep=";")
print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

#Check missing values
missing_summary = df.isna().sum()
missing_pct = (missing_summary / len(df)) * 100

print("\nMissing values (count):")
print(missing_summary[missing_summary > 0].sort_values(ascending=False))
print("\nMissing values (percentage):")
print(missing_pct[missing_pct > 0].sort_values(ascending=False))

#Inspect key columns
print("\nStandard Type value counts:")
print(df["Standard Type"].value_counts())

print("\nStandard Units value counts:")
print(df["Standard Units"].value_counts())

print("\nStandard Relation value counts:")
print(df["Standard Relation"].value_counts())

#Filter data 
df = df[df["Standard Type"].isin(["IC50", "Ki"])]
df = df[df["Standard Units"] == "nM"]
df = df[df["Standard Relation"].str.strip(" '\"") == "="]
df = df[df["Standard Value"].notnull() & (df["Standard Value"] > 0)]

print("\nAfter potency filtering:", df.shape)

#SMILES parsing 
df = df.dropna(subset=["Smiles"])
df["mol"] = df["Smiles"].apply(Chem.MolFromSmiles)
df = df[df["mol"].notna()]
print("After SMILES parsing:", df.shape)

#Compute pPotency and activity label 
df["pPotency"] = -np.log10(df["Standard Value"] * 1e-9)
df["active"] = df["Standard Value"].apply(lambda x: 1 if x < 1000 else 0)
print("pPotency range:", df["pPotency"].min(), "to", df["pPotency"].max())
print("Active compounds:", df['active'].sum(), "Inactive:", len(df) - df['active'].sum())

#Calculate descriptors
df["mol_weight"] = df["mol"].apply(Descriptors.MolWt)
df["logP"] = df["mol"].apply(Descriptors.MolLogP)
df["TPSA"] = df["mol"].apply(Descriptors.TPSA)
df["HBD"] = df["mol"].apply(Descriptors.NumHDonors)
df["HBA"] = df["mol"].apply(Descriptors.NumHAcceptors)
df["RotatableBonds"] = df["mol"].apply(Descriptors.NumRotatableBonds)
df["Fsp3"] = df["mol"].apply(Descriptors.FractionCSP3)
df["NumAromaticRings"] = df["mol"].apply(Descriptors.NumAromaticRings)
df["NumAliphaticRings"] = df["mol"].apply(Descriptors.NumAliphaticRings)
df["NumSaturatedRings"] = df["mol"].apply(Descriptors.NumSaturatedRings)
df["HeavyAtomCount"] = df["mol"].apply(Descriptors.HeavyAtomCount)
df["FormalCharge"] = df["mol"].apply(lambda mol: Chem.GetFormalCharge(mol))

#Drop unnecessary columns 
cols_to_drop = [
    "Molecule Name", "Molecule Max Phase", "Ligand Efficiency BEI",
    "Ligand Efficiency LE", "Ligand Efficiency LLE", "Ligand Efficiency SEI",
    "pChEMBL Value", "Data Validity Comment", "Comment", "Assay Tissue ChEMBL ID",
    "Assay Tissue Name", "Assay Cell Type", "Assay Subcellular Fraction",
    "Assay Parameters", "Assay Variant Accession", "Assay Variant Mutation",
    "Cell ChEMBL ID", "Properties", "Action Type", "Standard Text Value"
]
df_cleaned = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

#Save cleaned data 
df_cleaned.to_csv(output_path, index=False)
print(f"\nSaved cleaned dataset to: {output_path}")
print("File exists:", os.path.exists(output_path))


