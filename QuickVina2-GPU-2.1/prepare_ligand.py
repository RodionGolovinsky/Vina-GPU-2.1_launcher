

import os
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolAlign
import time
try:
    from openbabel import openbabel
except:
    print(">>> Warning:\n"
          "            Could not find OpenBabel!!! SMILES2MOL2 and SMILES2PDBQT are not available!\n"
          ">>> To install openbabel:\n"
          "            conda install -c conda-forge openbabel")
    

def smiles2sdf(smiles, labels=None, ref=None, mergesdf=False, onlysdf=True):
    if isinstance(smiles, str):
        mols = []
        mols.append(smiles)
        if labels:
            label = []
            label.append(labels)
        else:
            mlabel = []
        mergesdf=False
    elif isinstance(smiles, list):
        mols = smiles
        if labels:
            label = labels
        else:
            mlabel = []
        if mergesdf:
            w0 = Chem.SDWriter('all_mols.sdf')
    for i, smile in enumerate(mols):
        try:
            mol = Chem.MolFromSmiles(smile)
            if labels:
                mol.SetProp("_Name",label[i])
            mol = Chem.AddHs(mol)
            ps = AllChem.ETKDGv3()
            ps.randomSeed = 0xf00d
            AllChem.EmbedMolecule(mol,ps)
            if ref:
                suppl = Chem.SDMolSupplier(ref)
                refmol = suppl[0]
                refmol = Chem.AddHs(refmol)
                o3d = rdMolAlign.GetO3A(mol,refmol)
                try:
                    o3d.Align()
                except:
                    pass
            if labels:
                w = Chem.SDWriter('%s.sdf'%(label[i]))
            else:
                l = "mol-%d"%(i+1)
                mlabel.append(l)
                w = Chem.SDWriter('%s.sdf'%(l))
            if mergesdf:
                w0.write(mol)
            else:
                w.write(mol)
        except:
            if labels:
                print("Failed to prepare molecule : %s with SMILES = %s" % (label[i],smile))
            else:
                print("Failed to prepare molecule : mol-%d with SMILES = %s"%(i+1,smile))
    if not onlysdf:
        if labels:
            return mols, label
        else:
            return mols, mlabel

def smiles2mol2(smiles, labels=None, ref=None):
    mergesdf=False
    mols, label = smiles2sdf(smiles, labels=labels, ref=ref, mergesdf=mergesdf, onlysdf=False)
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("sdf", "mol2")
    for l in label:
        mol = openbabel.OBMol()
        obConversion.ReadFile(mol, "%s.sdf"%(l))
        obConversion.WriteFile(mol, "%s.mol2"%(l))
        os.system("rm %s.sdf"%(l))
    return mols, label

def smiles2pdbqt(smiles, path_to_save, ref=None):
    labels = os.path.split(path_to_save)[-1]
    mols, label = smiles2mol2(smiles, labels=labels, ref=ref)
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("mol2", "pdbqt")
    for l in label:
        mol = openbabel.OBMol()
        obConversion.ReadFile(mol, "%s.mol2"%(l))
        path_pdbqt = os.path.join(os.path.split(path_to_save)[0], "%s.pdbqt"%(l))
        obConversion.WriteFile(mol, path_pdbqt)
        os.system("rm %s.mol2"%(l))
    return mols, label

def preprocess_ligand(smiles: str, path_to_save: str = os.getcwd()) -> None:
    smiles2pdbqt(smiles, path_to_save)
