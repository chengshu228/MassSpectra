from utils_function import split

from rdkit import Chem
from rdkit import Chem
from rdkit.Chem import Draw
mol = Chem.MolFromSmiles('CS(=O)CCCCCCCN=C=S')
mol = Chem.MolFromSmiles('C\\C(\\C=C\\C=C(/C)\\C=C\\[C@H]1C(C)=CCCC1(C)C)=C/C=C/C=C(\\C)/C=C/C=C(\\C)/C=C/C1=C(C)CCCC1(C)C')
# Draw.MolToImage(mol, size=(150,150), kekulize=True)

# Draw.ShowMol(mol, size=(150,150), kekulize=False)
Draw.MolToFile(mol, 'data/output.png', size=(600, 600))

def test_split():
    sm = 'C(=O)CC(Br)C[N+]CN'
    pred = 'C ( = O ) C C ( Br ) C [ N + ] C N'
    assert split(sm)==pred
    print(sm)
    print(pred)
    print(split(sm)==pred)
    print(split("CN1C(=NS(=O)(=O)c2ccc(Cl)cc2)C(=NN=P(c3ccccc3)(c4ccccc4)c5ccccc5)c6ccccc16COc1ccc(Cl)cc1c2cc([nH]n2)C(=O)Nc3ccc(OC)nc3"))

test_split()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, MolFromSmiles, MolToSmiles
import json

# df = pd.read_table('data/chembl_25_chemreps.txt')
# L = len(df)
# print(L)
# df.head()
import os

current_path = os.getcwd()
smiles = []
# with open(os.path.join(current_path, 'data/smiles.json'), 'r', encoding='utf-8') as f_smiles:
with open(os.path.join(current_path, 'data/smiles.txt'), 'r', encoding='utf-8') as f_smiles:
    # smiles = json.load(f_smiles)
    lines = f_smiles.readlines()

for l in lines:
    smiles.append(l.replace('\n', ''))
del lines
smiles = np.array(smiles)
print(smiles[-1])

# smiles = smiles.strip().split('\n')
print(smiles)

mols = []
for s in smiles:
    if s is None:
        continue
    mol = MolFromSmiles(s)
    if mol is not None:
        mols.append(mol)
# img = Draw.MolsToGridImage(mols, molsPerRow=1, subImgSize=(300,150))
# # from matplotlib.colors import ColorConverter
# # img = Draw.MolToImage(mols, highlightAtoms=[1,2], highlightColor=ColorConverter().to_rgb('aqua'))
# img.save('molecule.png')

lengths = list(map(len, smiles))

plt.figure()
plt.hist(lengths, bins=100)
plt.grid()
plt.xlabel('SMILES length')
plt.ylabel('Counts')
plt.yscale('log')
plt.savefig('smiles_length.png')
plt.show()

print(np.sum(np.array(lengths)<=100)/len(smiles)*100)

sub = np.array(lengths)<=100
print(len(sub))