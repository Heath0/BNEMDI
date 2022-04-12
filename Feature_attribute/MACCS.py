from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.AtomPairs import Torsions

from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker

import matplotlib.pyplot as plt  # 画图A+B
import pandas as pd

if __name__ == '__main__':
    dataset = 'ncDR'
    drug = pd.read_csv(r'../Data/'+dataset+' drug SMILES.csv', sep=',')
    drug_name = drug['drug'].values.tolist()
    drug_smile = drug['smile'].values.tolist()
    mol2 = [Chem.MolFromSmiles(x) for x in drug_smile]
    maccs = [MACCSkeys.GenMACCSKeys(x).ToBitString() for x in mol2]

    maccs_list = []
    for a in range(len(maccs)):
        temp = []
        temp.append(drug_name[a])
        print(drug_name[a])
        for b in range(len(maccs[a])):
            #print(maccs[a][b])
            temp.extend(maccs[a][b])

        maccs_list.append(temp)

    pd.DataFrame({'drug':drug_name,'maccs':maccs_list}).to_csv(dataset+' drug MACCS.csv', index=None)
    print(len(maccs[0]))    #166