import os 
import numpy as np
import json
import csv

from utils_function import numberFile

# current_loc = os.path.dirname(os.path.realpath(__file__))
data_loc = r'/data/cshu/mass_spectra/smiles_intensity'
SMILES_loc = os.path.join(data_loc, 'SMILES')

for _,_,filenames in os.walk(SMILES_loc): continue 

number_of_file = numberFile(os.path.join(os.getcwd(), 'SMILES'))
print(number_of_file)

with open(f"{data_loc}/smiles.txt", 'w+', newline='') as f_smiles_json:
    for filename in np.arange(number_of_file):
        with open(f"{SMILES_loc}/{filenames[filename]}", 'r', encoding='utf-8') as f_smiles:
            content = f_smiles.read()
            f_smiles_json.write(content+'\n')
            if filename % 1000 == 0:
                print(filenames[filename], content) # PS071412.txt 78 228 79 999