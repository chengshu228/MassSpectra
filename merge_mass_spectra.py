import os 
import numpy as np
import json

from utils_function import numberFile
import config

data_loc = config.data_loc

new_ms_loc = os.path.join(data_loc, 'new_mass_spectra')

filenames = ''
for _,_,filenames in os.walk(new_ms_loc): continue 

number_of_file = numberFile(os.path.join(data_loc, 'SMILES'))

mass_spectra = []
for filename in np.arange(number_of_file):
    new_line = ''
    with open(f"{new_ms_loc}/{filenames[filename]}", 'r', encoding='utf-8') as f_ms:     
        for line in f_ms.readlines():
            if not line: continue
            else:
                this_line = line.strip().split()
                for key, value in enumerate(this_line):
                    if key != 1:
                        new_line += value + ' '
    if filename % 1000 == 0:
        print(filenames[filename], new_line) 
    mass_spectra.append(new_line)

with open(f"{data_loc}/mass_spectra.json", 'w', newline='') as f_ms_json:
    json.dump(mass_spectra, f_ms_json, indent=2)
