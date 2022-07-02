import os 
import numpy as np
import json

from utils_function import numberFile

# current_loc = os.path.dirname(os.path.realpath(__file__))
data_loc = r'/data/cshu/mass_spectra/smiles_intensity'
new_ms_loc = os.path.join(data_loc, 'new_mass_spectra')

filenames = ''
for _,_,filenames in os.walk(new_ms_loc): 
    # filenames.append(file_names)
    continue 
print(filenames)
number_of_file = numberFile(os.path.join(data_loc, 'SMILES'))
print(number_of_file)

mass_spectra = []
for filename in np.arange(number_of_file):
    new_line = ''
    print(filename)
    print(filenames[filename])
    with open(f"{new_ms_loc}/{filenames[filename]}", 'r', encoding='utf-8') as f_ms:     
        for line in f_ms.readlines():
            # print(line)
            if not line: continue
            else:
                this_line = line.strip().split()
                # print(this_line)
                for key, value in enumerate(this_line):
                    if key != 1:
                        new_line += value + ' '
    if filename % 1000 == 0:
        print(filenames[filename], new_line) # PS071412.txt 78 228 79 999
    mass_spectra.append(new_line)

with open(f"{data_loc}/mass_spectra.json", 'w', newline='') as f_ms_json:
    json.dump(mass_spectra, f_ms_json, indent=2)
