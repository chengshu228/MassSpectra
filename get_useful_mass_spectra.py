import os
import numpy as np

from utils_function import numberFile

# current_loc = os.path.dirname(os.path.realpath(__file__))
data_loc = r'/data/cshu/mass_spectra/smiles_intensity'

# origin_ms_loc = os.path.join(data_loc, 'origin_data')
ms_loc = os.path.join(data_loc, 'mass_spectra')
smiles_loc = os.path.join(data_loc, 'SMILES')
new_ms_loc = os.path.join(data_loc, 'new_mass_spectra')

if not os.path.exists(new_ms_loc):
    os.mkdir(new_ms_loc)

for _,_,filenames in os.walk(smiles_loc):
    continue
print(len(filenames))

for filename in np.arange(len(filenames)):
# for filename in np.arange(3,5):
    with open(f'{ms_loc}/{filenames[filename]}', 'r', encoding='utf-8') as file_ms:
        context = file_ms.read()
        # if context
        with open(f'{new_ms_loc}/{filenames[filename]}', 'w', encoding='utf-8') as file_ms_new:
            file_ms_new.write(context)


print(f'\n\tThe number of useful SMILES is {numberFile(smiles_loc)}.')
print(f'\n\tThe number of useful mass spetra is {numberFile(new_ms_loc)}.')
# print(f'\n\tThe number of origin mass spetra is {numberFile(origin_ms_loc)}.')

print('\n\tEnd...\n')
