import os
import numpy as np
import re

from utils_function import numberFile

# current_loc = os.path.dirname(os.path.realpath(__file__))
data_loc = r'/data/cshu/mass_spectra'

origin_data_loc = os.path.join(data_loc, 'origin_data')
if not os.path.exists(origin_data_loc):
    os.mkdir(origin_data_loc)

mz_int_loc = os.path.join(data_loc, 'mass_spectra')
if not os.path.exists(mz_int_loc):
    os.mkdir(mz_int_loc)

for _,_,filenames in os.walk(origin_data_loc): continue

for filename in np.arange(len(filenames)):
    # file_no = int(filenames[i][2:-4])
    with open(f"{origin_data_loc}/{filenames[filename]}", 'r', encoding='utf-8') as f_origin:
        f_context = f_origin.read()
        find_ms = re.findall(r"rel.int.(.+?)//", f_context, re.S)
        # print(find_ms)
        with open(f'{mz_int_loc}/{filenames[filename]}', 'w', encoding='utf-8') as f_mz_int:
            for item in find_ms:
                f_mz_int.write(item)

print(f'\n\tThe number of origin data is {numberFile(origin_data_loc)}.')
print(f'\n\tThe number of mass spetra and intensity is {numberFile(mz_int_loc)}.')
print('\n\tEnd...\n')
