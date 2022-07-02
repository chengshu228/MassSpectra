import numpy as np
import os
import re
from urllib import error
 
import config

data_loc = config.data_loc

mass_spectra_location = os.path.join(data_loc, 'mass_spectra')
if not os.path.exists(mass_spectra_location):
    os.mkdir(mass_spectra_location)

smiles_intensity_location = os.path.join(data_loc, 'smiles_intensity')
if not os.path.exists(smiles_intensity_location):
    os.mkdir(smiles_intensity_location)

html_intensity_location = os.path.join(data_loc, 'html_intensity')
if not os.path.exists(html_intensity_location):
    os.mkdir(html_intensity_location)

for _,_,filename in os.walk(mass_spectra_location):  continue

try :
    with open(data_loc+'get_name.txt', 'w', encoding='utf-8') as f_out:
        for i in np.arange(len(filename)):
            file_no = int(filename[i][2:-4])
            with open(mass_spectra_location+r'/{}'.format(filename[i]), 'r', encoding='utf-8') as f_in:
                f_context = f_in.read()
                result = re.findall(r'RECORD_TITLE: (.+?); ', f_context, re.S)
                print(filename[i], result)
                for item in result:
                    f_out.write(item+'\n')
                f_in.close()
        f_out.close()
except error.HTTPError as e:
    print('HTTPError:{}'.format(e.reason))
    print('HTTPError:{}'.format(e))    
except error.URLError as e:
    print('URLError:{}'.format(e.reason))
    print('URLError:{}'.format(e))
except Exception as e:
    print(e)