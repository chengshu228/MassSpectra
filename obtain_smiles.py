import os
import numpy as np
import re
import requests
import urllib
import http.client

from utils_function import get_redirect_url

http.client.HTTPConnection._http_vsn = 10
http.client.HTTPConnection._http_vsn_str = 'HTTP/1.0'

current_loc = os.path.dirname(os.path.realpath(__file__))
origin_data_loc = os.path.join(current_loc, 'origin_data')
if not os.path.exists(origin_data_loc):
    os.mkdir(origin_data_loc)
SMILES_loc = os.path.join(current_loc, 'SMILES')
if not os.path.exists(SMILES_loc):
    os.mkdir(SMILES_loc)
html_loc = os.path.join(current_loc, 'html_intensity')
if not os.path.exists(html_loc):
    os.mkdir(html_loc)

for _,_,filenames in os.walk(origin_data_loc): 
    continue 

for i in np.arange(0, len(filenames)):
    print(f'\n第{i}个原始文件文件')
    # file_no = int(filenames[i][2:-4])
    with open(f"{origin_data_loc}/{filenames[i]}", 'r', encoding='utf-8') as f_origin:
        f_context = f_origin.read()
        cas = re.compile(r"CAS (.*?)\n", re.DOTALL)
        cas_result = cas.findall(f_context)
        # print(f'\tcas_result: {cas_result}')
        if cas_result!=[''] and cas_result!=[]:            
            print(f'\t文件名:{filenames[i]} CAS:{cas_result[0]}')
            with open(f"{SMILES_loc}/{filenames[i]}", 'w', encoding='utf-8') as f_smiles:
                redirect_url = get_redirect_url(query=cas_result, html_loc=html_loc, filename=filenames[i][:-4])
                with open(f"{html_loc}/{filenames[i][:-4]}.html", 'r', encoding='utf-8') as f_html:
                    html = f_html.read()
                    pattern = re.compile(r"&lt;SMILES&gt;(.*?)&gt; &lt;INCHI_IDENTIFIER&gt;", re.DOTALL)
                    smiles_result = pattern.findall(html)
                    if smiles_result != []:
                        print(f'\tSMILSES: {smiles_result[0][1:-2]}')
                        f_smiles.write(str(smiles_result[0][1:-2]))

print(f'\n\tThe number of origin data is {numberFile(origin_data_loc)}.')
print(f'\n\tThe number of SMILES is {numberFile(SMILES)}.')
print(f'\n\tThe number of html is {numberFile(mz_int_loc)}.')

print('\n\tEnd...\n')