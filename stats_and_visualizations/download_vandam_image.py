from os.path import exists
from bs4 import BeautifulSoup
import requests
import astropy.table

def download_image(
    source_name,obs='both',folder='./data/observation',
    data_table='./data/VANDAM_T20_properties.txt',
    verbose=False,
    ):
    """
    Download images for a source from the VANDAM Orion online database.
    """
    # obs: ALMA, VLA, or both
    if obs=='both':
        suc1 = download_image(source_name,obs='ALMA',folder=folder,data_table=data_table,verbose=verbose)
        suc2 = download_image(source_name,obs='VLA',folder=folder,data_table=data_table,verbose=verbose)
        return suc1*suc2
    # load data table
    data = astropy.table.Table.read(data_table, format="ascii")
    data.add_index('Source') # add index by source
    # get file name
    i_source = data.loc_indices[source_name]
    if obs=='ALMA':
        field_name = data['FieldA'][i_source]
        file_name = field_name + '_cont_robust0.5.pbcor.fits'
    elif obs=='VLA':
        field_name = data['FieldV'][i_source]
        file_name = field_name + '.A.Ka.cont.0.5.robust.image.pbcor.fits'
    else:
        print('invalid observation type (obs)!')
        return 0
    if verbose:
        print('file name:', file_name)
    # check if file already exists
    download_path = folder + '/' + file_name
    if exists(download_path):
        if verbose: print('file already exists, skipping download')
        return 1
    # search the database
    search_url_base = 'https://dataverse.harvard.edu/dataverse/VANDAMOrion/?q=fileName%3A'
    search_url = search_url_base+file_name
    if verbose:
        print(search_url)
    # grab persistentId
    reqs = requests.get(search_url)
    soup = BeautifulSoup(reqs.text, 'html.parser')
    Id = ''
    for link in soup.find_all('a'):
        lt = link.get('href')
        if lt[:5]=='/file':
            Id = lt[11:]
            link_text = link.get_text()
            if link_text.lower()!=file_name.lower(): # case insensitive comparison
                if verbose:
                    print('requested file not found!')
                    print('closest match:',link_text)
                    return 0
            if verbose:
                print('persistent ID:',Id)
            break
    download_url_base =  'https://dataverse.harvard.edu/api/access/datafile/:persistentId'
    download_url =  download_url_base+Id
    # download file by URL
    r = requests.get(download_url, allow_redirects=True)
    open(download_path, 'wb').write(r.content)
    if verbose: print('finished downloading')
    return 1