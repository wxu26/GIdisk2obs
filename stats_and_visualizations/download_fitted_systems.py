import requests
from os.path import exists

ids = {
    'fit_1mm_Q1d5_age1e5.pkl': 6045345,
    'fit_1mm_Q1_age1e5.pkl': 6045350,
    'fit_1mm_Q2_age1e5.pkl': 6045351,
    'fit_1mm_Q10_age1e5.pkl': 6045352,
    'fit_1mm_Q100_age1e5.pkl': 6045353,
    'fit_10um_Q1d5_age1e5.pkl': 6045348,
    'fit_100um_Q1d5_age1e5.pkl': 6045349,
    'fit_1cm_Q1d5_age1e5.pkl': 6045347,
    'fit_1mm_Q1d5_age2e5.pkl': 6045355,
    'fit_1mm_Q1d5_age5e4.pkl': 6045354,
    'fit_1mm_Q1d5_age1e5_alma_only.pkl': 6045346,
}

def download_one_file(fname, folder='../data/fitted_systems'):
    if fname not in ids:
        raise ValueError('invalid file name!')
    download_url =  'https://dataverse.harvard.edu/api/access/datafile/{}'.format(ids[fname])
    download_path = folder+'/{}'.format(fname)
    if exists(download_path):
        print('file already exists, skipping download')
    else:
        r = requests.get(download_url, allow_redirects=True)
        open(download_path, 'wb').write(r.content)
        print('finished downloading', download_path)
    return

if __name__=='__main__':
    for fname in list(ids.keys()):
        download_one_file(fname)
