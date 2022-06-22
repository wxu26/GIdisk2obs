import requests
from os.path import exists

ids = {
    'fit_1mm_Q1d5_age1e5.pkl': 6347016,
    'fit_1mm_Q1_age1e5.pkl': 6347079,
    'fit_1mm_Q2_age1e5.pkl': 6347088,
    'fit_1mm_Q3_age1e5.pkl': 6347095,
    'fit_1mm_Q10_age1e5.pkl': 6347007,
    'fit_1mm_Q100_age1e5.pkl': 6346998,
    'fit_10um_Q1d5_age1e5.pkl': 6346982,
    'fit_100um_Q1d5_age1e5.pkl': 6346975,
    'fit_1cm_Q1d5_age1e5.pkl': 6346991,
    'fit_1mm_Q1d5_age2e5.pkl': 6347069,
    'fit_1mm_Q1d5_age5e4.pkl': 6347074,
    'fit_1mm_Q1d5_age1e5_alma_only.pkl': 6347060,
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
