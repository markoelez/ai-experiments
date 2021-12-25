import os
import gzip
import hashlib
import numpy as np


def fetch(url):
    if not os.path.isdir('data'):
        os.mkdir('data')
    fp = md5 = 'data/' + hashlib.md5(url.encode()).hexdigest()
    if os.path.isfile(fp):
        with open(fp, 'rb') as f:
            dat = f.read()
    else:
        with open(fp, 'wb') as f:
            res = requests.get(url)
            res.raise_for_status()
            dat = res.content
            f.write(dat)
    assert b'503' not in dat, 'request failed'
    return np.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()
