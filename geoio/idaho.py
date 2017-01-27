from timbr.snapshot.snapshot import Snapshot
from timbr.machine import serializer

try:
    from io import BytesIO
except ImportError:
    from StringIO import cStringIO as BytesIO

import tables
import h5py

import os
import codecs
import sys
import json
import inspect
from functools import partial
from collections import defaultdict
from itertools import groupby
import threading
import contextlib
from IPython.display import display, Javascript

import warnings
warnings.filterwarnings('ignore')

import requests
from requests.compat import urljoin
import xml.etree.cElementTree as ET
import pycurl

import rasterio
from rasterio.io import MemoryFile
import dask
from dask.delayed import delayed
import dask.array as da
import numpy as np

from gbdxtools import Interface

import geoio

from osgeo import gdal
gdal.UseExceptions()

NTHREAD_DEFAULT = 4
_num_workers = NTHREAD_DEFAULT

if "TIMBR_DGSNAP_NTHREAD" in os.environ:
    try:
        _num_workers = int(os.environ["TIMBR_DGSNAP_NTHREAD"])
    except ValueError as ve:
        os.environ["TIMBR_DGSNAP_NTHREAD"] = NTHREAD_DEFAULT

threaded_get = partial(dask.threaded.get, num_workers=_num_workers)
_curl_pool = defaultdict(pycurl.Curl)

class NotSupportedException(NotImplementedError):
    pass

def data_to_np(data):
    return np.fromstring(serializer.dumps(data), dtype="uint8")

def parse_bounds(raw):
    bounds = [float(n.strip()) for n in raw.split(",")]
    return bounds

def index_to_slice(ind, rowstep, colstep):
    i, j = ind
    window = ((i * rowstep, (i + 1) * rowstep), (j * colstep, (j + 1) * colstep))
    return window

def roi_from_bbox_projection(src, user_bounds, block_shapes=None, preserve_blocksize=True):
    roi = src.window(*user_bounds)
    if not preserve_blocksize:
        return roi
    if block_shapes is None:
        blocksafe_roi = rasterio.windows.round_window_to_full_blocks(roi, src.block_shapes)
    else:
        blocksafe_roi = rasterio.windows.round_window_to_full_blocks(roi, block_shapes)
    return blocksafe_roi

def generate_blocks(window, blocksize):
    rowsize, colsize = blocksize
    nrowblocks = window.num_rows / rowsize
    ncolblocks = window.num_cols / colsize
    for ind in np.ndindex((nrowblocks, ncolblocks)):
        yield index_to_slice(ind, nrowblocks, ncolblocks)

def build_url(gid, base_url="http://idaho.timbr.io", node="TOAReflectance", level="0"):
    relpath = "/".join([gid, node, str(level) + ".vrt"])
    return urljoin(base_url, relpath)

def collect_urls(vrt):
    doc = ET.parse(vrt)
    urls = list(set(item.text for item in doc.getroot().iter("SourceFilename")
                if item.text.startswith("http://")))
    chunks = []
    for url in urls:
        head, _ = os.path.splitext(url)
        head, y = os.path.split(head)
        head, x = os.path.split(head)
        head, key = os.path.split(head)
        y = int(y)
        x = int(x)
        chunks.append((x, y, url))

    grid = [[rec[-1] for rec in sorted(it, key=lambda x: x[1])]
            for key, it in groupby(sorted(chunks, key=lambda x: x[0]), lambda x: x[0])]
    return grid

@delayed
def load_url(url, bands=8):
    #print('fetching', url)
    thread_id = threading.current_thread().ident
    _curl = _curl_pool[thread_id]
    buf = BytesIO()
    _curl.setopt(_curl.URL, url)
    _curl.setopt(_curl.WRITEDATA, buf)
    _curl.perform()

    with MemoryFile(buf.getvalue()) as memfile:
      try:
          with memfile.open(driver="GTiff") as dataset:
              arr = dataset.read()
      except (TypeError, rasterio.RasterioIOError) as e:
          print("Errored on {} with {}".format(url, e))
          arr = np.zeros([bands,256,256], dtype=np.float32)
          _curl.close()
          del _curl_pool[thread_id]
    return arr

def build_array(urls, bands=8):
    buf = da.concatenate(
        [da.concatenate([da.from_delayed(load_url(url, bands=bands), (bands,256,256), np.float32) for url in row],
                        axis=1) for row in urls], axis=2)
    return buf

def ms_to_rgb(mbimage):
    nbands, x, y = mbimage.shape
    if nbands == 8:
        rgb_uint8 = (np.dstack((mbimage[4,:,:], mbimage[2,:,:], mbimage[1,:,:])).clip(min=0) * 255.0).astype(np.uint8)
        return rgb_uint8
    return mbimage

class IdahoImage(geoio.GeoImage):
    def __init__(self, dg_file_in, derived_dir=None, iid=None, vrt_dir="/home/gremlin/vrt"):
        self.update(data)
        self._snapshot = snapshot
        self._gid = data["id"]
        self._vrt_dir = vrt_dir

    def fetch(self, node="TOAReflectance", level="0"):
        user_bounds = parse_bounds(self._snapshot["bounds"])
        self._user_bounds = user_bounds
        url = build_url(self._gid, node=node, level=level)
        self._url = url
        self._src = rasterio.open(url)
        block_shapes = [(256, 256) for bs in self._src.block_shapes]
        self._roi = roi_from_bbox_projection(self._src, user_bounds, block_shapes=block_shapes)

        window = self._roi.flatten()
        px_bounds = [window[0], window[1], window[0] + window[2], window[1] + window[3] ]
        self._px_bounds = px_bounds
        res = requests.get(url, params={"window": ",".join([str(c) for c in px_bounds])})
        tmp_vrt = os.path.join(self._vrt_dir, ".".join([".tmp", node, level, self._gid + ".vrt"]))
        with open(tmp_vrt, "w") as f:
            f.write(res.content)

        dpath = "/{}_{}_{}".format(self._gid, node, level)
        urls = collect_urls(tmp_vrt)
        darr = build_array(urls, bands=self._src.meta['count'])
        self._snapshot._fileh.close()

        print("Starting parallel fetching... {} chips".format(sum([len(x) for x in urls])))
        with dask.set_options(get=threaded_get):
            darr.to_hdf5(self._snapshot._filename, dpath)
        for key in _curl_pool.keys():
            _curl_pool[key].close()
            del _curl_pool[key]
        print("Fetch complete")

        self._snapshot._fileh = tables.open_file(self._snapshot._filename, mode='r') #reopen snapfile w pytables
        self._snapshot._raw = self._snapshot._fileh.root.raw

        vrt_file = self._generate_vrt(node=node, level=level)
        self._src.close()
        return vrt_file

    def to_geotiff(self, node="TOAReflectance", level="0"):
        im = self.read(node=node, level=level))
        nbands, height, width = im.shape
        if nbands == 8:
            rgb = ms_to_rgb(im)
            rgb = np.rollaxis(rgb, 2, 0)
        elif nbands = 1:
            rgb = im
        else:
            raise TypeError
        if path is None:
            path = os.path.join(self._vrt_dir, ".".join([self._gid, node, level]) + ".tif")
        with rasterio.open(path, "w",
                           driver="GTiff",
                           width=width,
                           height=height,
                           dtype=rgb.dtype,
                           count=nbands) as dst:
            dst.write(rgb)

