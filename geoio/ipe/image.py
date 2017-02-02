try:
    from io import BytesIO
except ImportError:
    from StringIO import cStringIO as BytesIO

try:
    import signal
    signal.signal(signal.SIGPIPE, signal.SIG_IGN)
except ImportError:
    pass

try:
    from IPython.core.display import clear_output
    have_ipython = True
except ImportError:
    have_ipython = False


import tables
import h5py

import os, sys
import json
import inspect
from functools import partial
from collections import defaultdict
from itertools import groupby
import threading
import contextlib

import warnings
warnings.filterwarnings('ignore')

from geoio.plotting import imshow

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

from osgeo import gdal
gdal.UseExceptions()

from matplotlib import pyplot as plt

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
    return np.fromstring(json.dumps(data), dtype="uint8")

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
def load_url(url, progressFn, bands=8):
    thread_id = threading.current_thread().ident
    _curl = _curl_pool[thread_id]
    buf = BytesIO()
    _curl.setopt(_curl.URL, url)
    _curl.setopt(_curl.WRITEDATA, buf)
    _curl.setopt(pycurl.NOSIGNAL, 1)
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

def build_array(urls, progressFn, bands=8):
    total = sum([len(x) for x in urls])
    buf = da.concatenate(
        [da.concatenate([da.from_delayed(load_url(url, progressFn, bands=bands), (bands,256,256), np.float32) for u, url in enumerate(row)],
                        axis=1) for r, row in enumerate(urls)], axis=2)
    return buf


class Image(object):
    def __init__(self, image_id, bounds, node="TOAReflectance", level="0"):
        self._gid = image_id
        self.node = node
        self.level = level
        self._bounds = parse_bounds(bounds)
        self._dir = os.getcwd()
        self._filename = os.path.join(self._dir, self._gid + ".h5")
        self.vrt = self._vrt()
        if not os.path.exists(self.vrt):
            self.fetch()

    def fetch(self):
        try:
            os.remove(self.vrt)
            os.remove(self._filename)
        except:
            pass

        print("Fetching image")

        url = build_url(self._gid, node=self.node, level=self.level)
        self._src = rasterio.open(url)
        block_shapes = [(256, 256) for bs in self._src.block_shapes]
        self._roi = roi_from_bbox_projection(self._src, self._bounds, block_shapes=block_shapes)
        window = self._roi.flatten()
        px_bounds = [window[0], window[1], window[0] + window[2], window[1] + window[3] ]
        res = requests.get(url, params={"window": ",".join([str(c) for c in px_bounds])})
        tmp_vrt = os.path.join(self._dir, ".".join([".tmp", self.node, self.level, self._gid + ".vrt"]))
        with open(tmp_vrt, "w") as f:
            f.write(res.content)

        dpath = "/{}_{}_{}".format(self._gid, self.node, self.level)
        urls = collect_urls(tmp_vrt)
        if len(urls):
            self._total = sum([len(x) for x in urls])
            self._current = 0
            self.darr = build_array(urls, self._reportProgress(), bands=self._src.meta['count'])
            self._src.close()
        
            print("Starting parallel fetching... {} chips".format(self._total))
            with dask.set_options(get=threaded_get):
                self.darr.to_hdf5(self._filename, dpath)
            for key in _curl_pool.keys():
                _curl_pool[key].close()
                del _curl_pool[key]
            print("Fetch Complete")

            self._generate_vrt()
            os.remove(tmp_vrt)
            return self.vrt
        else:
            try:
                os.remove(tmp_vrt)
                os.remove(self.vrt)
                os.remove(self._filename)
            except:
                pass
            print("No data intersection within given bounds")
            return None
            
      


    def _reportProgress(self):
        def fn():
            self._current = self._current + 1
            if have_ipython:
                try:
                    clear_output()
                    print '%d%s' % (int((float(self._current) / float(self._total)) * 100.0), '%')
                    sys.stdout.flush()
                except Exception:
                    pass
        return fn

    def _generate_vrt(self):
        cols = str(self.darr.shape[-1])
        rows = str(self.darr.shape[1])
        (minx, miny, maxx, maxy) = rasterio.windows.bounds(self._roi, self._src.transform)
        affine = [c for c in rasterio.transform.from_bounds(minx, miny, maxx, maxy, int(cols), int(rows))]
        transform = [affine[2], affine[4], 0.0, affine[5], 0.0, affine[4]]
        
        vrt = ET.Element("VRTDataset", {"rasterXSize": cols,
                        "rasterYSize": rows})
        ET.SubElement(vrt, "SRS").text = str(self._src.crs['init']).upper()
        ET.SubElement(vrt, "GeoTransform").text = ", ".join(map(str, transform))
        for i in self._src.indexes:
            band = ET.SubElement(vrt, "VRTRasterBand", {"dataType": self._src.dtypes[i-1].title(), "band": str(i)})
            src = ET.SubElement(band, "SimpleSource")
            ET.SubElement(src, "SourceFilename").text = "HDF5:{}://{}_{}_{}".format(self._filename, self._gid, self.node, self.level)
            ET.SubElement(src, "SourceBand").text =str(i)
            ET.SubElement(src, "SrcRect", {"xOff": "0", "yOff": "0",
                                           "xSize": cols, "ySize": rows})
            ET.SubElement(src, "DstRect", {"xOff": "0", "yOff": "0",
                                           "xSize": cols, "ySize": rows})

            ET.SubElement(src, "SourceProperties", {"RasterXSize": cols, "RasterYSize": rows,
                                                    "BlockXSize": "128", "BlockYSize": "128", "DataType": self._src.dtypes[i-1].title()})
        vrt_str = ET.tostring(vrt)

        with open(self.vrt, "w") as f:
            f.write(vrt_str)

        return self.vrt

    def _vrt(self):
        return os.path.join(self._dir, ".".join([self._gid, self.node, str(self.level) + ".vrt"]))

    @contextlib.contextmanager
    def open(self):
        if os.path.exists(self.vrt):
            with rasterio.open(self.vrt) as src:
                yield src
        else:
            print("fetching image from vrt, writing to snapshot file and generating vrt reference")
            with rasterio.open(self.fetch()) as src:
                yield src

    def preview(self, stretch=[0.02, 0.98]):
        plt.axis('off') 
        data = self.read()
        if data.shape[0] == 8:
            img = np.stack((data[4,:,:], data[2,:,:], data[1,:,:]))
            return imshow(img)
        else:
            return imshow(data[0,:,:])

    def read(self, bands=[], **kwargs):
        for band in bands:
            if not isinstance(band, int):
                raise TypeError("Band arguments must be passed as integers")
        with self.open(**kwargs) as src:
            if len(bands) > 0:
                return src.read(bands)
            return src.read()

    def geotiff(self, path=None, dtype=None):
        im = self.read()
        if dtype is not None:
          im = im.astype(dtype)
        nbands, height, width = im.shape
        if path is None:
            path = os.path.join(self._dir, ".".join([self._gid, self.node, self.level]) + ".tif")

        with rasterio.open(self.vrt) as src:
            im = src.read()
            if dtype is not None:
                im = im.astype(dtype)
            if path is None:
                path = os.path.join(self._dir, self._name + ".tif")
            meta = src.meta.copy()
            meta.update({'driver': 'GTiff'})
            with rasterio.open(path, "w", **meta) as dst:
                dst.write(im)
        return path

if __name__ == '__main__':
    # MULTI
    #img = Image('cea67467-f90f-4eb8-85f0-62b875f51dea', bounds='-105.0121307373047,39.7481943650473,-104.99500823974611,39.75656032588025')
    #img = image = Image('94427d3e-8e7b-4452-a152-fc533e3c16b7', bounds='-0.08574485778808595,51.50158353472559,-0.06737709045410158,51.512909075833434')
    #img = Image('59f87923-7afa-4588-9f59-dc5ad8b821b0', bounds='-0.07840633392333984,51.506739141893,-0.07396459579467775,51.50955711581998')
    #img = Image('dc11bd43-401a-40c5-b937-a96cb44fe26c', bounds='-0.07840633392333984,51.506739141893,-0.07396459579467775,51.50955711581998')

    # PAN
    img = Image('c9e5a557-911c-4078-97af-3724cbf76a65', bounds='-105.26801347732545,40.00817393602954,-105.26406526565553,40.01108296862613')
    data = img.read()
    print data.shape
