try:
    from io import BytesIO
except ImportError:
    from StringIO import cStringIO as BytesIO

import tables
import h5py

import os
import json
import inspect
from functools import partial
from collections import defaultdict
from itertools import groupby
import threading
import contextlib

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

from osgeo import gdal
gdal.UseExceptions()

from matplotlib import pyplot as plt

def plot(img, cmap=None, w=5, h=5):
    #plt.figure(figsize=(w,h)
    plt.axis('off')
    plt.imshow(img, cmap=cmap)
    plt.show()

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

def roi_from_bbox_projection(src, user_bounds, preserve_blocksize=True):
    roi = src.window(*user_bounds)
    if not preserve_blocksize:
        return roi
    blocksafe_roi = rasterio.windows.round_window_to_full_blocks(roi, src.block_shapes)
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


class Image(object):
    def __init__(self, image_id, bounds, node="TOAReflectance", level="0"):
        self._gid = image_id
        self.node = node
        self.level = level
        self._bounds = parse_bounds(bounds)
        self._dir = os.getcwd()#os.path.dirname(os.path.abspath(__file__))
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

        print("Fetching image %s" % self._gid)

        url = build_url(self._gid, node=self.node, level=self.level)
        self._src = rasterio.open(url)
        self._roi = roi_from_bbox_projection(self._src, self._bounds)

        window = self._roi.flatten()
        px_bounds = [window[0], window[1], window[0] + window[2], window[1] + window[3] ]
        res = requests.get(url, params={"window": ",".join([str(c) for c in px_bounds])})
        tmp_vrt = os.path.join(self._dir, ".".join([".tmp", self.node, self.level, self._gid + ".vrt"]))
        with open(tmp_vrt, "w") as f:
            f.write(res.content)

        dpath = "/{}_{}_{}".format(self._gid, self.node, self.level)
        urls = collect_urls(tmp_vrt)
        darr = build_array(urls, bands=self._src.meta['count'])
        self._src.close()

        print("Starting parallel fetching... {} chips".format(sum([len(x) for x in urls])))
        with dask.set_options(get=threaded_get):
            darr.to_hdf5(self._filename, dpath)
        for key in _curl_pool.keys():
            _curl_pool[key].close()
            del _curl_pool[key]
        print("Fetch complete")

        self._generate_vrt()
        os.remove(tmp_vrt)
        return self.vrt

    def _generate_vrt(self):
        vrt = ET.Element("VRTDataset", {"rasterXSize": str(self._roi.num_cols),
                        "rasterYSize": str(self._roi.num_rows)})
        ET.SubElement(vrt, "SRS").text = str(self._src.crs['init']).upper()
        ET.SubElement(vrt, "GeoTransform").text = ", ".join([str(c) for c in self._src.get_transform()])
        for i in self._src.indexes:
            band = ET.SubElement(vrt, "VRTRasterBand", {"dataType": self._src.dtypes[i-1].title(), "band": str(i)})
            src = ET.SubElement(band, "SimpleSource")
            ET.SubElement(src, "SourceFilename").text = "HDF5:{}://{}_{}_{}".format(self._filename, self._gid, self.node, self.level)
            ET.SubElement(src, "SourceBand").text =str(i)
            ET.SubElement(src, "SrcRect", {"xOff": "0", "yOff": "0",
                                           "xSize": str(self._roi.num_cols), "ySize": str(self._roi.num_rows)})
            ET.SubElement(src, "DstRect", {"xOff": "0", "yOff": "0",
                                           "xSize": str(self._roi.num_cols), "ySize": str(self._roi.num_rows)})

            ET.SubElement(src, "SourceProperties", {"RasterXSize": str(self._roi.num_cols), "RasterYSize": str(self._roi.num_rows),
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

    def preview(self):
        data = self.read()
        nbands, x, y = data.shape
        if nbands == 8:
            img = np.dstack((data[4,:,:], data[2,:,:], data[1,:,:])) #.clip(min=0) * 255.0).astype(np.uint8)
            img[img < 0.0] = 0.0
            img[img > 1.0] = 1.0
            img = (255.0*np.power(img, 0.5)).astype('uint8')
            return plot(img)
        else:
            return plot(data, cmap='Greys')


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
        with rasterio.open(path, "w",
                           driver="GTiff",
                           width=width,
                           height=height,
                           dtype=im.dtype,
                           count=nbands) as dst:
            dst.write(im)
        return path


if __name__ == '__main__':
    img = Image('cea67467-f90f-4eb8-85f0-62b875f51dea', bounds='-105.0121307373047,39.7481943650473,-104.99500823974611,39.75656032588025')
    data = img.read()
    print data.shape
