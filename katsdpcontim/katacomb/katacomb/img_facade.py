import logging
import sys

import numpy as np
from pretty import pretty

import FArray
import Image
import Table
import TableList
import TableUtil

from katacomb import (AIPSTable,
                          AIPSHistory,
                          AIPSPath,
                          obit_err,
                          handle_obit_err)

log = logging.getLogger('katacomb')

from katacomb.uv_facade import _history_wrapper

def img_file_mode(mode):
    """ Returns UV file mode given string mode """
    read = 'r' in mode
    write = 'w' in mode

    if read and write:
        return Image.READWRITE
    elif write:
        return Image.WRITEONLY
    # Read by default
    else:
        return Image.READONLY

def open_img(aips_path, mode=None):
    """
    Opens an AIPS/FITS Image file and returns a wrapped :class:`ImageFacade` object.

    Parameters
    ----------
    aips_path: :class:`AIPSPath`
        Obit file object.
    mode(optional): str
        "r" to read, "w" to write, "rw" to read and write.
        Defaults to "r"

    Returns
    -------
    :class:`Image`
        An Obit Image object
    """

    err = obit_err()

    if mode is None:
        mode = "r"

    img_mode = img_file_mode(mode)
    exists = False  # Test if the file exists

    if aips_path.dtype == "AIPS":
        try:
            img = Image.newPAImage(aips_path.label, aips_path.name,
                            aips_path.aclass, aips_path.disk,
                            aips_path.seq, exists, err)
        except Exception:
            raise (ValueError("Error calling newPAImage on '%s'" % aips_path),
                                None, sys.exc_info()[2])
    elif aips_path.dtype == "FITS":
        raise NotImplementedError("newPFImage calls do not currently work")

        try:
            img = Image.newPFImage(aips_path.label, aips_path.name, aips_path.disk,
                            exists, err)
        except Exception:
            raise (ValueError("Error calling newPFImage on '%s'" % aips_path),
                                None, sys.exc_info()[2])
    else:
        raise ValueError("Invalid dtype '{}'".format(aips_path.dtype))

    handle_obit_err("Error opening '%s'" % aips_path, err)

    err_msg = "Error opening '%s'" % aips_path

    try:
        img.Open(img_mode, err)
    except Exception:
        raise (ValueError(err_msg), None, sys.exc_info()[2])

    handle_obit_err(err_msg, err)

    return img

def img_factory(**kwargs):
    """
    Factory for creating an AIPS Image file.

    Parameters
    ----------
    aips_path : :class:`AIPSPath`
        Path to the AIPS file
    mode (optional) : string
        File opening mode passed to :func:`uv_open`.

    Returns
    -------
    :class:`ImageFacade`
        Object representing the Image.
    """

    try:
        ofile = kwargs.pop('aips_path')
    except KeyError:
        raise ValueError("No 'aips_path' argument supplied")

    mode = kwargs.pop('mode', 'r')

    img = open_img(ofile, mode)
    return ImageFacade(img)

# Force order to ra, dec, freq, stokes
OBITIMAGEMF_ORDER = ["jlocr", "jlocd", "jlocf", "jlocs"]
OBITIMAGEMF_CTYPE = ["RA---SIN", "DEC--SIN", "SPECLNMF", "STOKES"]

def obit_image_mf_planes(imgf):
    """
    Generator returning SPECLNMF planes of an ObitImageMF AIPS Image.

    Parameters
    ----------
    imgf : :class:`ImageFacade`
        Image facade

    Yields
    ------
    np.ndarray
        Numpy arrays of shape (l,m,stokes)

    """
    desc = imgf.Desc.Dict
    inaxes = desc['inaxes']

    # Make sure imgf is an ObitImageMF
    imgf._requireObitImageMF()

    l = inaxes[desc['jlocr']]
    m = inaxes[desc['jlocd']]
    speclnmf = inaxes[desc['jlocf']]
    nstokes = inaxes[desc['jlocs']]

    for slnmf in range(1, speclnmf + 1):
        imgs = []

        for stokes in range(1, nstokes + 1):
            # Obit plane selects on axis 2 and onwards.
            # So we get the [l,m] axes by default.
            plane = [slnmf, stokes, 1, 1, 1]
            imgf.GetPlane(None, plane)
            imgs.append(imgf.np_farray.reshape(l, m).copy())

        # Yield arrays stacked on stokes
        yield np.stack(imgs, axis=2)

def obit_image_mf_rms(imgf):
    """
    Return the RMS in each image frequency plane of an ObitImageMF

    Parameters
    ----------
    imgf : :class:`ImageFacade`
        Image facade

    Returns
    -------
    np.ndarray
        RMS per plane per Stokes (nplanes,stokes)
    """
    # ObitImageMF?
    imgf._requireObitImageMF()

    desc = imgf.Desc.Dict
    inaxes = desc["inaxes"]
    nimplanes = inaxes[desc["jlocf"]]
    nstokes = inaxes[desc["jlocs"]]

    rms = np.empty((nimplanes, nstokes,), dtype=np.float32)
    for fplane in range (1, nimplanes + 1):
        for stokes in range(1, nstokes + 1):
            plane = [fplane, stokes, 1, 1, 1]
            imgf.GetPlane(None, plane)
            rms[fplane - 1, stokes - 1] = imgf.FArray.RMS
    return rms

class ImageFacade(object):
    def __init__(self, img, **kwargs):
        self._err = err = obit_err()
        self._open_logic(img, err, **kwargs)

    def _open_logic(self, img, err, **kwargs):
        """
        Peforms logic for opening a Image file

        * Opening the Image File if given an AIPS path.
        * Setting up the AIPS Path if given a Image file.
        * Open any Tables attached to the Image file.
        """

        # Given an AIPSPath. open it.
        if isinstance(img, AIPSPath):
            self._aips_path = img
            mode = kwargs.pop('mode', 'r')
            self._img = img = open_img(img, mode=mode)
        # Given an Obit Image file.
        # Construct an AIPSPath
        elif isinstance(img, Image.Image):
            # FITS and AIPS files have different properties
            if img.FileType == "FITS":
                name = img.FileName
                aclass = None
                seq = None
            elif img.FileType == "AIPS":
                name = img.Aname
                aclass = img.Aclass
                seq = img.Aseq
            else:
                raise ValueError("Invalid FileType '%s'" % img.FileType)

            self._aips_path = AIPSPath(name, img.Disk, aclass,
                                       seq, dtype=img.FileType)

            self._img = img
        else:
            raise TypeError("Invalid type '%s'. "
                            "Must be Obit Image object "
                            "or an AIPSPath." % type(img))

        # Open tables attached to this UV file.
        tables = TableList.PGetList(img.TableList, err)
        handle_obit_err("Error getting '%s' table list" % self.name)

        # History tables don't work like the other tables
        ignored_tables = ["AIPS HI"]

        self._tables = {name: AIPSTable(img, name, version, 'r', err)
                        for version, name in tables
                        if name not in ignored_tables}

        self._tables["AIPS HI"] = AIPSHistory(img, err)

    @property
    def tables(self):
        return self._tables

    def attach_table(self, name, version, **kwargs):
        self._tables[name] = AIPSTable(self.img, name, version, 'r',
                                       self._err, **kwargs)

    @property
    def aips_path(self):
        return self._aips_path

    @property
    def name(self):
        return str(self._aips_path)

    def close(self):
        """ Closes the wrapped Image file """

        # Close all attached tables
        for table in self._tables.values():
            table.close()

        self._tables = {}

        err_msg = "Exception closing image file '%s'" % self.name

        try:
            self._img.Close(self._err)
        except AttributeError:
            # Already closed
            return
        except Exception:
            raise (Exception(err_msg), None, sys.exc_info()[2])

        handle_obit_err(err_msg, self._err)
        self._clear_img()

    @property
    def img(self):
        try:
            return self._img
        except AttributeError:
            self._open_logic(self._aips_path, self._err)

        return self._img

    def _clear_img(self):
        """
        Calls :code:`del` on the wrapped Image object and sets it to None.
        Without this, internal resources on the Image object
        are not released and subsequent calls to *unrelated*
        objects will fail, claiming that the Image object
        is *not* a Image object.
        """
        try:
            del self._img
        except AttributeError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, etype, evalue, etraceback):
        self.close()

    def append_history(self, msg):
        """
        Appends ``msg`` to the HISTORY table of this UV file.

        Long lines will be broken at 70 chars.
        """
        for line in _history_wrapper.wrap(msg):
            self._tables["AIPS HI"].append(line)

    @property
    def FArray(self):
        return self.img.FArray

    @property
    def np_farray(self):
        try:
            buf = FArray.PGetBuf(self.img.FArray)
        except Exception:
            raise (Exception("Exception getting float array buffer "
                            " on image '%s'" % self.name),
                                None, sys.exc_info()[2])

        return np.frombuffer(buf, count=-1, dtype=np.float32)

    def Open(self, mode):
        err_msg = "Error opening Image file '%s'" % self.name

        try:
            self.img.Open(mode, self._err)
        except Exception:
            raise (Exception(err_msg), None, sys.exc_info()[2])

        handle_obit_err(err_msg, self._err)

    def GetPlane(self, array, plane):
        err_msg = ("Error getting plane '%s' "
                    "from image '%s'" % (plane, self.name))

        try:
            self.img.GetPlane(array, plane, self._err)
        except Exception:
            raise (Exception(err_msg), None, sys.exc_info()[2])

        handle_obit_err(err_msg, self._err)

    @property
    def Desc(self):
        return self.img.Desc

    @property
    def List(self):
        return self.img.Desc.List

    def Close(self):
        self.close()

    def Zap(self):
        err_msg = "Exception zapping image file '%s'" % self.name

        try:
            self.img.Zap(self._err)
        except Exception:
            raise (Exception(err_msg), None, sys.exc_info()[2])

        handle_obit_err("Error deleting Image file '%s'" % self.name, self._err)
        self._clear_img()

    def MergeCC(self):
        """
        Merge the positionally coincidental clean components in the
        attached CC table and attach the merged table.
        """
        err_msg = "Exception merging CC Table in '%s'" % self.name

        cctab = self.tables["AIPS CC"]
        init_nrow = cctab.nrow
        # Create an empty table to hold the merged table
        merged_cctab = self.img.NewTable(Table.READWRITE,
                                         "AIPS CC",
                                         cctab.version + 1,
                                         self._err)
        try:
            TableUtil.PCCMerge(cctab._table, merged_cctab, self._err)
        except Exception:
            raise (Exception(err_msg), None, sys.exc_info()[2])
        handle_obit_err(err_msg, self._err)

        # Attach merged version of CC Table
        self.attach_table("AIPS CC", cctab.version + 1)
        merged_cctab = self.tables["AIPS CC"]
        log.info("Merged %d CCs to %d for %s",
             init_nrow, merged_cctab.nrow, self.name)
        return merged_cctab

    def isObitImageMF(self):
        """
        Check if this image is an ObitImageMF.

        The CTYPE of an ObitImageMF AIPS Image looks like
        :code:`["RA---SIN", "DEC--SIN", "SPECLNMF", "STOKES", "", ""]` and
        respectively correspond to the l, m, spectral logarithmic and stokes parameters.
        """
        imdesc = self.Desc.Dict
        ctype = [s.strip() for s in imdesc["ctype"]]
        # Order ctype by defined order
        locs = [imdesc[label] for label in OBITIMAGEMF_ORDER]
        ord_ctype = [ctype[loc] for loc in locs]
        return ord_ctype == OBITIMAGEMF_CTYPE

    def _requireObitImageMF(self):
        """Raise ValueError if this is not an ObitImageMF
        """
        if not self.isObitImageMF():
            raise ValueError("'%s' doesn't appear to be an ObitImageMF. "
                             "Descriptor is '%s'." % (
                             self.aips_path, pretty(self.Desc.Dict)))
