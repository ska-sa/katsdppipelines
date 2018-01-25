import ast
import inspect
import os

from katacomb import obit_err, handle_obit_err

_VALID_DISK_TYPES = ["AIPS", "FITS"]

def next_seq_nr(aips_path):
    """
    Returns the highest available sequence number for which
    a catalogue entry does not exist

    Parameters
    ----------
    aips_path : :class:`AIPSPath`
        An AIPS path

    Returns
    -------
    integer
        Highest sequence number
    """
    from AIPSDir import PHiSeq, PTestCNO
    from OSystem import PGetAIPSuser


    err = obit_err()
    aips_user = PGetAIPSuser()

    hi_seq = PHiSeq(Aname=aips_path.name, user=aips_user,
                    disk=aips_path.disk, Aclass=aips_path.aclass,
                    Atype=aips_path.atype, err=err)

    handle_obit_err("Error finding highest sequence number", err)

    while True:
        cno = PTestCNO(disk=aips_path.disk, user=aips_user,
            Aname=aips_path.name, Aclass=aips_path.aclass,
            Atype=aips_path.atype, seq=hi_seq,
            err=err)

        handle_obit_err("Error finding catalogue entry", err)

        if cno == -1:
            return hi_seq

        hi_seq += 1


def _check_disk_type(dtype, check=True):
    """
    Checks that `dtype` is either "AIPS" or FITS",
    raising a `ValueError` if this is not the case.

    Parameters
    ----------
    dtype: string
        AIPS disk type
    check: boolean
        False to avoid the check and raise ValueError anyway

    Raises
    ------
    ValueError
        Raised if `dtype` is not "AIPS" or "FITS"
        or if check is `False`
    """
    if not check or dtype not in _VALID_DISK_TYPES:
        raise ValueError("Invalid disk type '%s'. "
                         "Should be one of '%s'" % (
                             dtype, _VALID_DISK_TYPES))

class AIPSPath(object):
    """
    A class representing the path properties of
    either an AIPS or FITS file.

    Instances merely abstract and encapsulate
    the naming properties of AIPS and FITS files,
    making it easier to pass this data as arguments.
    They do not abstract file access.

    Also, while FITS files don't technically have classes
    or sequences, these are defaulted to "fits" and 1, respectively.
    """

    def __init__(self, name, disk=1, aclass="aips",
                 seq=1, atype="UV",
                 label="katuv", dtype="AIPS"):
        """
        Constructs an :class:`AIPSPath`.

        Parameters
        ----------
        name : string
            File name
        disk (optional) : integer
            Disk on which the file is located. Defaults to 1.
        aclass (optional) : string
            AIPS file class.
        seq (optional) : integer
            AIPS file sequence number. Defaults to 1.
        atype (optional) : str
            AIPS file type. Typically either 'UV' or 'MA' for
            UV or Image data, respectively. Defaults to 'UV'
        label (optional) : string
            AIPS label. Defaults to 'katuv'.
        dtype (optional) : string
            Data type. Should be "AIPS" or "FITS".
            Defaults to "AIPS" if not provided.

        """
        self.name = name
        self.disk = disk
        self.aclass = aclass
        self.seq = seq
        self.label = label
        self.atype = atype
        self.dtype = dtype

        if dtype == "AIPS":
            pass
        elif dtype == "FITS":
            # FITS file don't have class or sequences,
            # just provide something sensible
            self.aclass = "fits"
            self.seq = 1
        else:
            _check_disk_type(dtype, False)


    def copy(self, name=None, disk=None, aclass=None,
             seq=None, atype=None, label=None, dtype=None):
        """
        Returns a copy of this object. Supplied parameters can
        override properties transferred to the new object.
        """
        return AIPSPath(name=self.name if name is None else name,
                        disk=self.disk if disk is None else disk,
                        aclass=self.aclass if aclass is None else aclass,
                        seq=self.seq if seq is None else seq,
                        atype=self.atype if atype is None else atype,
                        label=self.label if label is None else label,
                        dtype=self.dtype if dtype is None else dtype)

    def __str__(self):
        """ String representation """
        if self.dtype == "AIPS":
            return "%s.%s.%s.%s on AIPS %d" % (self.name, self.aclass,
                                            self.atype, self.seq, self.disk)
        elif self.dtype == "FITS":
            return "%s.%s on FITS %d" % (self.name, self.atype, self.disk)
        else:
            _check_disk_type(self.dtype, False)

    __repr__ = __str__

    def task_input_kwargs(self):
        """
        Returns
        -------
        dict
            Keyword arguments suitable for applying
            to an ObitTask as an input file.
        """
        if self.dtype == "AIPS":
            return {"DataType": self.dtype,
                    "inName": self.name,
                    "inClass": self.aclass,
                    "inSeq": self.seq,
                    "inDisk": self.disk}
        elif self.dtype == "FITS":
            return {"DataType": self.dtype,
                    "inFile": self.name}
        else:
            _check_disk_type(self.dtype, False)


    def task_output_kwargs(self, name=None, disk=None, aclass=None,
                           seq=None, dtype=None):
        """
        Returns
        -------
        dict
            Keyword arguments suitable for applying
            to an ObitTask as an output file.
        """
        dtype = self.dtype if dtype is None else dtype

        if dtype == "AIPS":
            return {"outDType": dtype,
                    "outName": self.name if name is None else name,
                    "outClass": self.aclass if aclass is None else aclass,
                    "outSeq": self.seq if seq is None else seq,
                    "outDisk": self.disk if disk is None else disk}
        elif dtype == "FITS":
            return {"outDType": dtype,
                    "outFile": self.name}
        else:
            _check_disk_type(dtype, False)


    def task_output2_kwargs(self, name=None, disk=None, aclass=None,
                            seq=None, dtype=None):
        """
        Returns
        -------
        dict
            Keyword arguments suitable for applying
            to an ObitTask as an output file.
        """
        dtype = self.dtype if dtype is None else dtype

        # NB. There doesn't seem to be an out2DType

        if dtype == "AIPS":
            return {"out2Name": self.name if name is None else name,
                    "out2Class": self.aclass if aclass is None else aclass,
                    "out2Seq": self.seq if seq is None else seq,
                    "out2Disk": self.disk if disk is None else disk}
        elif dtype == "FITS":
            return {"out2File": self.name}
        else:
            _check_disk_type(dtype, False)


_AIPS_PATH_TUPLE_ARGS = [a for a in inspect.getargspec(AIPSPath.__init__).args
                                                            if not a == "self"]
_AIPS_PATH_TUPLE_FORMAT = "(%s)" % ",".join(_AIPS_PATH_TUPLE_ARGS)
_AIPS_PATH_HELP = ("AIPS path should be a tuple of "
                    "names or numbers of the form "
                    "%s" % _AIPS_PATH_TUPLE_FORMAT)


def parse_aips_path(aips_path_str):
    """
    Parses a string describing an AIPS Path

    Parameters
    ----------
    aips_path_str : string
        String of the form
        %(fmt)s

    Returns
    -------
    :class:`AIPSPath`
        An AIPS Path object
    """
    stmts = ast.parse(aips_path_str, mode='single').body

    if not isinstance(stmts[0], ast.Expr):
        raise ValueError(_AIPS_PATH_HELP)

    sequence = stmts[0].value

    if not isinstance(sequence, (ast.Tuple, ast.List)):
        raise ValueError(_AIPS_PATH_HELP)

    xformed = []

    for value in sequence.elts:
        if isinstance(value, ast.Name):
            xformed.append(None if value.id == "None" else value.id)
        elif isinstance(value, ast.Num):
            xformed.append(value.n)
        else:
            raise ValueError(_AIPS_PATH_HELP)

    return AIPSPath(**dict(zip(_AIPS_PATH_TUPLE_ARGS, xformed)))

try:
    parse_aips_path.__doc__ %= {'fmt' : _AIPS_PATH_TUPLE_FORMAT}
except KeyError:
    # For python -OO
    pass
