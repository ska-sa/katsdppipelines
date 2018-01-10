import os

from katacomb import obit_err

from AIPSDir import PHiSeq, PTestCNO
from OSystem import PGetAIPSuser

_VALID_DISK_TYPES = ["AIPS", "FITS"]

def _highest_seq_nr(name, disk, aclass, atype):
    """
    Returns the highest sequence number

    Parameters
    ----------
    name : str
        AIPS name
    disk : integer
        AIPS disk
    aclass : str
        AIPS class
    atype : str
        AIPS type

    Returns
    -------
    integer
        Highest sequence number
    """

    return PHiSeq(Aname=name, user=PGetAIPSuser(),
                    disk=disk, Aclass=aclass,
                    Atype=atype, err=obit_err())

def _catalogue_entry(name, disk, aclass, seq, atype):
    """
    Returns catalogue entry if it exists for the supplied arguments,
    otherwise -1

    Parameters
    ----------
    name : str
        AIPS name
    disk : integer
        AIPS disk
    aclass : str
        AIPS class
    seq : integer
        AIPS sequence number
    atype : str
        AIPS type

    Returns
    -------
    integer
        catalogue entry, else -1 if it does not exist

    """
    return PTestCNO(disk=disk, user=PGetAIPSuser(),
        Aname=name, Aclass=aclass, Atype=atype, seq=seq,
        err=obit_err())

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
                 seq=None, atype="UV",
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
            AIPS file sequence number. If None or less than 1,
            the next available sequence number will be selected.
            Defaults to None.
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
        self.label = label
        self.atype = atype
        self.dtype = dtype

        if dtype == "AIPS":
            # Provide sensible default  for missing sequence
            if seq is None or seq < 1:
                seq = _highest_seq_nr(name, disk, aclass, atype)
                cno = _catalogue_entry(name, disk, aclass, seq, atype)

                # Choose next highest seq nr if no catalogue entry exists
                if not cno == -1:
                    seq += 1

            self.seq = seq
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
