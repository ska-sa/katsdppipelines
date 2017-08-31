_VALID_DISK_TYPES = ["AIPS", "FITS"]

class ObitFile(object):
    """
    A class representing the naming properties of
    either an AIPS or FITS file.

    Note instances of this merely abstract and encapsulate
    the naming properties of AIPS and FITS files,
    making it easier to pass this data as arguments.
    They do not abstract file access.

    Also, while FITS files don't technically have classes
    or sequences, these are defaulted to "fits" and 1, respectively.
    """
    def __init__(self, name, disk, aclass=None, seq=None, dtype=None):
        """
        Constructs an :class:`ObitFile`.

        Parameters
        ----------
        name: string
            File name
        disk: integer
            Disk on which the file is located
        aclass (optional): string
            AIPS file class
        seq (optional): integer
            AIPS file sequence number
        dtype (optional): string
            Disk type. Should be "AIPS" or "FITS".
            Defaults to "AIPS" if not provided.

        """
        if dtype is None:
            dtype = "AIPS"

        if dtype == "AIPS":
            # Provide sensible defaults for missing class and sequence
            self._aclass = "default" if aclass is None else aclass
            self._seq = 1 if seq is None else seq
        elif dtype == "FITS":
            # FITS file don't have class or sequences,
            # just provide something sensible
            self._aclass = "fits"
            self._seq = 1
        else:
            raise ValueError("Invalid disk type '%s'. "
                            "Should be one of '%s'" % (
                                dtype, _VALID_DISK_TYPES))

        self._dtype = dtype
        self._name = name
        self._disk = disk

    def copy(self):
        """ Returns a copy of this object """
        return ObitFile(self._name, self._disk, self._aclass,
                        self._seq, self.dtype)

    @property
    def name(self):
        """ File name """
        return self._name

    @name.setter
    def name(self, value):
        """ File name """
        self._name = value

    @property
    def disk(self):
        """ File disk """
        return self._disk

    @disk.setter
    def disk(self, value):
        """ File disk """
        self._disk = value

    @property
    def dtype(self):
        """ File disk type """
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        """ File disk type """
        if not value in _VALID_DISK_TYPES:
            raise ValueError("Invalid disk type '%s'. "
                            "Should be one of '%s'" % (
                                dtype, _VALID_DISK_TYPES))

        self._dtype = value

    @property
    def aclass(self):
        """ File class """
        return self._aclass

    @aclass.setter
    def aclass(self, value):
        """ File class """
        self._aclass = value

    @property
    def seq(self):
        """ File sequence number """
        return self._seq

    @seq.setter
    def seq(self, value):
        """ File sequence number """
        self._seq = value
