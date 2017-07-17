import logging
import re

logging.basicConfig(level=logging.INFO)

# Matches expressions of the form
# $Key = Sources Str (16, 30)
VAR_EXPR = re.compile("^\$Key\s+="
                      "\s+(?P<variable>\S+)\s+"
                      "(?P<type>\S+)\s+"
                      "\((?P<dimensions>[0-9,]+)\)\s*"
                      "#*\s*(?P<comment>.*)$")

# Should match groups in VAR_EXPR, in the correct order
REQUIRED_GROUPS = ["variable", "type", "dimensions"]

def parse_aips_config(aips_cfg_file):
    """
    Parses an AIPS config file.

    Entries look something like this:

    ```
    $Key = Sources Str (2, 30) # Sources to image
    DEEP2
    PKS 1934-638
    ```

    which defines a `Sources` variable which is a
    length 2 list of strings of length 30. The contents
    of the variable follow the definition on each line.
    """

    # Convert AIPS types to python types
    aipstype2py = {
        "Boo": bool,
        "Str": str,
        "Int": int,
        "Flt": float
    }

    # Convert AIPS boolean options to python bools
    aipsbool2py = { "T": True, "F": False }

    def _parse_dims(dim_str):
        """ Converts "20,30" into (20,30) for e.g. """
        return tuple(int(d.strip()) for d in dim_str.split(","))

    def _extract_group(group_dict, group, line):
        """ Handle key errors nicely for group_dict access """
        try:
            return group_dict[group]
        except KeyError:
            raise KeyError("AIPS %s was not present in "
                            "AIPS variable definition '%s'" %
                                (group, line))

    def _parse_file(f):
        """ Parse config file line by line """
        py_varname = None
        py_type = None
        py_dims = None
        py_comment = ""

        D = {}

        for line_nr, line in enumerate(f):
            line = line.strip()
            # Search for comment position
            cp = line.rfind("#")
            cp = len(line) if cp == -1 else cp

            # Ignore empty lines or lines that are merely space
            if not line[0:cp] or line[0:cp].isspace():
                continue
            # We've found an AIPS variable definition
            elif line.startswith("$Key"):
                match = VAR_EXPR.match(line)

                if not match:
                    raise ValueError("Error parsing AIPS variable "
                                     "definition '%s'" % line)

                gd = match.groupdict()

                # Extract variable name, type, and dimensions
                aips_varname, aips_type, aips_dims = (
                    _extract_group(gd, g, line)
                    for g in REQUIRED_GROUPS)

                # Assign variable name
                py_varname = aips_varname

                # Assign variable type
                try:
                    py_type = aipstype2py[aips_type]
                except KeyError:
                    raise ValueError("No known conversion "
                                     "to python type "
                                     "for AIPS type '%s'" % aips_type)

                # Construct a shape tuple
                py_dims = _parse_dims(aips_dims)

                # Try to extract the comment too, but don't worry if it's not there
                try:
                    py_comment = _extract_group(gd, "comment", line)
                except KeyError:
                    pass

                # print "Parsed", aips_varname, py_type, py_dims
            elif py_varname is None:
                continue
            else:
                # Get last dimension size
                L = len(py_dims)
                N = py_dims[-1]

                if py_type in (int, float):
                    value = [py_type(s.strip()) for
                             s in line.split(" ")][:N]

                    if len(value) == 1:
                        value = value[0]

                elif py_type in (bool,):
                    value = aipsbool2py[line.strip()]
                elif py_type in (str,):
                    value = line[0:N]

                # There should only be one dimension
                if L == 1:
                    D[py_varname] = value
                # Two dimensions, append this value to the list
                elif L == 2:
                    values = D.get(py_varname, [])
                    values.append(value)
                    D[py_varname] = values
                else:
                    raise ValueError("Not handling variables with rank %s" % L)

        # from pprint import pprint
        # pprint(D)

        return D

    with open(aips_cfg_file, "r") as f:
        try:
            return _parse_file(f)
        except BaseException as e:
            logging.exception("Parsing Error")
            raise

def aips_disk_config(infile, fitsdir, aipsdir):
    """
    Returns an AIPS disk configuration suitable for
    use with Obit
    """

    import os

    if infile is None:
        infile = ""
        logging.warn("No 'infile' was provided, setting to '%s'" % infile)

    if fitsdir is None:
        fitsdir = os.getcwd()
        logging.warn("No 'fitsdir' was provided, setting to '%s'" % fitsdir)

    if aipsdir is None:
        aipsdir = os.getcwd()
        logging.warn("No 'aipsdir' was provided, setting to '%s'" % fitsdir)

    # Override AIPS disk configuration options
    cfg = {
        "DataType": "FITS",
        "FITSdirs": [fitsdir],
        "AIPSdirs": [aipsdir],
        "inFile": infile,
        "inDisk": 0,
        "inSeq": 0,
        "outFile": ".out.fits",
        "out2File": ".out.fits",
    }

    # Set output disk options to input disk options
    cfg.update({
        "outDType": cfg["DataType"],
        "outDisk": cfg["inDisk"],
        "outSeq": cfg["inSeq"],
        "out2Disk": cfg["inDisk"],
        "out2Seq": cfg["inSeq"],
    })

    return cfg

def aips_user():
    """ Get the AIPS user """
    import OSystem

    try:
        return OSystem.PGetAIPSuser()
    except Exception as e:
        logging.exception("Exception getting AIPS User. "
                          "Returning 105 instead")
        return 105

def aips_cfg(aips_cfg_file, infile=None, fitsdir=None, aipsdir=None):
    """ Construct a usable AIPS configuration """

    # Parse the configuration file
    cfg = parse_aips_config(aips_cfg_file)

    # Set the user file
    cfg['userno'] = aips_user()

    # Set up some reasonable AIPS disk config
    cfg.update(aips_disk_config(infile, fitsdir, aipsdir))

    # These don't work with OBIT for some reason
    for k in ('nFITS','nAIPS', 'AIPSuser'):
        cfg.pop(k, None)

    return cfg