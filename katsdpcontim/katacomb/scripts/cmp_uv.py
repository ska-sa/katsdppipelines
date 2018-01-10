"""
Mainly exists to compare export from
``legacy_export.py`` to the newer ``uv_export.py``.
"""

import argparse
import logging
from pprint import pprint, pformat

import numpy as np

from katacomb import (KatdalAdapter, UVFacade,
                            handle_obit_err,
                            obit_context,
                            obit_err)

import UV

log = logging.getLogger('katacomb')

def create_parser():
    parser = argparse.ArgumentParser(description="Good for comparing "
                                                 "files exported by 'h5touvfits.py' "
                                                 "and 'uv_export.py")
    parser.add_argument("-n1", "--name1", help="First AIPS file name")
    parser.add_argument("-c1", "--class1", default="raw",
                                        help="First AIPS file class")
    parser.add_argument("-d1", "--disk1", type=int, default=1,
                                        help="First AIPS file disk")
    parser.add_argument("-s1", "--seq1", type=int, default=1,
                                        help="First AIPS file sequence")

    parser.add_argument("-n2", "--name2", help="Second AIPS file name")
    parser.add_argument("-c2", "--class2", default="legacy",
                                        help="Second AIPS file class")
    parser.add_argument("-d2", "--disk2", type=int, default=1,
                                        help="Second AIPS file disk")
    parser.add_argument("-s2", "--seq2", type=int, default=1,
                                        help="Second AIPS file sequence")

    parser.add_argument("--nvispio", type=int, default=1024)
    parser.add_argument("--iloct-rtol", type=float, default=1e-3,
                        help="Relative tolerance for comparing the time "
                            "random parameter. Set lower because "
                            "legacy time's have slightly different "
                            "starting positions.")
    parser.add_argument("-N", type=int, default=5,
                                        help="Number of first/last values "
                                            "to print on difference")

    return parser

args = create_parser().parse_args()

# Assume similar names when comparing
if args.name2 is None:
    args.name2 = args.name1

def _sanity_check_desc(d1, d2, attrs=None):
    """ Sanity check descriptors """
    if attrs is None:
        attrs = ['nvis', 'numVisBuff', 'firstVis', 'lrec']
        attrs.extend([k for k in d1.keys() if k.startswith('iloc')])
        attrs.extend([k for k in d1.keys() if k.startswith('jloc')])

    for k in attrs:
        if not d1[k] == d2[k]:
            raise ValueError("'{}' differ '{}' vs '{}'"
                            .format(k, d1[k], d2[k]))

def _diff_dicts(d1, d2):
    """ Return { key: (value1, value2) } where d1 and d2 differ """
    return { k: (v, d2[k]) for k, v in d1.items()
            if k in d2 and not v == d2[k]}

with obit_context():
    err = obit_err()
    uv1_name = "{}.{}.{}".format(args.name1, args.class1, args.seq1)
    uv2_name = "{}.{}.{}".format(args.name2, args.class2, args.seq2)

    log.info("Comparing '{}' with '{}'".format(uv1_name, uv2_name))

    uv1 = UV.newPAUV("uv1", args.name1, args.class1, args.disk1, args.seq1, True, err)
    handle_obit_err("Error opening {}".format(uv1_name), err)
    uv1.List.set('nVisPIO', args.nvispio)

    uv2 = UV.newPAUV("uv2", args.name2, args.class2, args.disk2, args.seq2, True, err)
    handle_obit_err("Error opening {}".format(uv2_name), err)
    uv2.List.set('nVisPIO', args.nvispio)

    # Mention differences in descriptors
    diff = _diff_dicts(uv1.Desc.Dict, uv2.Desc.Dict)
    if len(diff) > 0:
        log.info("UV Descriptors differ as follows\n%s" %
                                pformat(diff))

    uv1.Open(UV.READONLY, err)
    handle_obit_err("Error reading {}".format(uv1_name), err)
    uv2.Open(UV.READONLY, err)
    handle_obit_err("Error reading {}".format(uv2_name), err)

    nvis = uv1.Desc.Dict['nvis']

    _sanity_check_desc(uv1.Desc.Dict, uv2.Desc.Dict)

    # Get configured random parameters
    rand_parms = { k: v for k, v in uv1.Desc.Dict.items()
                        if k.startswith('iloc')
                        and not v == -1 }

    # Iterate over chunks of nvispio visibilities, comparing
    for vi in xrange(0, nvis, args.nvispio):
        # Set the number of visibilities and the offset we wish to read
        d1, d2 = uv1.Desc.Dict, uv2.Desc.Dict
        d1.update({'numVisBuff': args.nvispio, 'firstVis': vi})
        d2.update({'numVisBuff': args.nvispio, 'firstVis': vi})
        uv1.Desc.Dict, uv2.Desc.Dict = d1, d2

        # UV.Read wants 1-rel indexing.
        uv1.Read(err)
        handle_obit_err("Error reading {}".format(uv1_name), err)
        uv2.Read(err)
        handle_obit_err("Error reading {}".format(uv2_name), err)

        # Query the descriptors, post read
        d1, d2 = uv1.Desc.Dict, uv2.Desc.Dict
        _sanity_check_desc(d1, d2)

        nv = d1['numVisBuff']

        # Reference the buffers
        buf1 = np.frombuffer(uv1.VisBuf, count=-1, dtype=np.float32)
        buf2 = np.frombuffer(uv2.VisBuf, count=-1, dtype=np.float32)

        # Compare random parameters in each file
        for k, idx in rand_parms.iteritems():
            # Check for any configured comparison tolerances
            # for this random parameter in args.
            # 1e-5 is the default numpy rtol
            arg_key = '%s_rtol' % k
            rtol = getattr(args, arg_key, 1e-5)
            lrec = d1['lrec']

            # Extract random parameters for each visibility
            rparm1 = buf1[idx:idx+nv*lrec:lrec]
            rparm2 = buf2[idx:idx+nv*lrec:lrec]

            assert len(rparm1) == len(rparm2) == nv

            close = np.isclose(rparm1, rparm2, rtol=rtol)
            diff = np.invert(close)

            if np.count_nonzero(diff) > 0:
                diff, = np.nonzero(diff)
                raise ValueError("'%s' differed from '%s' in "
                          "random parameter '%s' with rtol '%s'\n"
                          "visibility numbers first %s last %s\n"
                          "buffer1 first %s last %s\n"
                          "buffer2 first %s last %s" %
                                    (uv1_name, uv2_name, k, rtol,
                                    vi + diff[:args.N], vi + diff[-args.N:],
                                    rparm1[:args.N], rparm1[-args.N:],
                                    rparm2[:args.N], rparm2[-args.N:]))


        nrparm = d1['nrparm']

        # Now compare each visibility chunk
        for idx in range(nrparm, nrparm + nv*lrec, lrec):
            vis1 = buf1[idx:idx+lrec-nrparm]
            vis2 = buf1[idx:idx+lrec-nrparm]

            assert len(vis1) == len(vis2) == (lrec-nrparm)

            close = np.isclose(vis1, vis2)
            diff = np.invert(close)

            if np.count_nonzero(diff) > 0:
                diff, = np.nonzero(diff)
                raise ValueError("'%s' differed from '%s' in "
                    "visibility '%s'\n"
                    "indices with visibility chunk first %s last %s\n"
                    "buffer1 first %s last %s\n"
                    "buffer2 first %s last %s" %
                        (uv1_name, uv2_name, vi,
                        diff[:args.N]), diff[-args.N:],
                        vis1[:args.N], vis1[-args.N:],
                        vis2[:args.N], vis2[-args.N:])

    uv1.Close(err)
    handle_obit_err("Error closing {}".format(uv1_name), err)
    uv2.Close(err)
    handle_obit_err("Error closing {}".format(uv2_name), err)
