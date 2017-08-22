"""
Mainly exists to compare export from the older ``h5touvfits.py``
to the newer ``uv_export.py``. For this to work,
``h5touvits.py`` should be modified to

1. run within an obit_context()
2. not remove the temporary AIPS directories it creates

Can probably be removed at some point.
"""

import argparse
from pprint import pprint

import numpy as np

from katsdpcontim import KatdalAdapter, UVFacade, handle_obit_err, obit_context, obit_err
import UV

def create_parser():
    parser = argparse.ArgumentParser(description="Good for comparing "
                                                 "files exported by 'h5touvfits.py' "
                                                 "and 'uv_export.py")
    parser.add_argument("-n1", "--name1")
    parser.add_argument("-c1", "--class1")
    parser.add_argument("-d1", "--disk1", type=int, default=1)
    parser.add_argument("-s1", "--seq1", type=int, default=1)

    parser.add_argument("-n2", "--name2")
    parser.add_argument("-c2", "--class2")
    parser.add_argument("-d2", "--disk2", type=int, default=1)
    parser.add_argument("-s2", "--seq2", type=int, default=1)

    parser.add_argument("--nvispio", type=int, default=1024)
    parser.add_argument("-tj", "--time-jitter", type=float)

    return parser

args = create_parser().parse_args()

with obit_context():
    err = obit_err()

    uv1_name = "{}.{}.{}".format(args.name1, args.class1, args.seq1)
    uv2_name = "{}.{}.{}".format(args.name2, args.class2, args.seq2)

    uv1 = UV.newPAUV("uv1", args.name1, args.class1, args.disk1, args.seq1, True, err)
    handle_obit_err("Error opening {}".format(uv1_name), err)
    uv1.List.set('nVisPIO', args.nvispio)

    uv2 = UV.newPAUV("uv2", args.name2, args.class2, args.disk2, args.seq2, True, err)
    handle_obit_err("Error opening {}".format(uv2_name), err)
    uv2.List.set('nVisPIO', args.nvispio)

    def _diff_dicts(d1, d2):
        """ Return { key: (value1, value2) } where d1 and d2 differ """
        return { k: (v, d2[k]) for k, v in d1.items()
                if k in d2 and not v == d2[k]}

    pprint(_diff_dicts(uv1.Desc.Dict, uv2.Desc.Dict))

    uv1.Open(UV.READONLY, err)
    handle_obit_err("Error reading {}".format(uv1_name), err)
    uv2.Open(UV.READONLY, err)
    handle_obit_err("Error reading {}".format(uv2_name), err)

    nvis = uv1.Desc.Dict['nvis']

    def _sanity_check(d1, d2):
        """ Sanity check descriptors """
        attrs = ['nvis', 'numVisBuff', 'firstVis', 'lrec']
        for k in attrs:
            if not d1[k] == d2[k]:
                raise ValueError("'{}' differ '{}' vs '{}'"
                                .format(k, d1[k], d2[k]))

        pprint(['Looking good', {k: d1[k] for k in attrs}])

    _sanity_check(uv1.Desc.Dict, uv2.Desc.Dict)

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
        _sanity_check(d1, d2)

        # Reference the buffers
        buf1 = np.frombuffer(uv1.VisBuf, count=-1, dtype=np.float32)
        buf2 = np.frombuffer(uv2.VisBuf, count=-1, dtype=np.float32)

        # Are they close?
        close = np.isclose(buf1, buf2)

        if np.all(close):
            print "All close", vi, vi+1
        else:
            d = np.invert(close)
            nz, = np.nonzero(d)
            print nz
            print buf1[d][0:20]
            print buf2[d][0:20]

            def _log_rand_parms(buf, desc):
                u, v, w, t, b, s = (buf[desc[k]] for k in ('ilocu', 'ilocv', 'ilocw',
                                                            'iloct', 'ilocb', 'ilocsu'))

                return ("u: {: 2.6f} v: {: 2.6f} w: {: 2.6f} "
                        "t: {: 2.6f} bl: {: 2.6f} s: {:1.0f}"
                            .format(u,v,w,t,b,s))

            def _log_some_vis(buf, desc, order='C'):
                i = desc['nrparm']
                lrec = desc['lrec']
                inaxes = tuple(desc['inaxes'][:6])
                vislen = np.product(inaxes)
                return buf[i:i+vislen]
                buf = buf[i:i+vislen].reshape(inaxes, order=order)
                return buf

            print _log_rand_parms(buf1[nz[0]:], d1)
            print _log_rand_parms(buf2[nz[0]:], d2)
            print _log_some_vis(buf1[nz[0]:], d1)
            print _log_some_vis(buf2[nz[0]:], d2)

            raise ValueError("Difference on {}".format(firstVis))

    uv1.Close(err)
    handle_obit_err("Error closing {}".format(uv1_name), err)
    uv2.Close(err)
    handle_obit_err("Error closing {}".format(uv2_name), err)
