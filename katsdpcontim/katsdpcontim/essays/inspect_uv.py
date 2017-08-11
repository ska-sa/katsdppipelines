import argparse

import katsdpcontim
from katsdpcontim import KatdalAdapter, UVFacade, handle_obit_err, obit_context, obit_err

import numpy as np

import UV

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("klass")
    parser.add_argument("seq", type=int)
    return parser

args = create_parser().parse_args()

with obit_context():
    err = obit_err()

    # Create the AIPS UV file
    disk = 1
    uv = UV.newPAUV("myuv", args.name, args.klass, disk, args.seq, True, err)
    handle_obit_err(err)

    uv.Open(UV.READONLY, err)

    # Configure number of visibilities written in a batch
    desc = uv.Desc.Dict

    # Number of random parameters
    nrparm = desc['nrparm']

    # Random parameter indices
    ilocu = desc['ilocu']     # U
    ilocv = desc['ilocv']     # V
    ilocw = desc['ilocw']     # W
    iloct = desc['iloct']     # time
    ilocb = desc['ilocb']     # baseline id
    ilocsu = desc['ilocsu']   # source id

    inaxes = tuple(desc['inaxes'][:6])  # Visibility shape, strip out trailing 0
    flat_inaxes = np.product(inaxes)
    lrec = nrparm + flat_inaxes         # Length of record in vis buffer

    desc['numVisBuff'] = 10
    uv.Desc.Dict = desc

    uv.Read(err, firstVis=1)
    handle_obit_err(err)

    vis_buffer = np.frombuffer(uv.VisBuf, count=-1, dtype=np.float32)
    idx = np.array([ilocu, ilocv, ilocw, iloct, ilocb, ilocsu])

    from pprint import pprint
    pprint(desc)

    print idx
    print vis_buffer[idx]
    print idx + lrec
    print vis_buffer[idx + lrec]
    print idx + 2*lrec
    print vis_buffer[idx + 2*lrec]


    uv.Close(err)
    handle_obit_err(err)
