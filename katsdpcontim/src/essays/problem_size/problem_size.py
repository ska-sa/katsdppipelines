import datetime as dt

import numpy as np
from hypercube import HyperCube

obs_length = dt.timedelta(hours=10)
dumps = dt.timedelta(seconds=2)

times = obs_length.seconds / dumps.seconds

cube = HyperCube()
cube.register_dimension('N', 7000, description="Image Width/Height")
cube.register_dimension('times', times, description="Timesteps")
cube.register_dimension('antenna', 64, description="Antenna")
cube.register_dimension('baselines', 64*(64-1)//2, description="Baselines")
cube.register_dimension('bands', 16, description="Sub-bands")
cube.register_dimension('channels', 4096, description="Channels")
cube.register_dimension('corrs', 4, description="Visibility Correlations")
cube.register_dimension('img-pols', 1, description="Imaging Polarisations")
cube.register_dimension('gain-pols', 1, description="Gain Solution Polarisations")

cube.register_array('visibilities', ('times', 'baselines', 'channels', 'corrs'),
                                    dtype=np.complex64)
cube.register_array('image', ('N', 'N', 'bands', 'img-pols'),
                                    dtype=np.complex64)
cube.register_array('gains', ('times', 'antenna', 'bands', 'gain-pols'),
                                    dtype=np.complex64)

print cube