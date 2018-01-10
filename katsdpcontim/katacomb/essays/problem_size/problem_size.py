import datetime as dt

import numpy as np
from hypercube import HyperCube

obs_length = dt.timedelta(hours=10)
dumps = dt.timedelta(seconds=2)

times = obs_length.seconds / dumps.seconds
antennae = 64
baselines = antennae*(antennae-1)//2
channels = 4096


cube = HyperCube()
cube.register_dimension('N', 7000, description="Image Width/Height")
cube.register_dimension('times', times, description="Timesteps")
cube.register_dimension('antenna', antennae, description="Antenna")
cube.register_dimension('baselines', baselines, description="Baselines")
cube.register_dimension('bands', 16, description="Sub-bands")
cube.register_dimension('facets', 100, description="Facets")
cube.register_dimension('channels', channels, description="Channels")
cube.register_dimension('corrs', 4, description="Visibility Correlations")
cube.register_dimension('img-pols', 1, description="Imaging Polarisations")
cube.register_dimension('gain-pols', 1, description="Gain Solution Polarisations")
cube.register_dimension('nthreads', 8, description="Number of Threads")

# Guessing times reduce by factor of 10 because of large inner core
# which can be greatly averaged
cube.register_dimension('avgtimes', times//10, description="Averaged Times")
# Thumbsuck. Improve.
cube.register_dimension('avgchannels', channels//4, description="Averaged Channels")

cube.register_array('pre-averaged visibilities',
                     ('times', 'baselines', 'channels', 'corrs'),
                     dtype=np.complex64)
cube.register_array('averaged visibilities',
                    ('avgtimes', 'baselines', 'avgchannels', 'corrs'),
                    dtype=np.complex64)
cube.register_array('grids',
                    ('nthreads', 'N', 'N', 'bands'),
                    dtype=np.complex64)
cube.register_array('final image',
                    ('N', 'N', 'bands', 'img-pols'),
                    dtype=np.complex64)
cube.register_array('gains',
                    ('times', 'antenna', 'bands', 'gain-pols'),
                    dtype=np.complex64)

print cube