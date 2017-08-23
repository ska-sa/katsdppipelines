=========================================
Obit Continuum Imaging Pipeline TODO List
=========================================

- Pull calibration solutions from the CL table
- Output per scan AIPS files.
- Pin software versions
    - Obit ``r570``
    - AIPS ``31DEC16``
    - katpoint ``master``
    - katdal ``master``
- Improve visibility massaging process. Presently, as a general process,
  visibility data is massaged into a
  :code:`(ntime,nbl,3,nstokes,nchan,nspw,1,1)` array.
  Ignoring the time and baseline dimensions, each :code:`(3,nstokes,nchan,nspw,1,1)` chunk
  (matching the FITS :code:`inaxes` shape)  is ravelled to FORTRAN order when writing to
  the visibility buffer.
  This is inefficient since memory is reordered twice.
  It should be possible to massage the visibility data into
  shape :code:`(ntime,nbl,1,1,nspw,nchan,nstokes,3)` and ravel to achieve the same effect.

