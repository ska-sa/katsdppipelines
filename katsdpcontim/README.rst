Obit Docker build
=================

Docker files for building ``AIPS`` and ``Obit`` on Ubuntu ``xenial``.
Both production (Dockerfile) and development (Dockerfile.dev) containers
are available.
The production container contains Obit installed under the
kat user's directory and runs python code and Obit executables
as the ``kat`` user.

The development container extends the production container by
installing AIPS under the kat user's directory,
in addition to a VNC server. In contrast to the production
container, it runs code as ``root``. This allows developers
to (easily) mount read/write docker volumes.

~~~~~~~~~~~~
Requirements
~~~~~~~~~~~~

You'll need some docker infrastructure:

- The ``docker-base`` docker image available in the
  `SDP docker registry <https://github.com/ska-sa/katsdpinfrastructure/tree/master/registry#client-setup>`_. ``docker-base-gpu`` may also be useful to
  build Obit with GPU acceleration.
- Docker Ubuntu `installation instructions <https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/>`_.
- docker-compose. ``pip install docker-compose``.

~~~~~
Build
~~~~~

The following builds the production and development docker containers.
Inspect the ``docker-compose.yml`` and ``Dockerfile's``
for further insight.

.. code-block::

    # docker-compose build xenial-obit
    # docker-compose build xenial-obit-dev


~~~~~~~~~~~~~
Docker Mounts
~~~~~~~~~~~~~

To persist AIPS and FITS disks between container runs, the first AIPS disk
and the single FITS area are mounted on the docker host via the following
specification in ``docker-compose.yml``:

.. code-block:: yaml

    volumes:
      - $HOME/.local/katacomb/aipsmounts/AIPS:/home/kat/AIPS/DATA/LOCALHOST_1:rw
      - $HOME/.local/katacomb/aipsmounts/FITS:/home/kat/AIPS/FITS:rw

Important points to note:

- Production container mounts will need to have the same permission as the kat user.
- The development container runs as root, so developers can aggressively
  mount writeable volumes, with all the security implications thereof.
- AIPS/Obit observation files **consume large quantities of disk space**.
- The mount points inside the container should match the configuration
  discussed in `AIPS Disk Setup`_.

It's useful to mount the KAT archives and other volumes within these containers.
Edit ``docker-compose.yml`` to mount KAT NFS ``archive2`` within the container,
for example.

.. code-block:: yaml

    volumes:
      ...
      - /var/kat/archive2:/var/kat/archive2:ro



~~~
Run
~~~

.. code-block::

    # docker-compose run --rm xenial-obit-dev

Access the VNC Server
~~~~~~~~~~~~~~~~~~~~~

You may find it useful to run ``ObitView`` and AIPS TV utilities from inside the container.
A VNC server is started inside the container and exposed on port 5901 on the ``loopback``
localhost interface of the docker host. This means that you'll need to forward the above port
when SSHing into the docker host in order to access the VNC server. This can be done as follows
in your ``.ssh/config`` file.

.. code-block::

    Host com4
        Hostname dockerhost
        ForwardX11 yes
        LocalForward 5901 localhost:5901

Then, on your local machine, you should direct your VNC client to ``localhost:5901`` to direct
your VNC traffic through the tunnel to the server inside the container.


Run Continuum Imaging Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``continuum_pipeline.py`` script runs the Continuum Pipeline.

.. code-block::

    $ continuum_pipeline.py --help
    $ continuum_pipeline.py /var/kat/archive2/data/MeerKATAR1/telescope_products/2017/07/15/1500148809.h5

Any resulting images will be stored in AIPS disks described in
`AIPS Disk Setup`_.

Inspecting , Viewing and Exporting data in ObitTalk
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Obit has a number of python utilities for inspecting, viewing and exporting
AIPS data.

Start ``ObitView`` in an X11 terminal to start the Obit Image viewer.

.. code-block::

    $ ObitView

Then, in another terminal start ``ObitTalk``, which interacts with
AIPS disk data.

.. code-block::

    $ ObitTalk

``ObitTalk`` starts a python terminal that can inspect AIPS disk data
and interact with ``ObitView``.

AIPS disk data can briefly be subdivided into UV and Image data.
This data can be viewed with the ``AUcat`` and ``AMcat``
commands.

For example, the following code,

- lists all the AIPS images on disk 1
- Creates a python object associated with the Mimosa image
- Writes the image as ``IMAGE.FITS`` on FITS disk 1

.. code-block:: python

    >>> AMcat(1)
    AIPS Directory listing for disk 1
      2 Merope      .IClean.    1 MA 08-Feb-2018 11:55:52
      5 Kaus Austral.IClean.    1 MA 08-Feb-2018 11:55:52
      7 Mimosa      .IClean.    1 MA 08-Feb-2018 11:55:52
      9 Rukbat      .IClean.    1 MA 08-Feb-2018 11:56:19
     11 Sirrah      .IClean.    1 MA 08-Feb-2018 11:55:52

    >>> x = getname(7)

    >>> err = OErr.OErr()
    >>> imtab(x, "IMAGE.FITS", 1, err)
    <C Image instance> FITS Image DATA>

Then, it is also possible to display the image in ``ObitView``
through the ``tvlod`` command.

.. code-block:: python

    >>> tvlod(x)

AIPS UV data can be inspected via the ``imhead`` command:

.. code-block:: python

    >>> AUcat(1)
    AIPS Directory listing for disk 1
      1 Merope      .MFImag.    1 UV 08-Feb-2018 11:55:52
      3 mock        .merge .    1 UV 08-Feb-2018 11:55:53
      4 Kaus Austral.MFImag.    1 UV 08-Feb-2018 11:55:52
      6 Mimosa      .MFImag.    1 UV 08-Feb-2018 11:55:52
      8 Rukbat      .MFImag.    1 UV 08-Feb-2018 11:55:52
     10 Sirrah      .MFImag.    1 UV 08-Feb-2018 11:55:52
     13 mock        .merge .    2 UV 08-Feb-2018 11:55:53


    >>> imhead(getname(3))
    AIPS UV mock         merge  1 1
    AIPS UV Data Name: mock         Class: merge  seq:        1 disk:    1
    Object: MULTI
    Observed: 2018-02-08 Telescope:  MeerKAT  Created: 2018-02-08
    Observer: ghost      Instrument: MeerKAT
     # visibilities       1430  Sort order = TB
    Rand axes: UU-L-SIN VV-L-SIN WW-L-SIN BASELINE TIME1
               SOURCE   INTTIM
    --------------------------------------------------------------
    Type    Pixels   Coord value     at Pixel     Coord incr   Rotat
    COMPLEX      3               1       1.00              1    0.00
    STOKES       2      XPol             1.00             -1    0.00
    FREQ         2      1.0432e+09       1.00       4.28e+08    0.00
    IF           1               1       1.00              1    0.00
    RA           1   0  0  0.00000       1.00              0    0.00
    DEC          1 -00  0  0.0000        1.00              0    0.00
    --------------------------------------------------------------
    Coordinate equinox 2000.0  Coordinate epoch 2000.00
    Observed RA    0  0  0.00000 Observed Dec -00  0  0.0000
    Rest freq            0 Vel type: Observer,  wrt  Optical
    Alt ref value            0  wrt pixel     0.88
    Maximum version number of AIPS FQ tables is 1
    Maximum version number of AIPS SU tables is 1
    Maximum version number of AIPS PS tables is 1
    Maximum version number of AIPS AN tables is 1
    Maximum version number of AIPS CL tables is 1
    Maximum version number of AIPS NX tables is 1



Export katdal observation
~~~~~~~~~~~~~~~~~~~~~~~~~

The ``uv_export.py`` script exports a katdal observation to a UV data file on an AIPS disk.

.. code-block::

    $ uv_export.py --help
    $ uv_export.py /var/kat/archive2/data/MeerKATAR1/telescope_products/2017/07/15/1500148809.h5


Run AIPS
~~~~~~~~

Run AIPS to view the observation. Remember to enter ``105`` when asked
to enter your user number. You should see something like the following:

.. code-block::

    # aips da=all notv tvok tpok
    START_AIPS: Your initial AIPS printer is the
    START_AIPS:  - system name , AIPS type

    START_AIPS: User data area assignments:
    DADEVS.PL: This program is untested under Perl version 5.022
      (Using global default file /home/kat/AIPS/DA00/DADEVS.LIST for DADEVS.PL)
       Disk 1 (1) is /home/kat/AIPS/DATA/LOCALHOST_1
       Disk 2 (2) is /home/kat/AIPS/DATA/LOCALHOST_2

    Tape assignments:
       Tape 1 is REMOTE
       Tape 2 is REMOTE

    START_AIPS: Assuming TV servers are already started (you said TVOK)
    START_AIPS: Assuming TPMON daemons are running or not used (you said TPOK)
    Starting up 31DEC16 AIPS with normal priority
    Begin the one true AIPS number 1 (release of 31DEC16) at priority =   0
    AIPS 1: You are NOT assigned a TV device or server
    AIPS 1: You are NOT assigned a graphics device or server
    AIPS 1: Enter user ID number
    ?105
    AIPS 1:                          31DEC16 AIPS:
    AIPS 1:      Copyright (C) 1995-2017 Associated Universities, Inc.
    AIPS 1:            AIPS comes with ABSOLUTELY NO WARRANTY;
    AIPS 1:                 for details, type HELP GNUGPL
    AIPS 1: This is free software, and you are welcome to redistribute it
    AIPS 1: under certain conditions; type EXPLAIN GNUGPL for details.
    AIPS 1: Previous session command-line history recovered.
    AIPS 1: TAB-key completions enabled, type HELP READLINE for details.
    AIPS 1: Recovered POPS environment from last exit
    >

Then, type ``UCAT`` to view and ``MCAT`` to list UV data and images
on the AIPS disks, respectively:

.. code-block::

    >UCAT
    AIPS 1: Catalog on disk  1
    AIPS 1:   Cat  Usid Mapname      Class   Seq  Pt    Last access     Stat
    AIPS 1:     1   105 1500148809  .raw   .    1 UV 22-AUG-17 16:58:43
    AIPS 1: Catalog on disk  2
    AIPS 1:   Cat  Usid Mapname      Class   Seq  Pt    Last access     Stat
    >

Then, exit AIPS

.. code-block::

    > EXIT


Image observation with MFImage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once an observation has been exported to a UV data file on an AIPS disk, we can run ``MFImage``
to image the observation. A number of standard configuration files for this in ``/obitconf``.
Edit ``mfimage_nosc.in`` to specify the AIPS file parameters for the observation above
and the run MFImage using the configuration file.

.. code-block::

    /obitconf $ MFImage -input mfimage_nosc.in &
    /obitconf $ tail -f IMAGE.log

Export AIPS CLEAN image to FITS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run AIPS and look for the CLEAN image with the ``MCAT`` command.
Then, run the ``FITTP`` task to export the CLEAN image from the
AIPS disk to the FITS disk.

~~~~~~~~~~~~~~~
AIPS Disk Setup
~~~~~~~~~~~~~~~

AIPS has its own concept of a filesystem: an AIPS `disk`.
It can simply be regarded as a standard unix subdirectory
containing visibility, table and image files following
an AIPS naming and indexing scheme.
Multiple AIPS disks can be present on the system.

Obit does not require an AIPS installation to run,
*faking* AIPS disks and FITS areas, but to run AIPS tasks
on Obit data, it is useful for these to be equivalent.

Furthermore, it is useful to mount AIPS disks as
subdirectories on the docker host so that data
persists between container runs.

For this functionality to be available, the disk setup
for all three pieces of software should be similarly configured.
**The ultimate authority for AIPS disk configuration is the
lies within the katsdpcontim configuration and the docker mounts
in "docker-compose.yml" should also be based on this configuration**.

AIPS Disks
~~~~~~~~~~

The Dockerfile installs AIPS into ``/home/kat/AIPS``.
AIPS disks are usually present in the ``DATA`` sub-directory of the AIPS installation
and ``/home/kat/AIPS/DATA/LOCALHOST_1`` is the first AIPS disk by default.

However, AIPS disks can live in any subdirectory and can be configured
by editing:

- ``/home/kat/AIPS/DA00/DADEVS.LIST``
- ``/home/kat/AIPS/DA00/NETSP``

AIPS also has a separate FITS area in which *normal* FITS files are stored,
and ``/home/kat/AIPS/FITS`` is this area by default.


Obit Disks
~~~~~~~~~~

The Dockerfile installs Obit into ``/home/kat/Obit``.
Obit *fakes* AIPS disks and FITS areas by calls to :code:`OSystem.OSystem`.
It should also be noted that Obit requires files in the
``/home/kat/Obit/ObitSystem/Obit/share/data/`` directory to be present in a FITS area,
source catalogues being the most obvious example.

In order to run AIPS tasks on Obit output it is useful make these
disks/areas equivalent to those of the AIPS installation.
This is achieved by running the ``cfg_aips_disks.py`` script which:

- modifies ``DADEVS.LIST`` and ``NETSP`` in the AIPS installation.
- Creates soft links in the Obit data directory into the FITS area.


~~~~~~~
Testing
~~~~~~~

A test suite exists, but must be executed inside the container:

.. code-block::

  $ nosetests /home/kat/src/katacomb
