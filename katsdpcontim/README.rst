Obit Docker build
=================

Docker files for building ``AIPS`` and ``Obit`` on Ubuntu ``xenial``.

~~~~~~~~~~~~
Requirements
~~~~~~~~~~~~

You'll need some docker infrastructure:

- Docker Ubuntu `installation instructions <https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/>`_.
- docker-compose. ``pip install docker-compose``.

~~~~~
Setup
~~~~~

Clone the following repositories:

.. code-block::

    # git clone git@github.com:ska-sa/katpoint.git
    # git clone git@github.com:ska-sa/katdal.git

Also, checkout the Obit source code @ revision ``570``.

.. code-block::

    # svn checkout -r 570 https://github.com/bill-cotton/Obit

~~~~~
Build
~~~~~

The following builds the docker containers.
Inspect the ``docker-compose.yml`` and ``Dockerfile's``
for further insight.

.. code-block::

    # docker-compose build xenial-obit-dev


~~~~~~~~~~~~~
Docker Mounts
~~~~~~~~~~~~~

To persist AIPS and FITS disks between container runs, the first AIPS disk
and the single FITS area are mounted on the docker host via the following
specification in ``docker-compose.yml``:

.. code-block:: yaml

    volumes:
      - $HOME/.local/katsdpcontim/aipsmounts/AIPS:/usr/local/AIPS/DATA/LOCALHOST_1:rw
      - $HOME/.local/katsdpcontim/aipsmounts/FITS:/usr/local/AIPS/FITS:rw

Two important points to note:

- These will be **mounted as the root** and **consume large quantities of disk space**.
- The mount points inside the container should match the configuration discussed in `AIPS Disk Setup`_.

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

Export katdal observation
~~~~~~~~~~~~~~~~~~~~~~~~~

The ``uv_export.py`` script exports a katdal observation to a UV data file on an AIPS disk.
For example:

.. code-block::

    # uv_export.py /var/kat/archive2/data/MeerKATAR1/telescope_products/2017/07/15/1500148809.h5

The AIPS filename is automatically derived from the input filename.
Five command line options can be specified to further customise the AIPS filenames.

--disk  AIPS disk number
--name  Name of the file. ``'1500148809'`` for example
--class  Short string indicating file class. Can be thought of as arbitrary tags
         assigned by the user.
         ``'raw'``  to indicate raw visibilities for example.
--seq  AIPS file sequence number.
       A number used to specify bits of data in a sequence. ``'1'`` for example.
--select  katdal select statement. Should only contain python
          assignment statements to python literals, separated
          by semi-colons. e.g. "scans='track';spw=0".

Compare export versus legacy export
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An observation export can be compared against the legacy export
code available in the older ``katim`` package:

.. code-block::

    # uv_export.py /var/kat/archive2/data/MeerKATAR1/telescope_products/2017/07/15/1500148809.h5
    # legacy_export.py /var/kat/archive2/data/MeerKATAR1/telescope_products/2017/07/15/1500148809.h5
    # cmp_uv.py -n1 1500148809 -c1 raw -n2 1500148809 -c2 legacy

This will iterate over the visibilities in each file comparing
one against the other and logging comparison failures.

**Note that the time random parameter is slightly different
in current vs legacy. This is because the starting time,
or midnight is computed from the** :code:`katdal.DataSet.start_time.sec`
**rather than** :code:`katdal.DataSet.timestamps[1:2]`.

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
      (Using global default file /usr/local/AIPS/DA00/DADEVS.LIST for DADEVS.PL)
       Disk 1 (1) is /usr/local/AIPS/DATA/LOCALHOST_1
       Disk 2 (2) is /usr/local/AIPS/DATA/LOCALHOST_2

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

Export CLEAN image with FITS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

The Dockerfile installs AIPS into ``/usr/local/AIPS``.
AIPS disks are usually present in the ``DATA`` sub-directory of the AIPS installation
and ``/usr/local/AIPS/DATA/LOCALHOST_1`` is the first AIPS disk by default.

However, AIPS disks can live in any subdirectory and can be configured
by editing:

- ``/usr/local/AIPS/DA00/DADEVS.LIST``
- ``/usr/local/AIPS/DA00/NETSP``

AIPS also has a separate FITS area in which *normal* FITS files are stored,
and ``/usr/local/AIPS/FITS`` is this area by default.


Obit Disks
~~~~~~~~~~

The Dockerfile installs Obit into ``/usr/local/Obit``.
Obit *fakes* AIPS disks and FITS areas by calls to :code:`OSystem.OSystem`.
It should also be noted that Obit requires files in the
``/usr/local/Obit/ObitSystem/Obit/share/data/`` directory to be present in a FITS area,
source catalogues being the most obvious example.

In order to run AIPS tasks on Obit output it is useful make these
disks/areas equivalent to those of the AIPS installation.
This is achieved by running the ``cfg_aips_disks.py`` script which:

- modifies ``DADEVS.LIST`` and ``NETSP`` in the AIPS installation.
- Creates soft links in the Obit data directory into the FITS area.

