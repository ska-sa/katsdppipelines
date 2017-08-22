Obit Docker build
=================

Docker files for building ``AIPS`` and ``Obit`` on Ubuntu ``trusty`` and ``xenial``. The ``xenial`` container should be preferred.

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

.. code-block:: bash

    $ git clone git@github.com:ska-sa/katpoint.git
    $ git clone git@github.com:ska-sa/katdal.git

Also, checkout the Obit source code @ revision ``570``.

.. code-block:: bash

    $ svn checkout -r 570 https://github.com/bill-cotton/Obit

~~~~~
Build
~~~~~

The following builds the docker containers.
Inspect the ``docker-compose.yml`` and ``Dockerfile's``
for further insight.

.. code-block:: bash

    $ docker-compose build xenial-obit-dev
    $ docker-compose build trusty-obit-dev

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


~~~~~~~~~~~~~
Docker Mounts
~~~~~~~~~~~~~

To persist AIPS and FITS disks between container runs, the first AIPS disk
and the single FITS area is mounted on the docker host:

- ``$HOME/.local/katsdpcontim/aipsmounts/AIPS:/usr/local/AIPS/DATA/LOCALHOST_1:rw``
- ``$HOME/.local/katsdpcontim/aipsmounts/FITS:/usr/local/AIPS/FITS:rw``

**Note that these will probably be mounted as the `root` owner
and consume large quantities of disk space**.

Its useful to mount the KAT archives and other volumes within these containers.
Edit the ``docker-compose.yml`` file to add the following, for example.

- ``/var/kat/archive2:/var/kat/archive2:ro``

~~~
Run
~~~

.. code-block:: bash

    $ docker-compose run --rm xenial-obit-dev
