Obit Docker build
=================

Docker files for building ``AIPS`` and ``Obit`` on Ubuntu ``trusty`` and ``xenial``. The ``xenial`` container should be preferred.

Requirements
~~~~~~~~~~~~

You'll need some docker infrastructure:

- Docker Ubuntu `installation instructions <https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/>`_.
- docker-compose. ``pip install docker-compose``.

Setup
~~~~~

Clone the following repositories:

.. code-block:: bash

    $ git clone git@github.com:ska-sa/katpoint.git
    $ git clone git@github.com:ska-sa/katdal.git

Also, checkout the Obit source code @ revision ``570``.

.. code-block:: bash

    $ svn checkout -r 570 https://github.com/bill-cotton/Obit

Build
~~~~~

.. code-block:: bash

    $ docker-compose build xenial-obit-dev
    $ docker-compose build trusty-obit-dev

Mounts
~~~~~~

docker-compose will mount the following first AIPS and FITS disks
inside the container in these local directories:

- `$HOME/.local/katsdpcontim/aipsmounts/AIPS`
- `$HOME/.local/katsdpcontim/aipsmounts/FITS`

**Note that these will probably be mounted as the `root` owner
and consumes large quantities of disk space!**.

Run
~~~

.. code-block:: bash

    $ docker-compose run --rm xenial-obit-dev
