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

Clone this repository:

.. code-block:: bash

    $ git clone --recursive git@giturl.com/katsdpcontim.git

The ``--recursive`` option above should initialise and update the various ``kat`` submodules.
If not, it may be necessary to give them a bit of a push.

.. code-block:: bash

   $ git submodule update --init

Also, checkout the Obit source code @ revision ``567``.

.. code-block:: bash

    $ svn checkout -r 567 https://github.com/bill-cotton/Obit

Build
~~~~~

.. code-block:: bash

    $ docker-compose build xenial-obit-dev
    $ docker-compose build trusty-obit-dev


Run
~~~

.. code-block:: bash

    $ docker-compose run --rm xenial-obit-dev
