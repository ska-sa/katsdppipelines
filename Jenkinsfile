#!groovy

@Library('katsdpjenkins') _
katsdp.killOldJobs()

katsdp.setDependencies([
    'ska-sa/katsdpdockerbase/master',
    'ska-sa/katpoint/master',
    'ska-sa/katdal/master',
    'ska-sa/katsdpsigproc/master',
    'ska-sa/katsdpservices/master',
    'ska-sa/katsdptelstate/master'])
katsdp.standardBuild(subdir: 'katsdpcal', python3: true, python2: false, docker_venv: true)
katsdp.standardBuild(subdir: 'katsdpcontim',
                     cuda: true,
                     label: 'cpu-avx2',
                     docker_timeout: [time: 90, unit: 'MINUTES'])
katsdp.mail('sdpdev+katsdpcal@ska.ac.za sdpdev+katsdpcontim@ska.ac.za')
