#!groovy

@Library('katsdpjenkins') _
katsdp.killOldJobs()


def maintainer = 'bmerry@ska.ac.za tmauch@ska.ac.za kmcalpine@ska.ac.za'
if (!katsdp.isTegra()) {
    katsdp.setDependencies([
        'ska-sa/katsdpdockerbase/master',
        'ska-sa/katpoint/master',
        'ska-sa/katdal/master',
        'ska-sa/katsdpsigproc/master',
        'ska-sa/katsdpservices/master',
        'ska-sa/katsdptelstate/master'])
    maintainer = "$maintainer ruby@ska.ac.za"
    katsdp.standardBuild(subdir: 'katsdpcal', python3: true, python2: false, docker_venv: true)
    katsdp.standardBuild(subdir: 'katsdpcontim',
                         cuda: true,
                         label: 'cpu-avx2')
}
else {
    katsdp.setDependencies(['tegra_ska-sa/katsdpdockerbase/master'])
}
katsdp.standardBuild(
    subdir: 'katsdpimager',
    cuda: true,
    python3: true,
    python2: false,
    prepare_timeout: [time: 90, unit: 'MINUTES'],
    test_timeout: [time: 90, unit: 'MINUTES'])
katsdp.mail(maintainer)
