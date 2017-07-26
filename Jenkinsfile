#!groovy

@Library('katsdpjenkins') _

def maintainer = 'bmerry@ska.ac.za'
if (!katsdp.isTegra()) {
    katsdp.setDependencies(['ska-sa/katsdpdockerbase/master', 'ska-sa/katpoint/master', 'ska-sa/katsdpsigproc/master'])
    maintainer = "$maintainer tmauch@ska.ac.za"
    katsdp.standardBuild(subdir: 'katsdpcal', docker_venv: true)
}
else {
    katsdp.setDependencies(['tegra_ska-sa/katsdpdockerbase/master'])
}
katsdp.standardBuild(
    subdir: 'katsdpimager',
    cuda: true,
    python3: true,
    prepare_timeout: [time: 90, unit: 'MINUTES'],
    test_timeout: [time: 90, unit: 'MINUTES'])
katsdp.mail(maintainer)
