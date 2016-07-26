#!groovy

def katsdp = fileLoader.fromGit('scripts/katsdp.groovy', 'git@github.com:ska-sa/katsdpjenkins', 'master', 'katpull', '')

if (!katsdp.isTegra()) {
    katsdp.setDependencies(['ska-sa/katsdpdockerbase/master', 'ska-sa/katpoint/master', 'ska-sa/katsdpsigproc/master'])
    katsdp.standardBuild(maintainer: 'laura@ska.ac.za', subdir: 'katsdpcal')
}
else {
    katsdp.setDependencies(['tegra_ska-sa/katsdpdockerbase/master'])
}
katsdp.standardBuild(
    maintainer: 'bmerry@ska.ac.za',
    subdir: 'katsdpimager',
    cuda: true,
    prepare_timeout: [time: 90, unit: 'MINUTES'],
    test_timeout: [time: 90, unit: 'MINUTES'])
