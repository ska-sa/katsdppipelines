#!groovy

def katsdp = fileLoader.fromGit('scripts/katsdp.groovy', 'git@github.com:ska-sa/katsdpjenkins', 'master', 'katpull', '')

katsdp.setDependencies(['ska-sa/katsdpdockerbase/master'])

katsdp.standardBuild(maintainer: 'laura@ska.ac.za', subdir: 'katsdpcal')
katsdp.standardBuild(maintainer: 'bmerry@ska.ac.za', subdir: 'katsdpimager')
