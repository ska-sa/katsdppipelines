#!groovy

def katsdp = fileLoader.fromGit('scripts/katsdp.groovy', 'git@github.com:ska-sa/katsdpjenkins', 'master', 'katpull', '')

parallel katsdpcal: { katsdp.standardBuild(maintainer: 'lrichter@ska.ac.za', subdir: 'katsdpcal') },
         katsdpimager: { katsdp.standardBuild(maintainer: 'bmerry@ska.ac.za', subdir: 'katsdpimager') }
