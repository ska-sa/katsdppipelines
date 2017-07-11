#!/bin/bash
#Handle mounts (specify AIPSMOUNT) using -e when running
if [ ! -d ${AIPSMOUNT} ]; then
	echo "You must specify AIPSMOUNT environment variable to valid mounted volume to load / store aips volumes"
	exit 1
fi
echo "Remember to mount a valid volume with uvfits files as '/usr/local/AIPS/FITS' if you want uvfits reading support"
echo "The following UVFITS are available for import:"
ls /usr/local/AIPS/FITS
echo "Creating an AIPS mount in ${AIPSMOUNT}/LOCALHOST_2 if it doesn't exist already"
echo "-  ${AIPSMOUNT}/LOCALHOST_2" >> /usr/local/AIPS/DA00/DADEVS.LIST
echo "${AIPSMOUNT}/LOCALHOST_2 365.0    0    0    0    0    0    0    0    0" >> /usr/local/AIPS/DA00/NETSP
if [ ! -d ${AIPSMOUNT}/LOCALHOST_2 ]; then
	mkdir ${AIPSMOUNT}/LOCALHOST_2 && touch ${AIPSMOUNT}/LOCALHOST_2/SPACE
fi

#. /usr/local/AIPS/LOGIN.SH && aips da=all tv=local
. /usr/local/AIPS/LOGIN.SH

