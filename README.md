# Ferret
The Ferret program from NOAA/PMEL.  
See [https://ferret.pmel.noaa.gov/Ferret/](https://ferret.pmel.noaa.gov/Ferret/)
for more information about Ferret and PyFerret.

This repository is regularly synchronized with Ferret repository at PMEL
(the trunk of the ferret project in the subversion repository at PMEL)
using git-svn.

#### Legal Disclaimer
*This repository is a software product and is not official communication
of the National Oceanic and Atmospheric Administration (NOAA), or the
United States Department of Commerce (DOC).  All NOAA GitHub project
code is provided on an 'as is' basis and the user assumes responsibility
for its use.  Any claims against the DOC or DOC bureaus stemming from
the use of this GitHub project will be governed by all applicable Federal
law.  Any reference to specific commercial products, processes, or services
by service mark, trademark, manufacturer, or otherwise, does not constitute
or imply their endorsement, recommendation, or favoring by the DOC.
The DOC seal and logo, or the seal and logo of a DOC bureau, shall not
be used in any manner to imply endorsement of any commercial product
or activity by the DOC or the United States Government.*

## Ferret Documentation

For more information on using Ferret, see the Ferret documentation under
[https://ferret.pmel.noaa.gov/Ferret/](https://ferret.pmel.noaa.gov/Ferret/)

Information about the Ferret email users group, and archives of past discussions
from the group (which should be searched prior to sending a question to the email
users group) can be found at
[https://ferret.pmel.noaa.gov/Ferret/email-users-group](https://ferret.pmel.noaa.gov/Ferret/email-users-group)

## If you build Ferret from these source files:

1. We highly recommend you build and use PyFerret (see:
[https://github.com/NOAA-PMEL/PyFerret](https://github.com/NOAA-PMEL/PyFerret))
instead of Ferret.
PyFerret provides publication-quality graphics and much more funtionality while still maintaining
the Ferret command prompt and compatibility with existing Ferret scripts.
PyFerret uses Qt for displayed graphics, removing the need for X-Windows software libraries and 
header files for Ferret builds.  PyFerret maintains this Ferret repository as an upstream
source, so updates to Ferret will also be found in the "Ferret engine" within PyFerret.

2. If you still wish to build Ferret, more information on building Ferret can be found in the 
[README_build_ferret](https://github.com/NOAA-PMEL/Ferret/blob/master/README_build_ferret) 
and the 
[README_ferret_mac_homebrew.md](https://github.com/NOAA-PMEL/Ferret/blob/master/README_ferret_mac_homebrew.md)
files in this repository.
This second file can also provide some useful information about building Ferret on Linux and Unix-type systems. 

3. The `site_specific.mk.in` and `external_functions/ef_utilites/site_specific.mk.in`
files in the repository must be copied to files without the `.in` extensions, and
the contents of these `site_specific.mk` files edited for your system configuration.
The `site_specific.mk` files will be ignored by git (the name was added to
`.gitignore`) so your customized configuration files will not be added to your
repository if you have cloned this repository.

4. The definitions of CC, FC, and LD (the last only for building external
functions written in Fortran) were moved to the `site_specific.mk.in` files.
(These were previously defined in the `platform_specific.mk.*` files.)
If you already have customized `site_specific.mk` files, please appropriately
updated your customized files with these additional definitions.

