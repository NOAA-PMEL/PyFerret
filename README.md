# Ferret
The Ferret program from NOAA/PMEL. 
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
[http://ferret.pmel.noaa.gov/Ferret/](http://ferret.pmel.noaa.gov/Ferret/)

Information about the Ferret email users group, and archives of past discussions
from the group (which should be searched prior to sending a question to the email 
users group) can be found at 
[http://ferret.pmel.noaa.gov/Ferret/email-users-group](http://ferret.pmel.noaa.gov/Ferret/email-users-group)

## If you build Ferret from these source files, please note:

The `site_specific.mk` and `external_functions/ef_utilites/site_specific.mk` 
files in the repository have been renamed with a `.in` appended to the name. 
You must copy these files with the `.in` extensions to create files with the 
`site_specific.mk` name and edit the contents to configure these for your 
system.  The `site_specific.mk` files will be ignored by git (the name was 
added to `.gitignore`) so your customized configuration files will not be 
added to your repository if you have cloned this repository. 

#### The `site_specific.mk.in` configuration template file changed April, 2018.

Library directories are now specified instead of installation directories
for NetCDF, HDF5, ReadLine, and SZ, with the makefile environment names
changing from `..._DIR` to `..._LIBDIR`.
The `include` directories, if needed, are assumed to be sibling directories
to these library directories.
You may wish to create a new `site_specific.mk` configuration file 
from this updated configuration template file.

