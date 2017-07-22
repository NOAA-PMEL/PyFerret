# Ferret
The Ferret program from NOAA/PMEL. 
This repository is regularly synchronized with Ferret repository at PMEL 
(the trunk of the ferret project in the subversion repository at PMEL) 
using git-svn.

**If you build Ferret from these source files, please note:**  
The `site_specific.mk` and `external_functions/ef_utilites/site_specific.mk` 
files in the repository have been renamed with a `.in` appended to the name. 
You must copy these files with the `.in` extensions to create files with the 
`site_specific.mk` name and edit the contents to configure these for your 
system.  The `site_specific.mk` files will be ignored by git (the name was 
added to `.gitignore`) so your customized configuration files will not be 
added to your repository if you have cloned this repository. 
