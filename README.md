# Ferret
The Ferret program from NOAA/PMEL. 
This repository is regularly synchronized with Ferret repository at PMEL 
(the trunk of the ferret project in the subversion repository at PMEL) 
using git-svn.


*This repository is a scientific product and is not official communication 
of the National Oceanic and Atmospheric Administration, or the United 
States Department of Commerce.  All NOAA GitHub project code is provided 
on an 'as is' basis and the user assumes responsibility for its use.  Any 
claims against the Department of Commerce or Department of Commerce bureaus 
stemming from the use of this GitHub project will be governed by all 
applicable Federal law.  Any reference to specific commercial products, 
processes, or services by service mark, trademark, manufacturer, or 
otherwise, does not constitute or imply their endorsement, recommendation 
or favoring by the Department of Commerce.  The Department of Commerce 
seal and logo, or the seal and logo of a DOC bureau, shall not be used 
in any manner to imply endorsement of any commercial product or activity 
by DOC or the United States Government.*


**If you build Ferret from these source files, please note:**  
The `site_specific.mk` and `external_functions/ef_utilites/site_specific.mk` 
files in the repository have been renamed with a `.in` appended to the name. 
You must copy these files with the `.in` extensions to create files with the 
`site_specific.mk` name and edit the contents to configure these for your 
system.  The `site_specific.mk` files will be ignored by git (the name was 
added to `.gitignore`) so your customized configuration files will not be 
added to your repository if you have cloned this repository. 
