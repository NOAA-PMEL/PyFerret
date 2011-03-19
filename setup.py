from numpy.distutils.core import setup, Extension
import sys
import os
import os.path

# Note: the shared-object library libferret.so needs to be built before building the pyferret modules

# Make sure everything is resolved
addn_link_args = [ "-Xlinker", "--no-undefined", ]

# Directories containg .h include files
incdir_list = [ "pyfermod",
                os.path.join("fer", "common"),
                os.path.join("fmt", "cmn"),
                os.path.join("fer", "ef_utility"), ]

# Get the list of ferret static libraries
# Stripping off the "lib" prefix and the ".a" suffix
fer_lib_list = [ ]
for libname in os.listdir("lib"):
    fer_lib_list.append(libname[3:-2])
# Create the list of libraries to link
# fer_lib_list is included three times to resolve interdependencies
lib_list = fer_lib_list[:]
lib_list.extend(fer_lib_list)
lib_list.extend(fer_lib_list)
lib_list.extend(fer_lib_list)
lib_list.append("python%i.%i" % sys.version_info[:2])
lib_list.extend( ( "netcdff", "netcdf", "hdf5_hl", "hdf5",
                   "readline", "history", "ncurses", "X11",
                   "curl", "z", "dl", "gfortran", "m", ) )

# Get the list of C source files in pyfermod
src_list = [ ]
for srcname in os.listdir("pyfermod"):
    if srcname[-2:] == ".c":
        src_list.append(os.path.join("pyfermod", srcname))

# Get the list of extra additional objects to be linked in
addnobjs_list = [ ]
dirname = os.path.join("fer", "ef_utility")
for srcname in os.listdir(dirname):
    if srcname[-2:] == ".o":
        addnobjs_list.append(os.path.join(dirname, srcname))
dirname = os.path.join("fer", "special")
for srcname in ( "fakes3.o", "ferret_dispatch.o", "ferret_query_f.o", 
                 "gui_fakes.o", "linux_routines.o", ):
    addnobjs_list.append(os.path.join(dirname, srcname))
for srcname in os.listdir(dirname):
    if (srcname[0] == 'x') and (srcname[-7:] == "_data.o"):
        addnobjs_list.append(os.path.join(dirname, srcname))
# dirname = os.path.join("ppl", "tmapadds")
# for srcname in os.listdir(dirname):
#     if srcname[-2:] == ".o":
#         addnobjs_list.append(os.path.join(dirname, srcname))

# Create the pyferret._pyferret Extension
ext_mods = [ Extension("pyferret._pyferret", include_dirs = incdir_list, 
                                             sources = src_list,
                                             extra_objects = addnobjs_list,
                                             library_dirs = [ "lib", "/usr/local/lib", ],
                                             libraries = lib_list,
                                             extra_link_args = addn_link_args), ]

# Configure the setup
setup(name = "pyferret",
      version = "7.0.0",
      description = "python package providing ferret functionality",
      long_description = "python package providing ferret functionality",
      author = "Karl M. Smith",
      author_email = "karl.smith@noaa.gov",
      url = "http://ferret.pmel.noaa.gov/Ferret",
      py_modules = [ "ferret", ],
      packages = [ "pyferret", ],
      package_dir = { "pyferret":"pyfermod", },
      ext_modules = ext_mods)

