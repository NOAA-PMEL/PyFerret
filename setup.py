from numpy.distutils.core import setup, Extension
import sys
import os
import os.path

# Make sure everything is resolved
addn_link_args = [ "-fPIC", "-rdynamic", "-Xlinker", "--no-undefined", ]

# (Non-standard) Directories containing .h include files
incdir_list = [ "pyfermod",
                os.path.join("fer", "common"),
                os.path.join("fmt", "cmn"),
                os.path.join("fer", "ef_utility"), ]

# Non-standard directories containing libraries to link
hdf5_libdir = os.getenv("HDF5_LIBDIR")
if hdf5_libdir == None:
    raise ValueError, "Environment variable HDF5_LIBDIR is not defined"
netcdf4_libdir = os.getenv("NETCDF4_LIBDIR")
if netcdf4_libdir == None:
    raise ValueError, "Environment variable NETCDF4_LIBDIR is not defined"
readline_libdir = os.getenv("READLINE_LIBDIR")
if readline_libdir == None:
    raise ValueError, "Environment variable READLINE_LIBDIR is not defined"
libz_libdir = os.getenv("LIBZ_LIBDIR")
if libz_libdir == None:
    raise ValueError, "Environment variable LIBZ_LIBDIR is not defined"
libdir_list = [ "lib", str(hdf5_libdir), str(netcdf4_libdir), str(readline_libdir), str(libz_libdir) ]

# Get the list of ferret static libraries
# Stripping off the "lib" prefix and the ".a" suffix
fer_lib_list = [ ]
for libname in os.listdir("lib"):
    fer_lib_list.append(libname[3:-2])
# Create the list of libraries to link
# fer_lib_list is included multiple times to resolve interdependencies
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

# Create the pyferret._pyferret Extension
ext_mods = [ Extension("pyferret._pyferret", include_dirs = incdir_list,
                                             sources = src_list,
                                             extra_objects = addnobjs_list,
                                             library_dirs = libdir_list,
                                             libraries = lib_list,
                                             extra_link_args = addn_link_args), ]

# Configure the setup
setup(name = "pyferret",
      version = "7.0.0.5",
      description = "python module providing ferret functionality",
      long_description = "python module providing ferret functionality",
      author = "Karl M. Smith",
      author_email = "karl.smith@noaa.gov",
      url = "http://ferret.pmel.noaa.gov/Ferret",
      packages = [ "pyferret", "pyferret.stats", "pyferret.fershp", ],
      package_dir = { "pyferret":"pyfermod", },
      ext_modules = ext_mods)
setup(name = "queuedviewer",
      version = "0.0.0.1",
      description = "graphics viewer controlled by a command queue",
      long_description = "A graphics viewer application that receives its " \
                         "drawing and other commands primarily from another " \
                         "application through a queue.  A limited number of " \
                         "commands are provided by the viewer itself to allow " \
                         "saving and some manipulation of the displayed scene.  " \
                         "The controlling application, however, will be unaware " \
                         "of these modifications made to the scene.",
      author = "Karl M. Smith",
      author_email = "karl.smith@noaa.gov",
      url = "http://ferret.pmel.noaa.gov/Ferret",
      packages = [ "queuedviewer", ],
      package_dir = { "queuedviewer":"qviewmod", })

