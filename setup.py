from numpy.distutils.core import setup, Extension
import distutils.sysconfig
import sys
import os
import os.path

# (Non-standard) Directories containing .h include files
incdir_list = [ "pyfermod",
                os.path.join("fer", "common"),
                os.path.join("fmt", "cmn"),
                os.path.join("fer", "ef_utility"),
                os.path.join("fer", "grdel"), ]

# Non-standard directories containing libraries to link
netcdf4_libdir = os.getenv("NETCDF4_LIBDIR")
if netcdf4_libdir:
    netcdf4_libdir = netcdf4_libdir.strip()
if not netcdf4_libdir:
    raise ValueError("Environment variable NETCDF4_LIBDIR is not defined")
hdf5_libdir = os.getenv("HDF5_LIBDIR")
if hdf5_libdir:
    hdf5_libdir = hdf5_libdir.strip()
if not hdf5_libdir:
    raise ValueError("Environment variable HDF5_LIBDIR is not defined")
# CAIRO_LIBDIR is only given if the cairo library is to be statically linked in
cairo_libdir = os.getenv("CAIRO_LIBDIR")
if cairo_libdir:
    cairo_libdir = cairo_libdir.strip()
# The location of libpython2.x.so, in case it is not in a standard location
python_libdir = os.path.split(
                   distutils.sysconfig.get_python_lib(standard_lib=True))[0]
# The list of additional directories to examine for libraries
if cairo_libdir:
    libdir_list = [ "lib", netcdf4_libdir, hdf5_libdir,
                    cairo_libdir, python_libdir, ]
else:
    libdir_list = [ "lib", netcdf4_libdir, hdf5_libdir,
                    python_libdir, ]

# Get the list of ferret static libraries
# Stripping off the "lib" prefix and the ".a" suffix
fer_lib_list = [ ]
for libname in os.listdir("lib"):
    if (libname[:3] == "lib") and (libname[-2:] == ".a"):
        fer_lib_list.append(libname[3:-2])
# Create the list of libraries to link
# fer_lib_list is included multiple times to resolve interdependencies
lib_list = fer_lib_list[:]
lib_list.extend(fer_lib_list)
lib_list.extend(fer_lib_list)
lib_list.extend(fer_lib_list)
lib_list.extend(fer_lib_list)
# Add required system libraries to the list to link in
lib_list.append("python%i.%i" % sys.version_info[:2])

#
# Linking in the rest of the system libraries were moved to addn_link_flags
# in order to make sure the static netcdff, netcdf, hdf5_hl, and hdf5
# (and possibly cairo) libraries are used.
#
# lib_list.extend( ( "netcdff", "netcdf", "hdf5_hl", "hdf5",
#                    "cairo", "gfortran", "curl", "z", "dl", "m", ) )

#
# The pixman-1, freetype, fontconfig, png12, Xrender, and X11 libraries
# are only used to resolve cairo library function calls when statically
# linking in the cairo-1.8.8 library.
#
# Make sure everything is resolved in the final linking step.
#
netcdff_lib = "-Wl," + os.path.join(netcdf4_libdir, "libnetcdff.a")
netcdf_lib = "-Wl," + os.path.join(netcdf4_libdir, "libnetcdf.a")
hdf5_hl_lib = "-Wl," + os.path.join(hdf5_libdir, "libhdf5_hl.a")
hdf5_lib = "-Wl," + os.path.join(hdf5_libdir, "libhdf5.a")
if cairo_libdir:
   cairo_lib = "-Wl," + os.path.join(cairo_libdir, "libcairo.a")
   addn_link_args = [ netcdff_lib, netcdf_lib, hdf5_hl_lib, hdf5_lib,
                      cairo_lib, "-lpixman-1", "-lfreetype",
                      "-lfontconfig", "-lpng12", "-lXrender", "-lX11",
                      "-lgfortran", "-lcurl", "-lz", "-ldl",
                      "-lm", "-fPIC", "-Wl,--no-undefined", ]
else:
   addn_link_args = [ netcdff_lib, netcdf_lib, hdf5_hl_lib, hdf5_lib,
                      "-lcairo", "-lgfortran", "-lcurl", "-lz", "-ldl",
                      "-lm", "-fPIC", "-Wl,--no-undefined", ]

# Get the list of C source files in pyfermod
src_list = [ ]
for srcname in os.listdir("pyfermod"):
    if srcname[-2:] == ".c":
        src_list.append(os.path.join("pyfermod", srcname))

# Get the list of additional objects to be linked in
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

# Create the pyferret.libpyferret Extension
ext_mods = [ Extension("pyferret.libpyferret", include_dirs = incdir_list,
                                               sources = src_list,
                                               extra_objects = addnobjs_list,
                                               library_dirs = libdir_list,
                                               libraries = lib_list,
                                               extra_link_args = addn_link_args), ]

pyferret_version = os.getenv("PYFERRET_VERSION")
if pyferret_version == None:
    raise ValueError("Environment variable PYFERRET_VERSION is not defined")

# Configure the setup
setup(name = "pyferret",
      version = pyferret_version,
      description = "Python module providing Ferret functionality",
      long_description = "Python module providing Ferret functionality",
      author = "Karl M. Smith",
      author_email = "karl.smith@noaa.gov",
      url = "http://ferret.pmel.noaa.gov/Ferret/documentation/pyferret",
      license = "Public Domain",
      requires = [ "numpy", ],
      packages = [ "pyferret", "pyferret.eofanal", "pyferret.fershp",
                   "pyferret.graphbind", "pyferret.regrid", "pyferret.stats", ],
      package_dir = { "pyferret":"pyfermod", },
      ext_modules = ext_mods)

setup(name = "pipedviewer",
      version = "0.0.3",
      description = "Graphics viewer controlled by a command pipe",
      long_description = "A graphics viewer application that receives its " \
                         "drawing and other commands primarily from another " \
                         "application through a pipe.  A limited number of " \
                         "commands are provided by the viewer itself to allow " \
                         "saving and some manipulation of the displayed scene.  " \
                         "The controlling application, however, will be unaware " \
                         "of these modifications made to the scene.",
      author = "Karl M. Smith",
      author_email = "karl.smith@noaa.gov",
      url = "http://ferret.pmel.noaa.gov/Ferret/documentation/pyferret",
      license = "Public Domain",
      requires = [ "multiprocessing", ],
      packages = [ "pipedviewer", ],
      package_dir = { "pipedviewer":"pviewmod", })

setup(name = "gcircle",
      version = "0.0.1",
      description = "Module of functions involving great circles with " \
                    "points given in longitudes and latitudes (thus " \
                    "assuming a spheroid model of the earth).",
      long_description = "Module of functions involving great circles with " \
                         "points given in longitudes and latitudes (thus " \
                         "assuming a spheroid model of the earth).",
      author = "Karl M. Smith",
      author_email = "karl.smith@noaa.gov",
      url = "http://ferret.pmel.noaa.gov/Ferret/documentation/pyferret",
      license = "Public Domain",
      requires = [ "numpy", ],
      py_modules = [ "gcircle", ])

