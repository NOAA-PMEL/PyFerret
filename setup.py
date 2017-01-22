from numpy.distutils.core import setup, Extension
import distutils.sysconfig
import sys
import os
import os.path
import re

# Get BUILDTYPE for checking if this is intel-mac
buildtype = os.getenv("BUILDTYPE")
if buildtype:
    buildtype = buildtype.strip()
if not buildtype:
    raise ValueError("Environment variable BUILDTYPE is not defined")

# (Non-standard) Directories containing .h include files
incdir_list = [ "pyfermod",
                os.path.join("fer", "common"),
                os.path.join("fmt", "cmn"),
                os.path.join("fer", "ef_utility"),
                os.path.join("fer", "grdel"), ]

is_linux_system = os.getenv("IS_LINUX_SYSTEM")
if is_linux_system:
    is_linux_system = is_linux_system.strip()

# NETCDF4_LIBDIR must be given, either for the static library or the shared-object library
netcdf4_libdir = os.getenv("NETCDF4_LIBDIR")
if netcdf4_libdir:
    netcdf4_libdir = netcdf4_libdir.strip()
if not netcdf4_libdir:
    raise ValueError("Environment variable NETCDF4_LIBDIR is not defined")

# HDF5_LIBDIR is only given if the HDF5 and NetCDF libraries are to be statically linked in
# COMPRESS_LIB is the compression library used by this HDF5, if it is given
hdf5_libdir = os.getenv("HDF5_LIBDIR")
if hdf5_libdir:
    hdf5_libdir = hdf5_libdir.strip()
    compress_lib = os.getenv("COMPRESS_LIB")
    if compress_lib:
        compress_lib = compress_lib.strip()
    if not compress_lib in ('z', 'sz'):
        raise ValueError("Environment variable COMPRESS_LIB must be either 'z' or 'sz'")
    compress_lib = '-l' + compress_lib

# CAIRO_LIBDIR is only given if the cairo library is to be statically linked in
cairo_libdir = os.getenv("CAIRO_LIBDIR")
if cairo_libdir:
    cairo_libdir = cairo_libdir.strip()

# PIXMAN_LIBDIR is only given if the pixman-1 library is to be statically linked in
pixman_libdir = os.getenv("PIXMAN_LIBDIR")
if pixman_libdir:
    pixman_libdir = pixman_libdir.strip()

# PANGO_LIBDIR gives a non-standard location of the pango libraries and
# the libraries it uses (ie, pangocairo, pango, freetype, fontconfig, png)
pango_libdir = os.getenv("PANGO_LIBDIR")
if pango_libdir:
    pango_libdir = pango_libdir.strip()

# The location of libpythonx.x.so, in case it is not in a standard location
python_libdir = os.path.split( distutils.sysconfig.get_python_lib(standard_lib=True) )[0]

# The list of additional directories to examine for libraries
libdir_list = [ "lib", netcdf4_libdir, ]
if hdf5_libdir:
    libdir_list.append(hdf5_libdir)
if cairo_libdir:
    libdir_list.append(cairo_libdir)
    if pixman_libdir:
        libdir_list.append(pixman_libdir)
    if pango_libdir:
        libdir_list.append(pango_libdir)
libdir_list.append(python_libdir)

# Get the list of ferret static libraries
# Stripping off the "lib" prefix and the ".a" suffix
fer_lib_list = [ ]
for libname in os.listdir("lib"):
    if (libname[:3] == "lib") and (libname[-2:] == ".a"):
        fer_lib_list.append(libname[3:-2])

# Create the list of libraries to link
lib_list = fer_lib_list[:]
if buildtype != "intel-mac":
    # fer_lib_list is included multiple times to resolve interdependencies
    lib_list.extend(fer_lib_list)
    lib_list.extend(fer_lib_list)
    lib_list.extend(fer_lib_list)
    lib_list.extend(fer_lib_list)
# Add required system libraries to the list to link in
lib_list.append("python%d.%d" % sys.version_info[:2])

# Linking in the rest of the system libraries were moved to addn_link_flags
# in order to make sure the appropriate netcdff, netcdf, hdf5_hl, hdf5, and
# cairo libraries are used.
#
# lib_list.extend( ( "netcdff", "netcdf", "hdf5_hl", "hdf5",
#                    "cairo", "gfortran", "curl", "z", "dl", "m", ) )

addn_link_args = [ ]
# Link to the appropriate netcdf libraries.
# The hdf5 libraries are only used to resolve netcdf library function
# calls when statically linking in the netcdf libraries.
if hdf5_libdir:
    netcdff_lib = "-Wl," + os.path.join(netcdf4_libdir, "libnetcdff.a")
    addn_link_args.append(netcdff_lib)
    netcdf_lib = "-Wl," + os.path.join(netcdf4_libdir, "libnetcdf.a")
    addn_link_args.append(netcdf_lib)
    hdf5_hl_lib = "-Wl," + os.path.join(hdf5_libdir, "libhdf5_hl.a")
    addn_link_args.append(hdf5_hl_lib)
    hdf5_lib = "-Wl," + os.path.join(hdf5_libdir, "libhdf5.a")
    addn_link_args.append(hdf5_lib)
else:
    addn_link_args.extend([ "-lnetcdff", "-lnetcdf" ])

# Link to the cairo library and the libraries it requires.
if cairo_libdir:
    cairo_lib = "-Wl," + os.path.join(cairo_libdir, "libcairo.a")
    addn_link_args.append(cairo_lib);
    if pixman_libdir:
        pixman_lib = "-Wl," + os.path.join(pixman_libdir, "libpixman-1.a")
    else:
        pixman_lib = "-lpixman-1"
    addn_link_args.append(pixman_lib);
    addn_link_args.extend([ "-lfreetype", "-lfontconfig", "-lpng" ])
    if is_linux_system:
        # Bind symbols and function symbols to any internal definitions
        # and do not make any of the symbols or function symbols defined
        # in any libraries externally visible (mainly for cairo and pixman).
        # Those in the object files (including those from pyfermod and
        # fer/ef_utility) will still be visible.
        addn_link_args.extend([ "-lXrender", "-lX11",
                                "-Wl,-Bsymbolic", "-Wl,--exclude-libs -Wl,ALL"])
else:
    addn_link_args.append("-lcairo")

# The Pango text-rendering libraries
addn_link_args.extend([ "-lpangocairo-1.0", "-lpango-1.0", "-lgobject-2.0" ])

# Link in the appropriate system libraries
if hdf5_libdir:
    addn_link_args.extend(["-lcurl", compress_lib])
addn_link_args.extend([ "-lgfortran", "-ldl", "-lm", "-fPIC", ])

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

if cairo_libdir:
    # Duplicate objects in libraries to make them externally visible (e.g., for las
    # external functions) if the '--exclude-libs ALL' flag was passed to the linker.
    dirname = os.path.join("fmt", "src")
    addnobjs_list.append(os.path.join(dirname, "tm_lenstr.o"));
    addnobjs_list.append(os.path.join(dirname, "tm_fmt.o"));
    addnobjs_list.append(os.path.join(dirname, "tm_lefint.o"));

# Create the pyferret.libpyferret Extension
ext_mods = [ Extension("pyferret.libpyferret", include_dirs = incdir_list,
                                               sources = src_list,
                                               extra_objects = addnobjs_list,
                                               library_dirs = libdir_list,
                                               libraries = lib_list,
                                               extra_link_args = addn_link_args), ]

pyferret_version = None
xrev_name = os.path.join("fer", "dat", "xrevision_data.F")
xrev_file = open(xrev_name)
try:
    pat = re.compile('\\s+DATA\\s+revision_level\\s*/\\s*(\\S+)\\s*/\\s*', flags=re.IGNORECASE)
    for datlin in xrev_file:
        mat = re.match(pat, datlin)
        if mat:
            pyferret_version = mat.group(1)
            break
finally:
    xrev_file.close()
if not pyferret_version:
    raise ValueError('Unable to find the version number in ' + xrev_name)

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
      version = pyferret_version,
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
      version = pyferret_version,
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

