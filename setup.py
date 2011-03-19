from numpy.distutils.core import setup, Extension
import sys
import os
import os.path

# Note: the shared-object library libferret.so needs to be built before building the pyferret modules

# Make sure everything is resolved
addn_link_args = [ "-Xlinker", "--no-undefined", ]

# Create the pyferret._pyferret Extension
ext_mods = [ Extension("pyferret._pyferret", sources = [ os.path.join("pyfermod", "_pyferretmodule.c"), ],
                                             include_dirs = [ os.path.join("fer", "common"), 
                                                              os.path.join("fmt", "cmn"), 
                                                              os.path.join("fer", "ef_utility"), ],
                                             library_dirs = [ "lib", ],
                                             libraries = [ "ferret", "python%i.%i" % sys.version_info[:2], ],
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

