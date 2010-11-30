from numpy.distutils.core import setup, Extension
import os
import os.path

# Note: the shared-object library libferret.so needs to be built before building the pyferret modules

# Create the pyferret._pyferret Extension
ext_mods = [ Extension("pyferret._pyferret", sources = [ os.path.join("pyfer", "_pyferretmodule.c"), ],
                                             include_dirs = [ "ferlib", 
                                                              os.path.join("fer", "common"), 
                                                              os.path.join("fmt", "cmn"), 
                                                              os.path.join("fer", "ef_utility"), ],
                                             library_dirs = [ "ferlib", ],
                                             libraries = [ "ferret", "python2.6", ]), ]

# Configure the setup
setup(name = "pyferret", 
      version = "7.0",
      description = "python package providing ferret functionality",
      long_description = "python package providing ferret functionality",
      author = "Karl M. Smith",
      author_email = "karl.smith@noaa.gov",
      url = "http://ferret.pmel.noaa.gov/Ferret",
      packages = [ "pyferret", ],
      package_dir = { "pyferret":"pyfer", },
      ext_modules = ext_mods)

