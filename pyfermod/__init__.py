"""
A Python module for running Ferret.
For running the Ferret engine that is part of this module:
    init or start must first be called to initialize Ferret
    resize can be used to resize Ferret's allocated memory block
    run is used to submit individual Ferret commands or enter
            into Ferret's command prompting mode
    stop can be used to shutdown Ferret and free the allocated
            memory.

The FERR_* values are the possible values of err_int in the return
values from the run command.  The err_int return value FERR_OK
indicates no errors.

To transfer data and metadata between the Ferret engine and Python,
see the help message for pyferret.datamethods, whose methods are
imported into the pyferret module.

For writing Ferret external functions in Python, see the help message
for pyferret.pyefmethods, whose methods are imported into the pyferret
module.

The FerRegion, FerAxis, FerGrid, FerVar, FerPyVar, and FerDSet objects
assist in working with the Ferret engine in a Python environment,
reducing the need to know the Ferret language and syntax.

The convenience methods for executing common Ferret commands are found
under pyferret.fermethods and are imported into the pyferret module.
These methods also help reduce the need to know the Ferret language
and syntax.
"""

from __future__ import print_function

import sys
import os
import atexit
try:
    import rlcompleter
except ImportError:
    pass
import readline

from pyferret import libpyferret
# also import everything (not starting with an underscore) from libpyferret
# so constants in that module are seen as part of this module
from pyferret.libpyferret import *

# methods for transferring data between the Ferret engine and Python
from pyferret import datamethods
# also import the methods given in datamethods into pyferret
from pyferret.datamethods import *

# methods to assist in writing Ferret external functions written in Python
from pyferret import pyefmethods
# also import the methods given in pyefmethods into pyferret
from pyferret.pyefmethods import *

# the Fer... objects for working with Ferret from Python
from pyferret import ferregion
from pyferret.ferregion import *
from pyferret import feraxis
from pyferret.feraxis import *
from pyferret import fergrid
from pyferret.fergrid import *
from pyferret import fervar
from pyferret.fervar import *
from pyferret import ferpyvar
from pyferret.ferpyvar import *
from pyferret import ferdset
from pyferret.ferdset import *
from pyferret import feraggdset
from pyferret.feraggdset import *
from pyferret import ferfmrcdset
from pyferret.ferfmrcdset import *
from pyferret import ferfmrcvar
from pyferret.ferfmrcvar import *

# Create "the" anonymous dataset
anondset = FerDSet('', qual=ferdset._anonymous_dataset_qualifier)

# convenience methods for executing common Ferret commands
from pyferret import fermethods
from pyferret.fermethods import *

# bindings for the PyQt-based graphics engines
import pipedviewer.pyferretbindings

from pyferret import filenamecompleter
from pyferret import graphbind
from pyferret import regrid


def _addtoenvdirlist(envname, dirname):
    """
    Adds the directory path given by dirname to the list of directory
    paths given in the environment variable envname.  The directory 
    path is added to the beginning of the environment value, but 
    immediately after '.'  if that is the first directory in the list.
    If the environment variable given in envname does not exist, it 
    is initialized to '.' before adding the given directory path. 
    No checking or modification of the directory path is performed.
    """
    currval = os.getenv(envname, '.')
    # Spaces in directory names are a problem for ferret 
    # so do not worry about it here. 
    dirlist = currval.split()
    if dirlist[0] == '.':
        dirlist.insert(1, dirname)
    else:
        dirlist.insert(0, dirname);
    os.environ[envname] = ' '.join(dirlist)


def addenv(FER_DIR=None, FER_DAT=None, FER_DATA=None, FER_DESCR=None, 
           FER_GRIDS=None, FER_GO=None, FER_FONTS=None, FER_PALETTE=None):
    """
    A convenience method that adds the specified directory paths to the 
    current Ferret environment variables in the current Python session. 
    Each value should be a single directory path.  More directories can 
    be added to an environment variable, if appropriate, by calling this 
    method again.  

    Except for FER_DIR, FER_DAT, and FER_FONTS (see below), the directory 
    specified will be added to the beginning of the list of directory paths 
    given by the environment variable, but immediately after '.' (which 
    represents the current working directory) if that is the first directory 
    in the list.  If an enviroment variable is not defined, it will be 
    initialized to '.' before adding the given directory path.

    The value for FER_DIR, FER_DAT or FER_FONTS, if specified, replaces 
    the current value (if any) for these environment variables. (These 
    values are each suppose to be a single directory path).  

    The value for FER_DIR (which actually is not used directly in PyFerret)
    is also used to construct directory paths to be added, if the subdirectory 
    exists, to the enviroment variables: 
        FER_GO ("$FER_DIR/go"), 
        FER_PALETTE ("$FER_DIR/ppl"), 
        FER_FONTS ("$FER_DIR/ppl/fonts"), and 
        LD_LIBRARY_PATH ("$FER_DIR/lib/pythonX.x/site-packages/pyferret", 
                         where X and x are the major and minor version 
                         numbers of this python). 
    These directory paths will be added prior to any directories specified 
    by other arguments.

    The value for FER_DAT (which actually is not used directly in PyFerret) 
    is also used to construct directory paths to be added, if the subdirectory 
    exists, to the environment variables:
        FER_DATA ("$FER_DAT/data"),
        FER_DESCR ("$FER_DAT/descr"), and
        FER_GRIDS ("$FER_DAT/grids")
    These directory paths will be added prior to any directories specified 
    by other arguments.

    Directory paths can use '~/' and '~user/' constructs, as well as shell 
    variables of the form '$var' and '${var}'.  Relative paths can also be 
    specified and will be used as-is (they are not converted to absolute 
    paths).

    Use os.environ to reset a Ferret environment variable to some specific
    value; for example, 
        os.environ['FER_DATA'] = '. /path/to/my/datasets'

    To replicate setting most of the environment variables assigned by the 
    ferret_paths scripts, call this method with the FER_DIR and FER_DAT
    arguments assigned the values used in generating the ferret_paths
    scripts.  So a python session using pyferret without using the 
    ferret_paths scripts will first require adding the 
        pyferret/lib/pythonX.x/site-packages 
    directory path, if not in a standard python location, to either the 
    shell environment variable PYTHONPATH or the Python list of paths given 
    in sys.path so the pyferret module can be found.  The rest of the
    session will look something like:
        import pyferret
        pyferret.addenv(FER_DIR='/path/to/pyferret/dir', 
                        FER_DAT='/path/to/ferret/datasets/dir')
        pyferret.start()
    """
    if FER_DIR:
        myferdir = os.path.expandvars(os.path.expanduser(FER_DIR))
        if not os.path.isdir(myferdir):
            raise ValueError('FER_DIR is given but is not a directory: %s' % myferdir)
        os.environ['FER_DIR'] = myferdir
        # Set FER_FONTS to ppl/fonts subdir
        myferfonts = os.path.join(myferdir, 'ppl', 'fonts')
        if os.path.isdir(myferfonts):
            os.environ['FER_FONTS'] = myferfonts
        # Add go subdir to FER_GO
        dirtoadd = os.path.join(myferdir, 'go')
        if os.path.isdir(dirtoadd):
            _addtoenvdirlist('FER_GO', dirtoadd)
        # Add ppl subdir to FER_PALETTE
        dirtoadd = os.path.join(myferdir, 'ppl')
        if os.path.isdir(dirtoadd):
            _addtoenvdirlist('FER_PALETTE', dirtoadd)
        # Add lib/pythonX.x/site-packages/pyferret subdir to LD_LIBRARY_PATH
        # but separate with appropriate separator (':') and do not worry about '.'
        dirtoadd = os.path.join(myferdir, 'lib', 'python%d.%d' % sys.version_info[:2], 
                                'site-packages', 'pyferret')
        currlist = os.getenv('LD_LIBRARY_PATH')
        if not currlist:
            os.environ['LD_LIBRARY_PATH'] = dirtoadd
        else:
            os.environ['LD_LIBRARY_PATH'] = dirtoadd + os.pathsep + currlist
    if FER_DAT:
        myferdat = os.path.expandvars(os.path.expanduser(FER_DAT))
        if not os.path.isdir(myferdat):
            raise ValueError('FER_DAT is given but is not a directory: %s' % myferdat)
        os.environ['FER_DAT'] = myferdat
        # Add data subdir to FER_DATA
        dirtoadd = os.path.join(myferdat, 'data')
        if os.path.isdir(dirtoadd):
            _addtoenvdirlist('FER_DATA', dirtoadd)
        # Add descr subdir to FER_DESCR
        dirtoadd = os.path.join(myferdat, 'descr')
        if os.path.isdir(dirtoadd):
            _addtoenvdirlist('FER_DESCR', dirtoadd)
        # Add grids subdir to FER_GRIDS
        dirtoadd = os.path.join(myferdat, 'grids')
        if os.path.isdir(dirtoadd):
            _addtoenvdirlist('FER_GRIDS', dirtoadd)
    if FER_FONTS:
        myferfonts = os.path.expandvars(os.path.expanduser(FER_FONTS))
        if not os.path.isdir(myferfonts):
            raise ValueError('FER_FONTS is given but is not a directory: %s' % myferfonts)
        os.environ['FER_FONTS'] = myferfonts
    if FER_GO:
        dirtoadd = os.path.expandvars(os.path.expanduser(FER_GO))
        if not os.path.isdir(dirtoadd):
            raise ValueError('FER_GO is given but is not a directory: %s' % dirtoadd)
        _addtoenvdirlist('FER_GO', dirtoadd)
    if FER_PALETTE:
        dirtoadd = os.path.expandvars(os.path.expanduser(FER_PALETTE))
        if not os.path.isdir(dirtoadd):
            raise ValueError('FER_PALETTE is given but is not a directory: %s' % dirtoadd)
        _addtoenvdirlist('FER_PALETTE', dirtoadd)
    if FER_DATA:
        dirtoadd = os.path.expandvars(os.path.expanduser(FER_DATA))
        if not os.path.isdir(dirtoadd):
            raise ValueError('FER_DATA is given but is not a directory: %s' % dirtoadd)
        _addtoenvdirlist('FER_DATA', dirtoadd)
    if FER_DESCR:
        dirtoadd = os.path.expandvars(os.path.expanduser(FER_DESCR))
        if not os.path.isdir(dirtoadd):
            raise ValueError('FER_DESCR is given but is not a directory: %s' % dirtoadd)
        _addtoenvdirlist('FER_DESCR', dirtoadd)
    if FER_GRIDS:
        dirtoadd = os.path.expandvars(os.path.expanduser(FER_GRIDS))
        if not os.path.isdir(dirtoadd):
            raise ValueError('FER_GRIDS is given but is not a directory: %s' % dirtoadd)
        _addtoenvdirlist('FER_GRIDS', dirtoadd)


def addpath(newdir):
    """
    A convenience method that replicates the Ferret Faddpath script by 
    adding the given directory to the environment variables FER_GO, 
    FER_DATA, FER_DESCR, and FER_GRIDS in the current Python session. 
    """
    addenv(FER_GO=newdir, FER_DATA=newdir, FER_DESCR=newdir, FER_GRIDS=newdir)


def init(arglist=None, enterferret=True):
    """
    Interprets the traditional Ferret options given in arglist and
    starts pyferret appropriately.  Defines all the standard Ferret
    Python external functions.  If ${HOME}/.ferret exists, that
    script is then executed.

    If '-script' is given with a script filename, this method calls
    the run method with the ferret go command, the script filename,
    and any arguments, and then exits completely (exits python).

    Otherwise, if enterferret is False, this method just returns the
    success return value of the run method: (FERR_OK, '')

    If enterferret is True (unless '-python' is given in arglist)
    this routine calls the run method with no arguments in order to
    enter into Ferret command-line processing.  The value returned
    from call to the run method is then returned.
    """

    ferret_help_message = \
    """

    Usage:  pyferret  [-memsize <N>]  [-nodisplay]  [-png]  [-nojnl]  [-noverify]
                      [-secure]  [-server]  [-python]  [-version]  [-help]
                      [-quiet]  [-linebuffer]  [-script <scriptname> [ <scriptarg> ... ]]

       -memsize:     initialize the memory cache size to <N> (default 25)
                     mega (10^6) floats (where 1 float = 8 bytes)

       -nodisplay    do not display to the console; a drawing can be saved
                     using the FRAME command in any of the supported file
                     formats.  The /QUALITY option of SET WINDOW will be
                     ignored when this is specified.

       -png          do not display to the console and only raster (PNG)
                     images can be saved.  Implies -nodisplay and removes
                     support for deletion of parts (segments) of a plot;
                     however, plots will be generated faster.

       -nojnl:       on startup do not open a journal file (can be turned
                     on later with SET MODE JOURNAL)

       -noverify:    on startup turn off verify mode (can be turned on
                     later with SET MODE VERIFY)

       -secure:      restrict Ferret's capabilities (e.g., SPAWN and
                     EXIT /TOPYTHON are not permitted)

       -server:      run in server mode (don't stop on message commands)

       -python:      start at the Python prompt instead of the Ferret prompt.
                     The Ferret prompt can be obtained using 'pyferret.run()'

       -version:     print the Ferret header with version number and quit

       -help:        print this help message and quit

       -quiet        do not display the startup header

       -linebuffer   print each line of output or error messages as soon as
                     a full line is written.  Useful when redirecting both
                     types of messages to a single file.
                     Note:
                       the environment variable GFORTRAN_UNBUFFERED_PRECONNECTED
                       needs to be set to 1 in order to unbuffer the Fortran
                       units for output and error messages

       -script:      execute the script <scriptname> with any arguments
                     specified and exit. (THIS MUST BE SPECIFIED LAST.)
                     The -script option also implies the -nojnl, -noverify,
                     -server, and -quiet options.

       Deprecated command-line options:

           -gif:         this option is now an alias for -png

	   -unmapped:    this option is now an alias for -nodisplay

           -batch:       draw to <filename> (default "ferret.png") instead of
                         displaying to the console.  The file format will be
                         guessed from the filename extension.  When using this
                         option, new windows should not be created and the
                         FRAME command should not be used.

           -transparent: use a transparent background instead of opaque white
                         when saving to the file given by -batch

        Use of -batch and -transparent options is strongly discouraged.  Instead,
        use the -nodisplay option and the FRAME /FILE=<filename> [ /TRANSPARENT ]
        Ferret command.
    """

    my_metaname = None
    my_transparent = False
    my_unmapped = False
    my_pngonly = False
    my_memsize = 25
    my_journal = True
    my_verify = True
    my_restrict = False
    my_server = False
    my_quiet = False
    my_linebuffer = False
    my_enterferret = enterferret
    script = None
    # To be compatible with traditional Ferret command-line options
    # (that are still supported), we need to parse the options by hand.
    if arglist:
        print_help = False
        just_exit = False
        try:
            k = 0
            while k < len(arglist):
                opt = arglist[k]
                if opt == "-memsize":
                    k += 1
                    try:
                        my_memsize = float(arglist[k])
                    except:
                        raise ValueError("a positive number must be given for a -memsize value")
                    if my_memsize <= 0.0:
                        raise ValueError("a positive number must be given for a -memsize value")
                elif opt == "-batch":
                    my_metaname = "ferret.png"
                    k += 1
                    # -batch has an optional argument
                    try:
                        if arglist[k][0] != '-':
                            my_metaname = arglist[k]
                        else:
                            k -= 1
                    except:
                        k -= 1
                elif opt == "-transparent":
                    my_transparent = True
                elif opt == "-nodisplay":
                    my_unmapped = True
                elif opt == "-unmapped":
                    my_unmapped = True
                elif opt == "-png":
                    my_unmapped = True
                    my_pngonly = True
                elif opt == "-gif":
                    my_unmapped = True
                    my_pngonly = True
                elif opt == "-nojnl":
                    my_journal = False
                elif opt == "-noverify":
                    my_verify = False
                elif opt == "-secure":
                    my_restrict = True
                elif opt == "-server":
                    my_server = True
                elif opt == "-quiet":
                    my_quiet = True
                elif opt == "-linebuffer":
                    my_linebuffer = True
                elif opt == "-python":
                    my_enterferret = False
                elif opt == "-version":
                    just_exit = True
                    break
                elif (opt == "-help") or (opt == "-h") or (opt == "--help"):
                    print_help = True
                    break
                elif opt == "-script":
                    my_journal = False
                    my_verify = False
                    my_server = True
                    my_quiet = True
                    k += 1
                    try:
                        script = arglist[k:]
                        if len(script) == 0:
                            raise ValueError("a script filename must be given for the -script value")
                    except:
                        raise ValueError("a script filename must be given for the -script value")
                    break
                else:
                    raise ValueError("unrecognized option '%s'" % opt)
                k += 1
        except ValueError as errmsg:
            # print the error message, then print the help message
            print("\n%s" % errmsg, file=sys.stderr)
            print_help = True
        if print_help:
            # print the help message, then mark for exiting
            print(ferret_help_message, file=sys.stderr)
            just_exit = True
        if just_exit:
            # print the ferret header then exit completely
            start(journal=False, verify=False, unmapped=True)
            result = run("exit /program")
            # should not get here
            raise SystemExit

    # Create the list of standard ferret PyEFs to create
    std_pyefs = [ ]
    # stats_* functions that do not need scipy
    std_pyefs.append("stats.stats_histogram")
    # stats_* functions that depend on scipy
    try:
        import scipy
        std_pyefs.extend((
                  "stats.stats_beta_cdf",
                  "stats.stats_beta_isf",
                  "stats.stats_beta_pdf",
                  "stats.stats_beta_ppf",
                  "stats.stats_beta_rvs",
                  "stats.stats_beta_sf",
                  "stats.stats_binom_cdf",
                  "stats.stats_binom_isf",
                  "stats.stats_binom_pmf",
                  "stats.stats_binom_ppf",
                  "stats.stats_binom_rvs",
                  "stats.stats_binom_sf",
                  "stats.stats_cauchy_cdf",
                  "stats.stats_cauchy_isf",
                  "stats.stats_cauchy_pdf",
                  "stats.stats_cauchy_ppf",
                  "stats.stats_cauchy_rvs",
                  "stats.stats_cauchy_sf",
                  "stats.stats_chi_cdf",
                  "stats.stats_chi_isf",
                  "stats.stats_chi_pdf",
                  "stats.stats_chi_ppf",
                  "stats.stats_chi_rvs",
                  "stats.stats_chi_sf",
                  "stats.stats_chi2_cdf",
                  "stats.stats_chi2_isf",
                  "stats.stats_chi2_pdf",
                  "stats.stats_chi2_ppf",
                  "stats.stats_chi2_rvs",
                  "stats.stats_chi2_sf",
                  "stats.stats_expon_cdf",
                  "stats.stats_expon_isf",
                  "stats.stats_expon_pdf",
                  "stats.stats_expon_ppf",
                  "stats.stats_expon_rvs",
                  "stats.stats_expon_sf",
                  "stats.stats_exponweib_cdf",
                  "stats.stats_exponweib_isf",
                  "stats.stats_exponweib_pdf",
                  "stats.stats_exponweib_ppf",
                  "stats.stats_exponweib_rvs",
                  "stats.stats_exponweib_sf",
                  "stats.stats_f_cdf",
                  "stats.stats_f_isf",
                  "stats.stats_f_pdf",
                  "stats.stats_f_ppf",
                  "stats.stats_f_rvs",
                  "stats.stats_f_sf",
                  "stats.stats_gamma_cdf",
                  "stats.stats_gamma_isf",
                  "stats.stats_gamma_pdf",
                  "stats.stats_gamma_ppf",
                  "stats.stats_gamma_rvs",
                  "stats.stats_gamma_sf",
                  "stats.stats_geom_cdf",
                  "stats.stats_geom_isf",
                  "stats.stats_geom_pmf",
                  "stats.stats_geom_ppf",
                  "stats.stats_geom_rvs",
                  "stats.stats_geom_sf",
                  "stats.stats_hypergeom_cdf",
                  "stats.stats_hypergeom_isf",
                  "stats.stats_hypergeom_pmf",
                  "stats.stats_hypergeom_ppf",
                  "stats.stats_hypergeom_rvs",
                  "stats.stats_hypergeom_sf",
                  "stats.stats_invgamma_cdf",
                  "stats.stats_invgamma_isf",
                  "stats.stats_invgamma_pdf",
                  "stats.stats_invgamma_ppf",
                  "stats.stats_invgamma_rvs",
                  "stats.stats_invgamma_sf",
                  "stats.stats_laplace_cdf",
                  "stats.stats_laplace_isf",
                  "stats.stats_laplace_pdf",
                  "stats.stats_laplace_ppf",
                  "stats.stats_laplace_rvs",
                  "stats.stats_laplace_sf",
                  "stats.stats_lognorm_cdf",
                  "stats.stats_lognorm_isf",
                  "stats.stats_lognorm_pdf",
                  "stats.stats_lognorm_ppf",
                  "stats.stats_lognorm_rvs",
                  "stats.stats_lognorm_sf",
                  "stats.stats_nbinom_cdf",
                  "stats.stats_nbinom_isf",
                  "stats.stats_nbinom_pmf",
                  "stats.stats_nbinom_ppf",
                  "stats.stats_nbinom_rvs",
                  "stats.stats_nbinom_sf",
                  "stats.stats_norm_cdf",
                  "stats.stats_norm_isf",
                  "stats.stats_norm_pdf",
                  "stats.stats_norm_ppf",
                  "stats.stats_norm_rvs",
                  "stats.stats_norm_sf",
                  "stats.stats_pareto_cdf",
                  "stats.stats_pareto_isf",
                  "stats.stats_pareto_pdf",
                  "stats.stats_pareto_ppf",
                  "stats.stats_pareto_rvs",
                  "stats.stats_pareto_sf",
                  "stats.stats_poisson_cdf",
                  "stats.stats_poisson_isf",
                  "stats.stats_poisson_pmf",
                  "stats.stats_poisson_ppf",
                  "stats.stats_poisson_rvs",
                  "stats.stats_poisson_sf",
                  "stats.stats_randint_cdf",
                  "stats.stats_randint_isf",
                  "stats.stats_randint_pmf",
                  "stats.stats_randint_ppf",
                  "stats.stats_randint_rvs",
                  "stats.stats_randint_sf",
                  "stats.stats_t_cdf",
                  "stats.stats_t_isf",
                  "stats.stats_t_pdf",
                  "stats.stats_t_ppf",
                  "stats.stats_t_rvs",
                  "stats.stats_t_sf",
                  "stats.stats_uniform_cdf",
                  "stats.stats_uniform_isf",
                  "stats.stats_uniform_pdf",
                  "stats.stats_uniform_ppf",
                  "stats.stats_uniform_rvs",
                  "stats.stats_uniform_sf",
                  "stats.stats_weibull_cdf",
                  "stats.stats_weibull_isf",
                  "stats.stats_weibull_pdf",
                  "stats.stats_weibull_ppf",
                  "stats.stats_weibull_rvs",
                  "stats.stats_weibull_sf",
                  "stats.stats_cdf",
                  "stats.stats_isf",
                  "stats.stats_pdf",
                  "stats.stats_pmf",
                  "stats.stats_ppf",
                  "stats.stats_rvs",
                  "stats.stats_sf",
                  "stats.stats_chisquare",
                  "stats.stats_fit",
                  "stats.stats_kstest1",
                  "stats.stats_kstest2",
                  "stats.stats_linregress",
                  "stats.stats_pearsonr",
                  "stats.stats_percentilesofscores",
                  "stats.stats_probplotvals",
                  "stats.stats_stats",
                  "stats.stats_scoresatpercentiles",
                  "stats.stats_spearmanr",
                  "stats.stats_ttest1",
                  "stats.stats_ttest2ind",
                  "stats.stats_ttest2rel",
                  "stats.stats_zscore",
                  "stats.stats_helper",
                  ))
    except ImportError:
        # if not my_quiet:
        #     print >>sys.stderr, "    WARNING: Unable to import scipy;\n" \
        #                         "             most stats_* Ferret functions will not be added."
        pass

    # shapefile_* functions
    try:
        import shapefile
        std_pyefs.extend((
                  "fershp.shapefile_readxy",
                  "fershp.shapefile_readxyval",
                  "fershp.shapefile_readxyz",
                  "fershp.shapefile_readxyzval",
                  "fershp.shapefile_writeval",
                  "fershp.shapefile_writexyval",
                  "fershp.shapefile_writexyzval",
                  ))
    except ImportError:
        # if not my_quiet:
        #     print >>sys.stderr, "    WARNING: Unable to import shapefile;\n" \
        #                         "             shapefile_* Ferret functions will not be added."
        pass

    # regrid functions
    try:
        import ESMP
        std_pyefs.extend((
                  "regrid.curv2rect",
                  "regrid.curv3srect",
                  ))
    except ImportError:
        # if not my_quiet:
        #     print >>sys.stderr, "    WARNING: Unable to import ESMP;\n" \
        #                         "             curv2rect* Ferret functions will not be added.\n" \
        #                         "             Use curv_to_rect* functions instead"
        pass

    # start ferret without journaling
    start(memsize=my_memsize, journal=False, verify=my_verify,
          restrict=my_restrict, server=my_server, metaname=my_metaname,
          transparent=my_transparent, unmapped=my_unmapped,
          pngonly=my_pngonly, quiet=my_quiet, linebuffer=my_linebuffer)

    # define all the Ferret standard Python external functions
    for fname in std_pyefs:
        result = run("define pyfunc pyferret.%s" % fname)

    # run the ${HOME}/.ferret script if it exists and not restricted environment
    if not my_restrict:
        home_val = os.getenv('HOME')
        if home_val:
            init_script = os.path.join(home_val, '.ferret')
            if os.path.exists(init_script):
                try:
                    result = run('go "%s"; exit /topy' % init_script)
                except:
                    print(" **Error: exception raised in runnning script %s" % init_script, file=sys.stderr)
                    result = run('exit /program')
                    # should not get here
                    raise SystemExit

    # if a command-line script is given, run the script and exit completely
    if script != None:
        # put double quotes around every script argument
        script_line = '"' + '" "'.join(script) + '"'
        try:
            result = run('go %s; exit /program' % script_line)
        except:
            print(" **Error: exception raised in running script %s" % script_line, file=sys.stderr)
        # If exception or if returned early, force shutdown
        result = run('exit /program')
        # should not get here
        raise SystemExit

    # if journaling desired, now turn on journaling
    if my_journal:
        result = run("set mode journal")

    # if they don't want to enter ferret, return the success value from run
    if not my_enterferret:
        return (libpyferret.FERR_OK, '')

    # otherwise, go into Ferret command-line processing until "exit /topy" or "exit /program"
    result = run()
    return result


def start(memsize=25, journal=True, verify=False, restrict=False,
          server=False, metaname=None, transparent=False,
          unmapped=False, pngonly=False, quiet=False, linebuffer=False):
    """
    Initializes Ferret.  This allocates the initial amount of memory
    for Ferret (from Python-managed memory), opens the journal file,
    if requested, and sets Ferret's verify mode.
    If restrict is True, some Ferret commands will not be available
    (to provide a secured session).  Once restrict is set, it cannot
    be unset.
    If metaname is not empty this value is used as the initial filename
    for automatic output of graphics, and the graphics viewer will not
    be displayed.
    If server is True, Ferret will be run in server mode.
    If unmapped is True, the graphics viewer will not be displayed.
    If pngonly is True, only raster images can be saved and the
    graphics viewer will not be displayed.
    If quiet is True, the Ferret start-up header is not displayed.
    If linebuffer is True, stdout and stderr are set user line
    buffering.  This cannot be reset once set.
    This routine does NOT run any user initialization scripts.

    Arguments:
        memsize:     the size, in mega (10^6) floats (where 1 float
                     = 8 bytes) to allocate for Ferret's memory block
        journal:     turn on Ferret's journal mode?
        verify:      turn on Ferret's verify mode?
        restrict:    restrict Ferret's capabilities?
        server:      put Ferret in server mode?
        metaname:    filename for Ferret graphics; can be None or empty
        transparent: autosave (e.g., on exit) image files with a
                     transparent background?
        unmapped:    hide the graphics viewer?
        pngonly:     only create raster (PNG) images?
        quiet:       do not display the Ferret start-up header?
        linebuffer:  print each line of output or error messages as soon as
                     a full line is written?  Useful when redirecting these
                     messages to a file.
                     Note:
                       the enviroment variable GFORTRAN_UNBUFFERED_PRECONNECTED
                       needs to be set to 1 in order to unbuffer the Fortran
                       units for output and error messages
    Returns:
        True is successful
        False if Ferret has already been started
    Raises:
        ValueError if memsize if not a positive number
        MemoryError if unable to allocate the needed memory
        IOError if unable to open the journal file
    """
    # check memsize
    try:
        flt_memsize = float(memsize)
        if flt_memsize <= 0.0:
            raise ValueError
    except:
        raise ValueError("memsize must be a positive number")
    # check metaname
    if metaname == None:
        str_metaname = ""
    elif not isinstance(metaname, str):
        raise ValueError("metaname must either be None or a string")
    elif metaname.isspace():
        str_metaname = ""
    else:
        str_metaname = metaname

    # Get the known viewer bindings
    knownengines = graphbind.knownPyFerretEngines()
    # Add PViewerPQPyFerretBindings, as "PipedViewerPQ" to the known bindings
    if not ("PipedViewerPQ" in knownengines):
        graphbind.addPyFerretBindings("PipedViewerPQ",
                  pipedviewer.pyferretbindings.PViewerPQPyFerretBindings)
    # Add PImagerPQPyFerretBindings, as "PipedImagerPQ" to the known bindings
    if not ("PipedImagerPQ" in knownengines):
        graphbind.addPyFerretBindings("PipedImagerPQ",
                  pipedviewer.pyferretbindings.PImagerPQPyFerretBindings)

    # the actual call to ferret's start
    success = libpyferret._start(flt_memsize, bool(journal), bool(verify),
                                 bool(restrict), bool(server), str_metaname,
                                 bool(transparent), bool(unmapped),
                                 bool(pngonly), bool(quiet), bool(linebuffer))
    if success:
        # register the libpyferret._quit function with atexit to ensure
        # open viewer windows do not hang a Python shutdown
        atexit.register(libpyferret._quit)

        # Use tab completion for readline
        if 'libedit' in readline.__doc__:
            # Mac OS X actually uses libedit for system readline
            readline.parse_and_bind('bind ^I rl_complete');
        else:
            # GNU readline
            readline.parse_and_bind('tab: complete');

        # Execute the $PYTHONSTARTUP file, if it exists and -secure not given
        if not restrict:
            try:
                startfilename = os.getenv('PYTHONSTARTUP', '')
                if startfilename:
                    if sys.version_info[0] > 2:
                        exec(compile(open(startfilename).read(), startfilename, 'exec'));
                    else:
                        execfile(startfilename)
            except Exception:
                pass

    return success


def resize(memsize):
    """
    Resets the the amount of memory allocated for Ferret from Python-managed memory.

    Arguments:
        memsize: the new size, in mega (10^) floats (where a "float" is 8 bytes),
                 for Ferret's memory block
    Returns:
        True if successful - Ferret has the new amount of memory
        False if unsuccessful - Ferret has the previous amount of memory
    Raises:
        ValueError if memsize if not a positive number
        MemoryError if Ferret has not been started or has been stopped
    """
    # check memsize
    try:
        flt_memsize = float(memsize)
        if flt_memsize <= 0.0:
            raise ValueError
    except:
        raise ValueError("memsize must be a positive number")
    # the actual call - now just a ferret command handled internally
    return libpyferret._run("SET MEMORY /SIZE=%d" % flt_memsize)


def run(command=None):
    """
    Runs a Ferret command just as if entering a command at the Ferret prompt.

    If the command is not given, is None, or is a blank string, Ferret will
    prompt you for commands until "EXIT /TOPYTHON" is given.  In this case,
    the return tuple will be for the last error, if any, that occurred in
    the sequence of commands given to Ferret.

    Arguments:
        command: the Ferret command to be executed.
    Returns:
        (err_int, err_string)
            err_int: one of the FERR_* data values (FERR_OK if there are no errors)
            err_string: error or warning message (can be empty)
        Error messages normally start with "**ERROR"
        Warning messages normally start with "*** NOTE:"
    Raises:
        ValueError if command is neither None nor a String
        MemoryError if Ferret has not been started or has been stopped
    """
    # check command
    if command == None:
        str_command = ""
    elif not isinstance(command, str):
        raise ValueError("command must either be None or a string")
    elif command.isspace():
        str_command = ""
    else:
        str_command = command
    # if going into Ferret-command mode,
    # use the filename completer for readline name completion
    if str_command == "":
        old_completer = readline.get_completer()
        filename_completer_obj = filenamecompleter.FilenameCompleter()
        readline.set_completer(filename_completer_obj.complete)
    # the actual Ferret function call
    retval = libpyferret._run(str_command)
    # return to the original readline completer
    if str_command == "":
        readline.set_completer(old_completer)
        del filename_completer_obj
    return retval


def stop():
    """
    Runs a series of Ferret commands to return Ferret to
    its default state, then shuts down and releases all
    memory used by Ferret.  After calling this function do
    not call any Ferret functions except start, which will
    restart Ferret and re-enable the other functions.

    Returns:
        False if Ferret has not been started or has already been stopped
        True otherwise
    """
    # If it had been started, shut down ESMP and delete the log file
    try:
        regrid.ESMPControl().stopESMP(True)
    except Exception:
        pass

    # clear "the" anonymous dataset
    pyferret.anondset.close()

    # Continue with Ferret shutdown
    return libpyferret._stop()


def _readline(myprompt):
    """
    Prompts the user for input and returns the line read.

    Used for reading commands in the Ferret command loop.
    Just uses the built-in function raw_input.  Since the
    readline module was imported, readline features are
    provided.

    Arguments:
        myprompt - prompt string to use
    Returns:
        the string read in, or None if an Exception occurs
    """
    try:
        if sys.version_info[0] > 2:
            if myprompt:
                myline = input(myprompt)
            else:
                myline = input()
        else:
            if myprompt:
                myline = raw_input(myprompt)
            else:
                myline = raw_input()
    except Exception:
        myline = None

    return myline

