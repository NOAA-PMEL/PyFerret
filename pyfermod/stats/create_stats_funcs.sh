#! /bin/env python2.6
#

import os.path
import scipy.stats
import pyferret.stats

def create_script(scriptname, distribname, distriblongname, funcname, funcreturn):
    """
    Creates scriptname from 'stats_template' using remaining arguments
    for replacement strings in the template file.
    """
    templatefile = file("stats_template", "r")
    scriptfile = file(scriptname, "w")
    for line in templatefile:
        line = line.replace("<distribname>", distribname)
        line = line.replace("<distriblongname>", distriblongname)
        line = line.replace("<funcname>", funcname)
        line = line.replace("<funcreturn>", funcreturn)
        print >>scriptfile, line,
    templatefile.close()
    scriptfile.close()

def create_all_scripts():
    """
    Creates the valid stats_<distribname>_<funcname>.py scripts, if they do
    not already exist, for all the supported distributions and functions.
    """
    # List of supported distributions
    distribinfo = pyferret.stats.getdistrib(None, None)
    distribnamelist = [ ]
    for j in xrange(len(distribinfo)):
        # Get the long name from the description resembling 
        # "Beta(ALPHA, BETA)" or "F or Fisher(DFN, DFD)" (use first long name)
        distriblongname = distribinfo[j][1].split()[0].split('(')[0]
        distribnamelist.append( ( distribinfo[j][0], distriblongname, ) )
    # List of supported functions
    funcnamelist  = [ ( "cdf", "cumulative density function values", ),
                      ( "isf", "inverse survival function values", ),
                      ( "pdf", "probability distribution function values", ),
                      ( "pmf", "probability mass function values", ),
                      ( "ppf", "percent point function values", ),
                      ( "sf",  "survival function values", ),
                      ( "rvs", "random variates", ), ]
    # Loop of the list of distributions and functions, creating the script
    # if it does not exist and if the function exists for that distribution.
    for (distribname, distriblongname) in distribnamelist:
        for (funcname, funcreturn) in funcnamelist:
            try:
                # Verify the function exists for the distribution.
                # This raises an AttributeError is it does not.
                statsfunc = eval("scipy.stats.%s.%s" % (distribname,funcname))
                if distribname == "weibull_min":
                    scriptname = "stats_weibull_%s.py" % funcname
                else:
                    scriptname = "stats_%s_%s.py" % (distribname, funcname)
                # Verify the script does not already exist.
                if not os.path.exists(scriptname):
                    create_script(scriptname, distribname, distriblongname,
                                              funcname, funcreturn)
            except AttributeError:
                # function does not exist for the distribution - skip
                pass

if __name__ == "__main__":
    # create all scripts from the 'stats_template' file
    if not os.path.exists("stats_template"):
        raise ValueError("The file 'stats_template' does not exist")
    create_all_scripts()

