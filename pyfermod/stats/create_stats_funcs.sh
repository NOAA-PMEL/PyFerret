#! /bin/env python
#

from __future__ import print_function

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
        print(line, end=" ", file=scriptfile)
    templatefile.close()
    scriptfile.close()

def create_all_scripts():
    """
    Creates the valid stats_<distribname>_<funcname>.py scripts, if they do
    not already exist, for all the supported distributions and functions.
    """
    # List of supported distributions
    distnamelist = pyferret.stats.getdistname(None)
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
    for nametuple in distnamelist:
        distname = nametuple[0]
        distlongname = nametuple[1]
        for (funcname, funcreturn) in funcnamelist:
            try:
                # Verify the function exists for the distribution.
                # This raises an AttributeError is it does not.
                statsfunc = eval("scipy.stats.%s.%s" % (distname,funcname))
                if distname == "weibull_min":
                    scriptname = "stats_weibull_%s.py" % funcname
                else:
                    scriptname = "stats_%s_%s.py" % (distname, funcname)
                # Verify the script does not already exist.
                if not os.path.exists(scriptname):
                    create_script(scriptname, distname, distlongname,
                                              funcname, funcreturn)
            except AttributeError:
                # function does not exist for the distribution - skip
                pass

if __name__ == "__main__":
    # create all scripts from the 'stats_template' file
    if not os.path.exists("stats_template"):
        raise ValueError("The file 'stats_template' does not exist")
    create_all_scripts()

