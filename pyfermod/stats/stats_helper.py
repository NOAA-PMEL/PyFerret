"""
Returns an array of strings describing
the parameters for a probability distribution.
"""

from __future__ import print_function

import numpy
import pyferret
import pyferret.stats


def ferret_init(id):
    """
    Initialization for the stats_helper Ferret PyEF
    """
    axes_values = [ pyferret.AXIS_DOES_NOT_EXIST ] * pyferret.MAX_FERRET_NDIM
    axes_values[0] = pyferret.AXIS_ABSTRACT
    false_influences = [ False ] * pyferret.MAX_FERRET_NDIM
    retdict = { "numargs": 1,
                "descript": "Help on probability distribution names or parameters",
                "restype": pyferret.STRING_ARRAY,
                "resstrlen": 256,
                "axes": axes_values,
                "argnames": ( "PDNAME", ),
                "argdescripts": ( "Name of a probability distribution (or blank for all)", ),
                "argtypes": ( pyferret.STRING_ONEVAL, ),
                "influences": ( false_influences, ),
              }
    return retdict


def ferret_result_limits(id):
    """
    Return the limits of the abstract X axis
    """
    # Get the maximum number of string pairs that will be return -
    # either distribution short names and long names with parameter names
    # or parameter names and descriptions; the number of distributions far
    # exceeds the number of parameters for any distribution
    distnamelist = pyferret.stats.getdistname(None)
    # One intro string and at least one empty line at the end
    max_num_string_pairs = len(distnamelist) + 2

    axis_lims = [ None ] * pyferret.MAX_FERRET_NDIM
    axis_lims[0] = ( 1, max_num_string_pairs )
    return axis_lims


def ferret_compute(id, result, resbdf, inputs, inpbdfs):
    """
    Assigns result with parameter desciption strings for the
    probability distribution indicated by inputs[0] (a string).
    """
    max_num_string_pairs = ferret_result_limits(id)[0][1]
    distribname = inputs[0].strip()
    if distribname:
        # list the parameters with descriptions for this distribution
        distname = pyferret.stats.getdistname(distribname)
        paramlist = pyferret.stats.getdistparams(distname, None)
        result[0] = "Parameters of probability distribution %s" % distribname
        for k in range(len(paramlist)):
            result[k+1] = "(%d) %s: %s" % (k+1, paramlist[k][0], paramlist[k][1])
        for k in range(len(paramlist)+1, max_num_string_pairs):
            result[k] = ""
    else:
        # list the all the distributions with parameter list arguments
        distnamelist = pyferret.stats.getdistname(None)
        result[0] = "Supported probability distributions"
        for k in range(len(distnamelist)):
            # create the parameter argument string
            paramlist = pyferret.stats.getdistparams(distnamelist[k][0], None)
            numparams = len(paramlist)
            if numparams == 1:
                paramstr = "(%s)" % paramlist[0][0]
            elif numparams == 2:
                paramstr = "(%s,%s)" % (paramlist[0][0], paramlist[1][0])
            elif numparams == 3:
                paramstr = "(%s,%s,%s)" % (paramlist[0][0], paramlist[1][0],
                                           paramlist[2][0])
            elif numparams == 4:
                paramstr = "(%s,%s,%s)" % (paramlist[0][0], paramlist[1][0],
                                           paramlist[2][0], paramlist[3][0])
            else:
                raise ValueError("Unexpected number of parameters: %s" % numparams)
            # create the help string
            numnames = len(distnamelist[k])
            if numnames == 2:
                result[k+1] = "   %s: %s%s" % \
                    (distnamelist[k][0], distnamelist[k][1], paramstr)
            elif numnames == 3:
                result[k+1] = "   %s: %s or %s%s" % \
                    (distnamelist[k][0], distnamelist[k][1],
                     distnamelist[k][2], paramstr)
            elif numnames == 4:
                result[k+1] = "   %s: %s, %s, or %s%s" % \
                    (distnamelist[k][0], distnamelist[k][1],
                     distnamelist[k][2], distnamelist[k][3], paramstr)
            else:
                raise ValueError("Unexpected number of names: %s" % numnames)
        for k in range(len(distnamelist)+1, max_num_string_pairs):
            result[k] = ""


def print_help():
    """
    Print the stats_helper messages to console (using print in Python).
    This is also designed to test the other functions of this module.
    """
    info = ferret_init(0)
    stype = "S%d" % info["resstrlen"]
    sizetuple = ferret_result_limits(0)
    max_strings = sizetuple[0][1]
    distrib_array = numpy.empty((max_strings,), \
                                dtype=numpy.dtype(stype), order='F')
    # Some initialization for testing
    for k in range(max_strings):
        distrib_array[k] = "Unassigned %d" % k
    # Get the list of distributions (string of spaces for testing)
    pfname = "    "
    ferret_compute(0, distrib_array, None, ( pfname, ), None)
    # Print all the distribution short and long names, and the empty line at the end
    for j in range(max_strings):
        print(distrib_array[j])
    # Now go through all the distributions
    params_array = numpy.empty((max_strings,), \
                               dtype=numpy.dtype('S128'), order='F')
    # Skip intro line, and empty line at end
    for j in range(1, max_strings-1):
        # Some initialization for testing
        for k in range(max_strings):
            params_array[k] = "Unassigned %d" % k
        # Use the distribution long name (second word, remove the param list)
        pfname = distrib_array[j].split()[1].split('(')[0]
        ferret_compute(0, params_array, None, ( pfname, ), None)
        for k in range(max_strings):
            print(params_array[k])
            # Stop after printing an empty line
            if not params_array[k]:
                break;


# For testing this module at the command line
if __name__ == "__main__":
    print_help()

