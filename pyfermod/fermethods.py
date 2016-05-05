"""
Convenience methods for executing common Ferret commands.
These method generate appropriate Ferret command strings and then
execute them using the pyferret.run method.
"""

import numbers
import pyferret


def setwindow(num=1, plotasp=None, axisasp=None, color=None, logo=None):
    """
    Assigns the plot window to use for subsequent plotting commands.
    Also provides assignment of common window plots.  
    Note that plotasp and axisasp cannot both be given.
        num (int): window number 1-8 to use for plots
        plotasp (float): aspect ratio (Y/X) for the plot window;
            if not given, the current ratio is unchanged; 
            the default ratio on start-up is 0.86
        axisasp (float): aspect ratio (Y/X) for the plot axes;
            if not given, the current ratio is unchanged; 
            the default ratio on start-up is 0.75
        color (string, tuple of int): background color for the plot;
            can be one of the color names 'black', 'blue', 'green', 
            'lightblue', 'purple', or 'red', or 
            a tuple of [0-100] int values giving RGB or RGBA values
        logo (boolean): include the Ferret logo in the plot?
            if not given, the current value is unchanged.
    Raises a ValueError if a problem occurs.
    """
    # create and execute the SET WINDOW command
    cmdstr = 'SET WINDOW'
    if plotasp and axisasp:
        raise ValueError('only one of plotasp and axisasp can be given')
    if plotasp:
        if (not isinstance(plotasp, numbers.Real)) or (plotasp <= 0):
            raise ValueError('given plotasp %s is not a positive floating-point value' % str(plotasp))
        cmdstr += '/ASPECT=' + str(plotasp)
    if axisasp:
        if (not isinstance(axisasp, numbers.Real)) or (axisasp <= 0):
            raise ValueError('given axisasp %s is not a positive floating-point value' % str(axisasp))
        cmdstr += '/ASPECT=' + str(axisasp) + ':AXIS'
    if color:
        cmdstr += '/COLOR=' + str(color)
    if (not isinstance(num, numbers.Integral)) or (num <= 0) or (num > 8):
        raise ValueError('window number %s is not a integer in [1,8]' % str(num))
    cmdstr += ' ' + str(num)
    (errval, errmsg) = pyferret.run(cmdstr)
    if errval != pyferret.FERR_OK:
        raise ValueError('Problems executing Ferret command %s: %s' % (cmdstr, errmsg))
    # create and execute the mode logo command if logo is given
    if logo is not None:
        if logo:
            cmdstr = 'SET MODE LOGO'
        else:
            cmdstr = 'CANCEL MODE LOGO'
        (errval, errmsg) = pyferret.run(cmdstr)
        if errval != pyferret.FERR_OK:
            raise ValueError('Problems executing Ferret command %s: %s' % (cmdstr, errmsg))


def settextstyle(font='', color='', bold = False, italic=False):
    """
    Sets the text style for any text in plots generated after this command
    using the Ferret SET TEXT command.
        font (string): name of the font to use; if empty, 'Arial' is used.
        color (string): color name, RGB tuple, or RGBA tuple describing the 
            color of the text.  The R,G,B, and A components are integer
            percentages; thus values in [0,100]
        bold (bool): use bold font?
        italic (bool): use italic font?
    """
    # First run CANCEL TEXT to clear any /BOLD and /ITALIC
    (errval, errmsg) = pyferret.run('CANCEL TEXT/ALL')
    if errval != pyferret.FERR_OK:
        raise ValueError('problems resetting text style to default: %s' % errmsg)
    # Now run SET TEXT with the appropriate qualifiers
    cmdstr = 'SET TEXT'
    if font:
        cmdstr += '/FONT='
        cmdstr += font
    else:
        cmdstr += '/FONT=Arial'
    if color:
        cmdstr += '/COLOR=' + str(color)
    if bold:
        cmdstr += '/BOLD'
    if italic:
        cmdstr += '/ITALIC'
    (errval, errmsg) = pyferret.run(cmdstr)
    if errval != pyferret.FERR_OK:
        raise ValueError('problems setting text style (%s): %s' % (cmdstr, errmsg))


def showdata(brief=True, qual=''):
    """
    Show the Ferret information about all datasets currently open in Ferret.  
    This uses the Ferret SHOW DATA command to create and display the information.
        brief (boolean): if True (default), a brief report is shown;
            otherwise a full report is shown.
        qual (string): Ferret qualifiers to add to the SHOW DATA command
    """
    if not isinstance(qual, str):
        raise ValueError('qual (Ferret qualifiers) must be a string')
    cmdstr = 'SHOW DATA'
    if not brief:
        cmdstr += '/FULL'
    if qual:
        cmdstr += qual
    (errval, errmsg) = pyferret.run(cmdstr)
    if errval != pyferret.FERR_OK:
        raise ValueError('Ferret command "%s" failed: %s' % (cmdstr, errmsg))


def contourplot(fvar, over=False, qual=''):
    """
    Create a contour plot of the specified Ferret variable using the Ferret CONTOUR command.
    Using the fill method to generated a color-filled contour plot.
    The variable needs to be 2D (or qualifiers need to be added to specify a 2D slice).
        fvar (string or FerVar): Ferret variable to plot
        over (bool): overlay on an existing plot?
        qual (string): qualifiers to add to the Ferret SHADE command
    """
    if not isinstance(qual, str):
        raise ValueError('qual (Ferret qualifiers) must be a string')
    if isinstance(fvar, str):
        plotvar = fvar
    elif isinstance(fvar, pyferret.FerVar):
        plotvar = fvar._definition
    else:
        raise ValueError('fvar (Ferret variable to plot) must be a string or FerVar')
    cmdstr = 'CONTOUR'
    if over:
        cmdstr += '/OVER'
    if qual:
        cmdstr += qual
    cmdstr += ' '
    cmdstr += plotvar
    (errval, errmsg) = pyferret.run(cmdstr)
    if errval != pyferret.FERR_OK:
        raise ValueError('Ferret shade command (%s) failed: %s' % (cmdstr, errmsg))


def fillplot(fvar, line=False, over=False, qual=''):
    """
    Create a color-filled contour plot of the specified Ferret variable using the Ferret 
    FILL command.  Drawing of the contour lines themselves is optional.
    The variable needs to be 2D (or qualifiers need to be added to specify a 2D slice).
        fvar (string or FerVar): Ferret variable to plot
        line (bool): draw the contour lines?
        over (bool): overlay on an existing plot?
        qual (string): qualifiers to add to the Ferret SHADE command
    """
    if not isinstance(qual, str):
        raise ValueError('qual (Ferret qualifiers) must be a string')
    if isinstance(fvar, str):
        plotvar = fvar
    elif isinstance(fvar, pyferret.FerVar):
        plotvar = fvar._definition
    else:
        raise ValueError('fvar (Ferret variable to plot) must be a string or FerVar')
    cmdstr = 'FILL'
    if line:
        cmdstr += '/LINE'
    if over:
        cmdstr += '/OVER'
    if qual:
        cmdstr += qual
    cmdstr += ' '
    cmdstr += plotvar
    (errval, errmsg) = pyferret.run(cmdstr)
    if errval != pyferret.FERR_OK:
        raise ValueError('Ferret shade command (%s) failed: %s' % (cmdstr, errmsg))


def shadeplot(fvar, over=False, qual=''):
    """
    Create a colored plot of the specified Ferret variable using the Ferret SHADE command.
    (Plot coloring grid cells based on the variable value in that cell.)
    The variable needs to be 2D (or qualifiers need to be added to specify a 2D slice).
        fvar (string or FerVar): Ferret variable to plot
        over (bool): overlay on an existing plot?
        qual (string): qualifiers to add to the Ferret SHADE command
    """
    if not isinstance(qual, str):
        raise ValueError('qual (Ferret qualifiers) must be a string')
    if isinstance(fvar, str):
        plotvar = fvar
    elif isinstance(fvar, pyferret.FerVar):
        plotvar = fvar._definition
    else:
        raise ValueError('fvar (Ferret variable to plot) must be a string or FerVar')
    cmdstr = 'SHADE'
    if over:
        cmdstr += '/OVER'
    if qual:
        cmdstr += qual
    cmdstr += ' '
    cmdstr += plotvar
    (errval, errmsg) = pyferret.run(cmdstr)
    if errval != pyferret.FERR_OK:
        raise ValueError('Ferret shade command (%s) failed: %s' % (cmdstr, errmsg))


def shadeland():
    """
    Shades land as gray figures to the current longitude-latitude plot.
    """
    cmdstr = 'GO FLAND'
    (errval, errmsg) = pyferret.run(cmdstr)
    if errval != pyferret.FERR_OK:
        raise ValueError('Ferret script command (%s) failed: %s' % (cmdstr, errmsg))


def shadewater():
    """
    Shades oceans as gray figures to the current longitude-latitude plot.
    """
    cmdstr = 'GO FOCEAN'
    (errval, errmsg) = pyferret.run(cmdstr)
    if errval != pyferret.FERR_OK:
        raise ValueError('Ferret script command (%s) failed: %s' % (cmdstr, errmsg))


def lineplot(fvar, vs=None, color=None, thick=1.0, dash=None, title=None, over=False, nolab=False, qual=''):
    """
    Create a line plot of the given value, or the given value versus another value (if vs is given),
    possibly colored by another value (if color is a FerVar).
    To create a line plot with symbols, use the symbolplot command with the line option set to True.
        fvar (string or FerVar): Ferret variable to plot
        vs  (string or FerVar): if given, plot the above variable versus this variables
        color: line color or variable used to determine line color; if
            None: Ferret default color used,
            color name (string): name of color to use,
            color tuple (3 or 4-tupe of [0,100] int values): RGB or RGBA of color to use,
            FerVar or variable name string: color according to the value of this variable
                Note: color name strings are limited to (case insensitive) 
                      'black', 'red', 'green', 'blue', 'lightblue', 'purple'
                      other strings are assumed to be variable names
        thick (float): line thickness scaling factor
        dash (4-tuple of float): draws the line as a dashed line where the four values 
             are the first drawn stroke length, first undrawn stroke length,
             second drawn stroke length, second undrawn stroke length of two dashes
        title (string): title for the plot; if not given,  Ferret's default title is used
        over (bool): overlay onto an existing plot
        nolab (bool): if true, suppress all plot labels
        qual (string): qualifiers to add to the Ferret PLOT/LINE command
    """
    if not isinstance(qual, str):
        raise ValueError('qual (Ferret qualifiers) must be a string')
    if isinstance(fvar, str):
        plotvar = fvar
    elif isinstance(fvar, pyferret.FerVar):
        plotvar = fvar._definition
    else:
        raise ValueError('fvar (Ferret variable to plot) must be a string or FerVar')
    cmdstr = 'PLOT/LINE'
    if vs is not None:
        cmdstr += '/VS'
        plotvar += ','
        if isinstance(vs, str):
            plotvar += vs
        elif isinstance(vs, pyferret.FerVar):
            plotvar += vs._definition
        else:
            raise ValueError('vs (second Ferret variable to plot) must be a string or FerVar')
    if color is not None:
        if isinstance(color, tuple):
           cmdstr += '/COLOR=' + str(color)
        elif isinstance(color, pyferret.FerVar):
           cmdstr += '/RIBBON'
           plotvar += ',' + color._definition
        elif isinstance(color, str):
            if color.upper() in ('BLACK','RED','GREEN','BLUE','LIGHTBLUE','PURPLE'):
               cmdstr += '/COLOR=' + color
            else:
               cmdstr += '/RIBBON'
               plotvar += ',' + color
        else:
            raise ValueError('color must be a tuple, string, or FerVar')
    if thick is not None:
        if (not isinstance(thick, numbers.Real)) or (thick <= 0):
            raise ValueError('thick must be a positive floating-point value')
        cmdstr += '/THICKNESS=' + str(thick)
    if dash is not None:
        if (not isinstance(dash, tuple)) or (len(dash) != 4):
            raise ValueError('dash must be a tuple of four floats');
        cmdstr += '/DASH=' + str(dash)
    if title is not None:
       if not isinstance(title, str):
           raise ValueError('title must be a string')
       cmdstr += '/TITLE="' + title + '"'
    if over:
        cmdstr += '/OVER'
    if nolab:
        cmdstr += '/NOLABEL'
    if qual:
        cmdstr += qual
    cmdstr += ' '
    cmdstr += plotvar
    (errval, errmsg) = pyferret.run(cmdstr)
    if errval != pyferret.FERR_OK:
        raise ValueError('Ferret plot command (%s) failed: %s' % (cmdstr, errmsg))
    

def saveplot(name, fmt='', xpix=None, ypix=None, xinch=None, yinch=None, qual=''):
    """
    Save the current plot.  If format is not given,
    the format is guessed from the filename extension.
        name (string): name of the file to contain the plot
        fmt (string): format of the plot file
        xpix (int): number of pixels in width of the saved raster (eg, PNG) plot
        ypix (int): number of pixels in the height of the saved raster (eg, PNG) plot
        xinch (float): inch width of the saved vector (eg, PDF) plot
        yinch (float): inch height of the save vector (eg, PDF) plot
        qual (string): qualifiers to add to the Ferret FRAME command
    """
    if not isinstance(name, str):
        raise ValueError('name (plot file name) must be a string')
    cmdstr = 'FRAME/FILE="%s"' % name
    if not isinstance(fmt, str):
        raise ValueError('fmt (plot file format) must be a string')
    if fmt:
        cmdstr += '/FORMAT=%s' % fmt
    if xpix is not None:
        if (not isinstance(xpix, int)) or (xpix <= 0):
            raise ValueError('xpix must be a positive integer')
        cmdstr += '/XPIX=' + str(xpix)
    if ypix is not None:
        if (not isinstance(ypix, int)) or (ypix <= 0):
            raise ValueError('ypix must be a positive integer')
        cmdstr += '/YPIX=' + str(ypix)
    if (xpix is not None) and (ypix is not None):
        raise ValueError('xpix and ypix cannot both be given')
    if xinch is not None:
        if (not isinstance(xinch, numbers.Real)) or (xinch <= 0.0):
            raise ValueError('xinch must be a positive number')
        cmdstr += '/XINCH=' + str(xinch)
    if yinch is not None:
        if (not isinstance(yinch, numbers.Real)) or (yinch <= 0.0):
            raise ValueError('yinch must be a positive number')
        cmdstr += '/YINCH=' + str(yinch)
    if (xinch is not None) and (yinch is not None):
        raise ValueError('xinch and yinch cannot both be given')
    if not isinstance(qual, str):
        raise ValueError('qual (Ferret qualifiers) must be a string')
    if qual:
        cmdstr += qual
    (errval, errmsg) = pyferret.run(cmdstr)
    if errval != pyferret.FERR_OK:
        raise ValueError('Ferret frame command (%s) failed: %s' % (cmdstr, errmsg))

