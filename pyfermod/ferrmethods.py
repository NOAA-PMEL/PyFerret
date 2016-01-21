"""
Convenience methods for executing common Ferret commands.
These method generate appropriate Ferret command strings and then
execute them using the pyferret.run method.
"""

import numbers
import pyferret


def setwindow(num, plotasp=None, axisasp=None, color=None, logo=None):
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
    if logo != None:
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


def showdatasets(brief=True, qual=''):
    """
    Show the Ferret information about all dataset currently open in Ferret.  
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


def contour(fvar, over=False, qual=''):
    """
    Create a contour plot of the specified Ferret variable using the Ferret CONTOUR command.
    Using the fill method to generated a color-filled contour plot.
    The variable needs to be 2D (or qualifiers need to be added to specify a 2D slice).
        fvar (string or FerrVar): Ferret variable to plot
        over (bool): overlay on an existing plot?
        qual (string): qualifiers to add to the Ferret SHADE command
    """
    if not isinstance(qual, str):
        raise ValueError('qual (Ferret qualifiers) must be a string')
    if isinstance(fvar, str):
        plotvar = fvar
    elif isinstance(fvar, pyferret.FerrVar):
        plotvar = fvar._definition
    else:
        raise ValueError('fvar (Ferret variable to plot) must be a string or FerrVar')
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


def fill(fvar, line=False, over=False, qual=''):
    """
    Create a color-filled contour plot of the specified Ferret variable using the Ferret 
    FILL command.  Drawing of the contour lines themselves is optional.
    The variable needs to be 2D (or qualifiers need to be added to specify a 2D slice).
        fvar (string or FerrVar): Ferret variable to plot
        line (bool): draw the contour lines?
        over (bool): overlay on an existing plot?
        qual (string): qualifiers to add to the Ferret SHADE command
    """
    if not isinstance(qual, str):
        raise ValueError('qual (Ferret qualifiers) must be a string')
    if isinstance(fvar, str):
        plotvar = fvar
    elif isinstance(fvar, pyferret.FerrVar):
        plotvar = fvar._definition
    else:
        raise ValueError('fvar (Ferret variable to plot) must be a string or FerrVar')
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


def shade(fvar, over=False, qual=''):
    """
    Create a colored plot of the specified Ferret variable using the Ferret SHADE command.
    (Plot coloring grid cells based on the variable value in that cell.)
    The variable needs to be 2D (or qualifiers need to be added to specify a 2D slice).
        fvar (string or FerrVar): Ferret variable to plot
        over (bool): overlay on an existing plot?
        qual (string): qualifiers to add to the Ferret SHADE command
    """
    if not isinstance(qual, str):
        raise ValueError('qual (Ferret qualifiers) must be a string')
    if isinstance(fvar, str):
        plotvar = fvar
    elif isinstance(fvar, pyferret.FerrVar):
        plotvar = fvar._definition
    else:
        raise ValueError('fvar (Ferret variable to plot) must be a string or FerrVar')
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

def saveplot(name, frmt='', qual=''):
    """
    Save the current plot.  If format is not given,
    the format is guessed from the filename extension.
        name (string): name of the file to contain the plot
        frmt (string): format of the plot file
        qual (string): qualifiers to add to the Ferret FRAME command
    """
    if not isinstance(name, str):
        raise ValueError('name (plot file name) must be a string')
    if not isinstance(frmt, str):
        raise ValueError('frmt (plot file format) must be a string')
    if not isinstance(qual, str):
        raise ValueError('qual (Ferret qualifiers) must be a string')
    cmdstr = 'FRAME/FILE="%s"' % name
    if frmt:
        cmdstr += '/FORMAT=%s' % frmt
    if qual:
        cmdstr += qual
    (errval, errmsg) = pyferret.run(cmdstr)
    if errval != pyferret.FERR_OK:
        raise ValueError('Ferret frame command (%s) failed: %s' % (cmdstr, errmsg))

