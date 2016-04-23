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


def setregion(X=None, Y=None, Z=None, T=None, E=None, F=None, 
              I=None, J=None, K=None, L=None, M=None, N=None, qual=None):
    """
    Specifies the default space-time region for the evaluation of expressions in Ferret.
    X (float or float slice): X (longitude) axis position or range (inclusive)
    Y (float or float slice): Y (latitude) axis position or range (inclusive)
    Z (float or float slice): Z (level) axis position or range (inclusive)
    T (float or float slice): T (time) axis position or range (inclusive)
    E (float or float slice): E (ensemble) axis position or range (inclusive)
    F (float or float slice): F (forecast) axis position or range (inclusive)
    I (int or int slice): X (longitude) axis index or range of indices (zero based)
    J (int or int slice): Y (latitude) axis index or range of indices (zero based)
    K (int or int slice): Z (level) axis index or range of indices (zero based)
    L (int or int slice): T (time) axis index or range of indices (zero based)
    M (int or int slice): E (ensemble) axis index or range of indices (zero based)
    N (int or int slice): F (forecast) axis index or range of indices (zero based)
    """
    cmdstr = 'SET REGION'
    if X is not None:
        if isinstance(X, numbers.Real):
            cmdstr += '/X=' + str(X)
        elif isinstance(X, str):
            cmdstr += '/X=' + X
        elif instance(X, slice) and isinstance(X.start, numbers.Real) and isinstance(X.stop, numbers.Real) and (X.step is None):
            cmdstr += '/X=' + str(X.start) + ':' + str(X.stop)
        elif instance(X, slice) and isinstance(X.start, str) and isinstance(X.stop, str) and (X.step is None):
            cmdstr += '/X=' + X.start + ':' + X.stop
        else:
            raise ValueError('definition for X is invalid')
    if Y is not None:
        if isinstance(Y, numbers.Real):
            cmdstr += '/Y=' + str(Y)
        elif isinstance(Y, str):
            cmdstr += '/Y=' + Y
        elif instance(Y, slice) and isinstance(Y.start, numbers.Real) and isinstance(Y.stop, numbers.Real) and (Y.step is None):
            cmdstr += '/Y=' + str(Y.start) + ':' + str(Y.stop)
        elif instance(Y, slice) and isinstance(Y.start, str) and isinstance(Y.stop, str) and (Y.step is None):
            cmdstr += '/Y=' + Y.start + ':' + Y.stop
        else:
            raise ValueError('definition for Y is invalid')
    if Z is not None:
        if isinstance(Z, numbers.Real):
            cmdstr += '/Z=' + str(Z)
        elif isinstance(Z, str):
            cmdstr += '/Z=' + Z
        elif instance(Z, slice) and isinstance(Z.start, numbers.Real) and isinstance(Z.stop, numbers.Real) and (Z.step is None):
            cmdstr += '/Z=' + str(Z.start) + ':' + str(Z.stop)
        elif instance(Z, slice) and isinstance(Z.start, str) and isinstance(Z.stop, str) and (Z.step is None):
            cmdstr += '/Z=' + Z.start + ':' + Z.stop
        else:
            raise ValueError('definition for Z is invalid')
    if T is not None:
        if isinstance(T, numbers.Real):
            cmdstr += '/T=' + str(T)
        elif isinstance(T, str):
            cmdstr += '/T=' + T
        elif instance(T, slice) and isinstance(T.start, numbers.Real) and isinstance(T.stop, numbers.Real) and (T.step is None):
            cmdstr += '/T=' + str(T.start) + ':' + str(T.stop)
        elif instance(T, slice) and isinstance(T.start, str) and isinstance(T.stop, str) and (T.step is None):
            cmdstr += '/T=' + T.start + ':' + T.stop
        else:
            raise ValueError('definition for T is invalid')
    if E is not None:
        if isinstance(E, numbers.Real):
            cmdstr += '/E=' + str(E)
        elif isinstance(E, str):
            cmdstr += '/E=' + E
        elif instance(E, slice) and isinstance(E.start, numbers.Real) and isinstance(E.stop, numbers.Real) and (E.step is None):
            cmdstr += '/E=' + str(E.start) + ':' + str(E.stop)
        elif instance(E, slice) and isinstance(E.start, str) and isinstance(E.stop, str) and (E.step is None):
            cmdstr += '/E=' + E.start + ':' + E.stop
        else:
            raise ValueError('definition for E is invalid')
    if F is not None:
        if isinstance(F, numbers.Real):
            cmdstr += '/F=' + str(F)
        elif isinstance(F, str):
            cmdstr += '/F=' + F
        elif instance(F, slice) and isinstance(F.start, numbers.Real) and isinstance(F.stop, numbers.Real) and (F.step is None):
            cmdstr += '/F=' + str(F.start) + ':' + str(F.stop)
        elif instance(F, slice) and isinstance(F.start, str) and isinstance(F.stop, str) and (F.step is None):
            cmdstr += '/F=' + F.start + ':' + F.stop
        else:
            raise ValueError('definition for F is invalid')
    if I is not None:
        if X is not None:
            raise ValueError('X and I cannot both be given')
        if isinstance(I, int):
            cmdstr += '/I=' + str(I + 1)
        elif instance(I, slice) and isinstance(I.start, int) and isinstance(I.stop, int) and (I.step is None):
            cmdstr += '/I=' + str(I.start + 1) + ':' + str(I.stop + 1)
        else:
            raise ValueError('definition for I is invalid')
    if J is not None:
        if Y is not None:
            raise ValueError('Y and J cannot both be given')
        if isinstance(J, int):
            cmdstr += '/J=' + str(J + 1)
        elif instance(J, slice) and isinstance(J.start, int) and isinstance(J.stop, int) and (J.step is None):
            cmdstr += '/J=' + str(J.start + 1) + ':' + str(J.stop + 1)
        else:
            raise ValueError('definition for J is invalid')
    if K is not None:
        if Z is not None:
            raise ValueError('Z and K cannot both be given')
        if isinstance(K, int):
            cmdstr += '/K=' + str(K + 1)
        elif instance(K, slice) and isinstance(K.start, int) and isinstance(K.stop, int) and (K.step is None):
            cmdstr += '/K=' + str(K.start + 1) + ':' + str(K.stop + 1)
        else:
            raise ValueError('definition for K is invalid')
    if L is not None:
        if T is not None:
            raise ValueError('T and L cannot both be given')
        if isinstance(L, int):
            cmdstr += '/L=' + str(L + 1)
        elif instance(L, slice) and isinstance(L.start, int) and isinstance(L.stop, int) and (L.step is None):
            cmdstr += '/L=' + str(L.start + 1) + ':' + str(L.stop + 1)
        else:
            raise ValueError('definition for L is invalid')
    if M is not None:
        if E is not None:
            raise ValueError('E and M cannot both be given')
        if isinstance(M, int):
            cmdstr += '/M=' + str(M + 1)
        elif instance(M, slice) and isinstance(M.start, int) and isinstance(M.stop, int) and (M.step is None):
            cmdstr += '/M=' + str(M.start + 1) + ':' + str(M.stop + 1)
        else:
            raise ValueError('definition for M is invalid')
    if N is not None:
        if F is not None:
            raise ValueError('F and N cannot both be given')
        if isinstance(N, int):
            cmdstr += '/N=' + str(N + 1)
        elif instance(N, slice) and isinstance(N.start, int) and isinstance(N.stop, int) and (N.step is None):
            cmdstr += '/N=' + str(N.start + 1) + ':' + str(N.stop + 1)
        else:
            raise ValueError('definition for N is invalid')
    (errval, errmsg) = pyferret.run(cmdstr)
    if errval != pyferret.FERR_OK:
        raise ValueError('Ferret set region command (%s) failed: %s' % (cmdstr, errmsg))


def contour(fvar, over=False, qual=''):
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


def fill(fvar, line=False, over=False, qual=''):
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


def shade(fvar, over=False, qual=''):
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
    

def saveplot(name, fmt='', qual=''):
    """
    Save the current plot.  If format is not given,
    the format is guessed from the filename extension.
        name (string): name of the file to contain the plot
        fmt (string): format of the plot file
        qual (string): qualifiers to add to the Ferret FRAME command
    """
    if not isinstance(name, str):
        raise ValueError('name (plot file name) must be a string')
    if not isinstance(fmt, str):
        raise ValueError('fmt (plot file format) must be a string')
    if not isinstance(qual, str):
        raise ValueError('qual (Ferret qualifiers) must be a string')
    cmdstr = 'FRAME/FILE="%s"' % name
    if fmt:
        cmdstr += '/FORMAT=%s' % fmt
    if qual:
        cmdstr += qual
    (errval, errmsg) = pyferret.run(cmdstr)
    if errval != pyferret.FERR_OK:
        raise ValueError('Ferret frame command (%s) failed: %s' % (cmdstr, errmsg))

