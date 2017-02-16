"""
Convenience methods for executing common Ferret commands.
These method generate appropriate Ferret command strings and then
execute them using the pyferret.run method.
"""

import numbers
import pyferret


def setwindow(num=1, plotasp=None, axisasp=None, color=None, pal=None, 
              thick=None, logo=None, outline=None):
    """
    Assigns the plot window to use for subsequent plotting commands.
    Also provides assignment of common window plots.  
    Note that plotasp and axisasp cannot both be given.
        num (int): window number 1-8 to use for plots.
        plotasp (float): aspect ratio (Y/X) for the plot window.
            If not given, the current ratio is unchanged.
            The default ratio on start-up is 0.86
        axisasp (float): aspect ratio (Y/X) for the plot axes.
            If not given, the current ratio is unchanged.
            The default ratio on start-up is 0.75
        color (string, tuple of int): background color for the plot;
            can be one of the color names 'black', 'blue', 'green', 
            'lightblue', 'purple', or 'red', or a tuple
            of int values in [0,100] giving RGB or RGBA values.
            If not given, the current value is unchanged.
            The default background color on start-up is opaque white.
        pal (string): default color palette to use in plots.
            If not given, thr current value is unchanged.
        thick (float): line thickness scaling factor for the plot.
            If not given, the current scaling factor is unchanged.
            The default line thickness scaling factor on start-up is 1.0
        logo (boolean): include the Ferret logo in the plot?
            If not given, the current value is unchanged.
            The default on start-up is to include the logo.
        outline (float): if positive, thickness of polygon outlines;
            used to fix the 'thin white line' issue in plots.
            If not given, the current value is unchanged.
            The default on start-up is zero (no outlines drawn).
            
    Raises a ValueError if a problem occurs.
    """
    # create and execute the SET WINDOW command
    cmdstr = 'SET WINDOW'
    if (plotasp is not None) and (axisasp is not None):
        raise ValueError('only one of plotasp and axisasp can be given')
    if plotasp is not None:
        if (not isinstance(plotasp, numbers.Real)) or (plotasp <= 0):
            raise ValueError('plotasp, if given, must be a positive number')
        cmdstr += '/ASPECT=' + str(plotasp)
    if axisasp is not None:
        if (not isinstance(axisasp, numbers.Real)) or (axisasp <= 0):
            raise ValueError('axisasp, if given, must be a positive number')
        cmdstr += '/ASPECT=' + str(axisasp) + ':AXIS'
    if thick is not None:
        if (not isinstance(thick, numbers.Real)) or (thick <= 0):
            raise ValueError('thick, if given, must be a positive number')
        cmdstr += '/THICK=' + str(thick)
    if outline is not None:
        if (not isinstance(outline, numbers.Real)) or (outline < 0):
            raise ValueErrror('outline, if given, must be a non-negative number')
        cmdstr += '/OUTLINE=' + str(outline)
    if color is not None:
        if isinstance(color, str):
            cmdstr += '/COLOR=' + color
        elif isinstance(color, tuple):
            if (len(color) < 3) or (len(color) > 4):
                raise ValueError('a color tuple must have three or four integer values')
            cmdstr += '/COLOR=' + str(color)
        else:
            raise ValueError('given color %s is not a string or tuple' % str(color))
    if (not isinstance(num, numbers.Integral)) or (num <= 0) or (num > 8):
        raise ValueError('window number %s is not a integer in [1,8]' % str(num))
    cmdstr += ' ' + str(num)
    (errval, errmsg) = pyferret.run(cmdstr)
    if errval != pyferret.FERR_OK:
        raise ValueError('Problems executing Ferret command %s: %s' % (cmdstr, errmsg))
    if pal is not None:
        cmdstr = 'PALETTE ' + str(pal)
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


def contourplot(fvar, region=None, over=False, pal=None, qual=''):
    """
    Create a contour plot of the specified Ferret variable using the Ferret CONTOUR command.
    Using the fill method to generated a color-filled contour plot.
    The variable needs to be 2D (or qualifiers need to be added to specify a 2D slice).
        fvar (string or FerVar): Ferret variable to plot
        region (FerRegion): space-time region to plot; 
                if None, the full extents of the data will be used
        over (bool): overlay on an existing plot?
        pal (string): color palette to use
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
    if region is not None:
        if not isinstance(region, pyferret.FerRegion):
            raise ValueError('region, if given, must be a FerRegion')
        cmdstr += region._ferretqualifierstr();
    if pal is not None:
        cmdstr += '/PALETTE=' + str(pal)
    if qual:
        cmdstr += qual
    cmdstr += ' '
    cmdstr += plotvar
    (errval, errmsg) = pyferret.run(cmdstr)
    if errval != pyferret.FERR_OK:
        raise ValueError('Ferret shade command (%s) failed: %s' % (cmdstr, errmsg))


def fillplot(fvar, region=None, line=False, over=False, pal=None, qual=''):
    """
    Create a color-filled contour plot of the specified Ferret variable using the Ferret 
    FILL command.  Drawing of the contour lines themselves is optional.
    The variable needs to be 2D (or qualifiers need to be added to specify a 2D slice).
        fvar (string or FerVar): Ferret variable to plot
        region (FerRegion): space-time region to plot; 
                if None, the full extents of the data will be used
        line (bool): draw the contour lines?
        over (bool): overlay on an existing plot?
        pal (string): color palette to use
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
    if region is not None:
        if not isinstance(region, pyferret.FerRegion):
            raise ValueError('region, if given, must be a FerRegion')
        cmdstr += region._ferretqualifierstr();
    if pal is not None:
        cmdstr += '/PALETTE=' + str(pal)
    if qual:
        cmdstr += qual
    cmdstr += ' '
    cmdstr += plotvar
    (errval, errmsg) = pyferret.run(cmdstr)
    if errval != pyferret.FERR_OK:
        raise ValueError('Ferret shade command (%s) failed: %s' % (cmdstr, errmsg))


def shadeplot(fvar, region=None, over=False, pal=None, qual=''):
    """
    Create a colored plot of the specified Ferret variable using the Ferret SHADE command.
    (Plot coloring grid cells based on the variable value in that cell.)
    The variable needs to be 2D (or qualifiers need to be added to specify a 2D slice).
        fvar (string or FerVar): Ferret variable to plot
        region (FerRegion): space-time region to plot; 
                if None, the full extents of the data will be used
        over (bool): overlay on an existing plot?
        pal (string): color palette to use
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
    if region is not None:
        if not isinstance(region, pyferret.FerRegion):
            raise ValueError('region, if given, must be a FerRegion')
        cmdstr += region._ferretqualifierstr();
    if pal is not None:
        cmdstr += '/PALETTE=' + str(pal)
    if qual:
        cmdstr += qual
    cmdstr += ' '
    cmdstr += plotvar
    (errval, errmsg) = pyferret.run(cmdstr)
    if errval != pyferret.FERR_OK:
        raise ValueError('Ferret shade command (%s) failed: %s' % (cmdstr, errmsg))


def shadeland(res=20, color='gray', over=True, solid=True, X=None, Y=None):
    """
    Shades land masses for the current longitude-latitude plot or the specified X-Y region.
        res (int): ETOPO dataset resolution (in minutes of a degree) to use; 
            the corresponding ETOPO dataset (eg, etopo20.cdf for 20) must be available.
            Typically 5, 10, 20, 40, 60, 120 are available from Ferret's standard datasets.
        color (str): name of the color or color palette to used for land.
        over (bool): if true, overlay onto the current longitude-latitude plot;
            if False, create a new plot of the given region
        solid (bool): if True, shade the land in a single solid color;
            if False, shade different elevations using the given color palette
        X (str): longitude limits for the region as low:high
            If not given and over is False, '0E:360E' is used.
            if not given and over is True, the full range of the given plot is used.
        Y (str): latitude limits for the region as low:high
            If not given and over is False, '90S:90N' is used.
            If not given and over is True, the full range of the given plot is used.
    """
    cmdstr = 'GO fland'
    cmdstr += ' ' + str(res)
    cmdstr += ' ' + str(color)
    if over:
        cmdstr += ' OVERLAY'
    else:
        cmdstr += ' BASEMAP'
    if solid:
        cmdstr += ' SOLID'
    else:
        cmdstr += ' DETAILED'
    if X is not None:
        cmdstr += ' X=' + str(X)
    elif not over:
        # assign the default here even though this matches the script
        cmdstr += ' X=0E:360E'
    elif Y is not None:
        # if Y is given, then have to have an X argument;
        # needs to be a double wrap for a complete overlay
        cmdstr += ' X=0E:720E'
    if Y is not None:
        cmdstr += ' Y=' + str(Y)
    elif not over:
        # assign the default here even though this matches the script
        cmdstr += ' Y=90S:90N'
    (errval, errmsg) = pyferret.run(cmdstr)
    if errval != pyferret.FERR_OK:
        raise ValueError('Ferret script command (%s) failed: %s' % (cmdstr, errmsg))


def shadewater(res=20, color='gray', over=True, solid=True, X=None, Y=None):
    """
    Shades water masses for the current longitude-latitude plot or the specified region.
        res (int): ETOPO dataset resolution (in minutes of a degree) to use; 
            the corresponding ETOPO dataset (eg, etopo20.cdf for 20) must be available.
            Typically 5, 10, 20, 40, 60, 120 are available from Ferret's standard datasets.
        color (str): name of the color or color palette to used for water masses
        over (bool): if true, overlay onto the current longitude-latitude plot;
            if False, create a new plot of the given region
        solid (bool): if True, shade the water masses in a single solid color;
            if False, shade different depths using the given color palette
        X (str): longitude limits for the region as low:high; 
            if not given and over is False, '0E:360E' is used
        Y (str): latitude limits for the region as low:high; 
            if not given and over is False, '90S:90N'
    """
    cmdstr = 'GO focean'
    cmdstr += ' ' + str(res)
    cmdstr += ' ' + str(color)
    if over:
        cmdstr += ' OVERLAY'
    else:
        cmdstr += ' BASEMAP'
    if solid:
        cmdstr += ' SOLID'
    else:
        cmdstr += ' DETAILED'
    if X is not None:
        cmdstr += ' X=' + str(X)
    elif not over:
        # assign the default here even though this matches the script
        cmdstr += ' X=0E:360E'
    elif Y is not None:
        # if Y is given, then have to have an X argument;
        # needs to be a double wrap for a complete overlay
        cmdstr += ' X=0E:720E'
    if Y is not None:
        cmdstr += ' Y=' + str(Y)
    elif not over:
        # assign the default here even though this matches the script
        cmdstr += ' Y=90S:90N'
    (errval, errmsg) = pyferret.run(cmdstr)
    if errval != pyferret.FERR_OK:
        raise ValueError('Ferret script command (%s) failed: %s' % (cmdstr, errmsg))


def pointplot(fvar, vs=None, color=None, sym=None, symsize=None, thick=None,
              line=False, title=None, region=None, over=False, label=True, qual=''):
    """
    Create a point plot of the given value, or the given value versus another value 
    (if vs is given), possibly colored by another value (if color is a FerVar).
    To create a line plot with symbols, use the pointplot command with the line 
    option set to True.
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
        sym (int): Ferret symbol number of the symbol to draw for the points.
                If not given, Ferret selects an appropriate symbol.
        symsize (float): size of the symbol in inches.
                If not given, Ferret select an appropriate size.
        thick (float): line thickness scaling factor when drawing symbols and lines
        line (bool): if True, draw a line between symbols/points
        title (string): title for the plot; if not given,  Ferret's default title is used
        region (FerRegion): space-time region to plot; 
                if None, the full extents of the data will be used
        over (bool): overlay onto an existing plot
        label (bool): if False, suppress all plot labels
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
    cmdstr = 'PLOT'
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
    # always draw the symbols
    cmdstr += '/SYMBOL'
    if sym is not None:
        if (not isinstance(sym, numbers.Integral)) or (sym < 0) or (sym > 88):
            raise ValueError('sym is not a valid Ferret symbol number')
        if sym == 0:
            cmdstr += '=DOT'
        else:
            cmdstr += '=' + str(sym)
    if symsize is not None:
        if (not isinstance(symsize, numbers.Real)) or (symsize <= 0):
            raise ValueError('symsize must be a positive number')
        cmdstr += '/SIZE=' + str(symsize)
    if thick is not None:
        if (not isinstance(thick, numbers.Real)) or (thick <= 0):
            raise ValueError('thick must be a positive number')
        cmdstr += '/THICK=' + str(thick)
    if line:
        cmdstr += '/LINE'
    if title is not None:
       if not isinstance(title, str):
           raise ValueError('title must be a string')
       cmdstr += '/TITLE="' + title + '"'
    if over:
        cmdstr += '/OVER'
    if region is not None:
        if not isinstance(region, pyferret.FerRegion):
            raise ValueError('region, if given, must be a FerRegion')
        cmdstr += region._ferretqualifierstr();
    if not label:
        cmdstr += '/NOLABEL'
    if qual:
        cmdstr += qual
    cmdstr += ' '
    cmdstr += plotvar
    (errval, errmsg) = pyferret.run(cmdstr)
    if errval != pyferret.FERR_OK:
        raise ValueError('Ferret plot command (%s) failed: %s' % (cmdstr, errmsg))


def lineplot(fvar, vs=None, color=None, thick=None, dash=None, title=None, 
             region=None, along=None, over=False, label=True, qual=''):
    """
    Create a line plot of the given value, or the given value versus another value 
    (if vs is given), possibly colored by another value (if color is a FerVar).
    To create a line plot with symbols, use the pointplot command with the line 
    option set to True.
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
        region (FerRegion): space-time region to plot; 
                if None, the full extents of the data will be used
        along (string; one of 'X','Y','Z','T','E','F', or lowercase): make a set of line 
                plots from two-dimensional data with this axis as the horizontal axis.
        over (bool): overlay onto an existing plot
        label (bool): if False, suppress all plot labels
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
            raise ValueError('thick must be a positive number')
        cmdstr += '/THICK=' + str(thick)
    if dash is not None:
        if (not isinstance(dash, tuple)) or (len(dash) != 4):
            raise ValueError('dash must be a tuple of four floats');
        cmdstr += '/DASH=' + str(dash)
    if title is not None:
       if not isinstance(title, str):
           raise ValueError('title must be a string')
       cmdstr += '/TITLE="' + title + '"'
    if along is not None:
       axisnames = ('X','Y','Z','T','E','F','x','y','z','t','e','f')
       if not along in axisnames:
           raise ValueError('along must be one of ' + str(axisnames))
       cmdstr += '/ALONG=' + along.upper()
    if over:
        cmdstr += '/OVER'
    if region is not None:
        if not isinstance(region, pyferret.FerRegion):
            raise ValueError('region must be a FerRegion')
        cmdstr += region._ferretqualifierstr();
    if not label:
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

