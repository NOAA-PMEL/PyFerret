'''
CmndHelperPQ is a helper class for dealing with commands
sent to a PyQt piped viewer.

This package was developed by the Thermal Modeling and Analysis
Project (TMAP) of the National Oceanographic and Atmospheric
Administration's (NOAA) Pacific Marine Environmental Lab (PMEL).
'''

import sys

# First try to import PyQt5, then try PyQt4 if that fails
try:
    import PyQt5
    QT_VERSION = 5
except ImportError:
    import PyQt4
    QT_VERSION = 4

# Now that the PyQt version is determined, import the parts
# allowing any import errors to propagate out
if QT_VERSION == 5:
    from PyQt5.QtCore import Qt, QPointF, QSizeF
    from PyQt5.QtGui  import QBrush, QColor, QFont, QPainterPath, QPen
else:
    from PyQt4.QtCore import Qt, QPointF, QSizeF
    from PyQt4.QtGui  import QBrush, QColor, QFont, QPainterPath, QPen


class SidesRectF(object):
    '''
    Trivial helper class for defining a rectangle with floating point
    values for the left-x, top-y, right-x, and bottom-y edges.
    '''
    def __init__(self, left, top, right, bottom):
        '''
        Create a SidesRectF with the given left, top, right,
        and bottom as float values.
        '''
        super(SidesRectF, self).__init__()
        self.__left = float(left)
        self.__top = float(top)
        self.__right = float(right)
        self.__bottom = float(bottom)

    def left(self):
        '''
        Return the left value as a float.
        '''
        return self.__left

    def setLeft(self, val):
        '''
        Set the SidesRectF left as a float value of the argument.
        '''
        self.__left = float(val)

    def top(self):
        '''
        Return the top value as a float.
        '''
        return self.__top

    def setTop(self, val):
        '''
        Set the SidesRectF top as a float value of the argument.
        '''
        self.__top = float(val)

    def right(self):
        '''
        Return the right value as a float.
        '''
        return self.__right

    def setRight(self, val):
        '''
        Set the SidesRectF right as a float value of the argument.
        '''
        self.__right = float(val)

    def bottom(self):
        '''
        Return the bottom value as a float.
        '''
        return self.__bottom

    def setBottom(self, val):
        '''
        Set the SidesRectF bottom as a float value of the argument.
        '''
        self.__bottom = float(val)


class SymbolPath(object):
    '''
    Trivial helper class for defining a symbol
    '''
    def __init__(self, painterpath, isfilled):
        '''
        Create a SymbolPath representing a symbol.

        Arguments:
            painterpath: the QPainterPath representing this symbol
            isfilled: if True, the symbol should be drawn with a
                    solid brush; if False, the symbol should be
                    drawn with a solid pen
        '''
        super(SymbolPath, self).__init__()
        self.__painterpath = painterpath
        self.__isfilled = isfilled
        if isfilled:
            try:
                self.__painterpath = painterpath.simplified()
            except:
                pass

    def painterPath(self):
        '''
        Return the QPainterPath for this symbol
        '''
        return self.__painterpath

    def isFilled(self):
        '''
        Return True if the symbol should be drawn with a solid brush;
        return False if the symbol should be drawn with a solid pen.
        '''
        return self.__isfilled


class CmndHelperPQ(object):
    '''
    Helper class of static methods for dealing with commands
    sent to a PyQt piped viewer.
    '''
    def __init__(self, viewer):
        '''
        Creates a cmndpipe command helper.  The widget viewer
        is only used for determining the default font and for
        translation of error messages.
        '''
        super(CmndHelperPQ, self).__init__()
        self.__viewer = viewer
        self.__symbolpaths = { }

    def getFontFromCmnd(self, fontinfo):
        '''
        Returns a QFont based on the information in the dictionary
        fontinfo.

        Recognized keys in the font dictionary are:
            "family": font family name (string)
            "size": text size in points (1/72 inches)
            "italic": italicize? (False/True)
            "bold": make bold? (False/True)
            "underline": underline?  (False/True)
        '''
        try:
            myfont = QFont(fontinfo["family"])
        except KeyError:
            myfont = self.__viewer.font()
        try:
            myfont.setPointSizeF(fontinfo["size"])
        except KeyError:
            pass
        try:
            myfont.setItalic(fontinfo["italic"])
        except KeyError:
            pass
        try:
            myfont.setBold(fontinfo["bold"])
        except KeyError:
            pass
        try:
            myfont.setUnderline(fontinfo["underline"])
        except KeyError:
            pass
        return myfont

    def getBrushFromCmnd(self, brushinfo):
        '''
        Returns a QBrush based on the information in the dictionary
        brushinfo.  A ValueError is raised if the value for the
        "style" key, if given, is not recognized.

        Recognized keys in the fill dictionary are:
            "color": color name or 24-bit RGB integer value
                         (eg, 0xFF0088)
            "alpha": alpha value from 0 (transparent) to 255 (opaque)
            "style": brush style name ("solid", "dense1" to "dense7",
                         "none", "hor", "ver", "cross",
                         "bdiag", "fdiag", "diagcross")
        '''
        try:
            mycolor = self.getColorFromCmnd(brushinfo)
            mybrush = QBrush(mycolor)
        except KeyError:
            mybrush = QBrush()
        try:
            mystyle = brushinfo["style"]
            if mystyle == "solid":
                mystyle = Qt.SolidPattern
            elif mystyle == "dense1":
                mystyle = Qt.Dense1Pattern
            elif mystyle == "dense2":
                mystyle = Qt.Dense2Pattern
            elif mystyle == "dense3":
                mystyle = Qt.Dense3Pattern
            elif mystyle == "dense4":
                mystyle = Qt.Dense4Pattern
            elif mystyle == "dense5":
                mystyle = Qt.Dense5Pattern
            elif mystyle == "dense6":
                mystyle = Qt.Dense6Pattern
            elif mystyle == "dense7":
                mystyle = Qt.Dense7Pattern
            elif mystyle == "none":
                mystyle = Qt.NoBrush
            elif mystyle == "hor":
                mystyle = Qt.HorPattern
            elif mystyle == "ver":
                mystyle = Qt.VerPattern
            elif mystyle == "cross":
                mystyle = Qt.CrossPattern
            elif mystyle == "bdiag":
                mystyle = Qt.BDiagPattern
            elif mystyle == "fdiag":
                mystyle = Qt.FDiagPattern
            elif mystyle == "diagcross":
                mystyle = Qt.DiagCrossPattern
            else:
                raise ValueError("Unknown brush style '%s'" % str(mystyle))
            mybrush.setStyle(mystyle)
        except KeyError:
            pass
        return mybrush

    def getPenFromCmnd(self, peninfo):
        '''
        Returns a QPen based on the information in the dictionary
        peninfo.  A ValueError is raised if the value for the
        "style", "capstyle", or "joinstyle" key, if given, is not
        recognized.

        Recognized keys in the outline dictionary are:
            "color": color name or 24-bit RGB integer value
                         (eg, 0xFF0088)
            "alpha": alpha value from 0 (transparent) to 255 (opaque)
            "width": pen width in points (1/72 inches); possibly 
                     further scaled by the width scaling factor 
            "style": pen style name ("solid", "dash", "dot", "dashdot",
                         "dashdotdot")
            "capstyle": pen cap style name ("square", "flat", "round")
            "joinstyle": pen join style name ("bevel", "miter", "round")
        '''
        try:
            mycolor = self.getColorFromCmnd(peninfo)
            mypen = QPen(mycolor)
        except KeyError:
            mypen = QPen()
        try:
            penwidth  = float(peninfo["width"])
            penwidth *= self.__viewer.widthScalingFactor()
            mypen.setWidthF(penwidth)
        except KeyError:
            pass
        try:
            mystyle = peninfo["style"]
            if mystyle == "solid":
                mystyle = Qt.SolidLine
            elif mystyle == "dash":
                mystyle = Qt.DashLine
            elif mystyle == "dot":
                mystyle = Qt.DotLine
            elif mystyle == "dashdot":
                mystyle = Qt.DashDotLine
            elif mystyle == "dashdotdot":
                mystyle = Qt.DashDotDotLine
            else:
                raise ValueError("Unknown pen style '%s'" % str(mystyle))
            mypen.setStyle(mystyle)
        except KeyError:
            pass
        try:
            mystyle = peninfo["capstyle"]
            if mystyle == "square":
                mystyle = Qt.SquareCap
            elif mystyle == "flat":
                mystyle = Qt.FlatCap
            elif mystyle == "round":
                mystyle = Qt.RoundCap
            else:
                raise ValueError("Unknown pen cap style '%s'" % str(mystyle))
            mypen.setCapStyle(mystyle)
        except KeyError:
            pass
        try:
            mystyle = peninfo["joinstyle"]
            if mystyle == "bevel":
                mystyle = Qt.BevelJoin
            elif mystyle == "miter":
                mystyle = Qt.MiterJoin
            elif mystyle == "round":
                mystyle = Qt.RoundJoin
            else:
                raise ValueError("Unknown pen join style '%s'" % str(mystyle))
            mypen.setJoinStyle(mystyle)
        except KeyError:
            pass
        return mypen

    def getSymbolFromCmnd(self, symbolinfo):
        '''
        Returns the SymbolPath for the symbol described in symbolinfo, 
        which can either be a string or a dictionary.  

        If symbolinfo is a string, it should be the name of a symbol that 
        has already been defined, either as a pre-defined symbol or from 
        a previous symbol definition.

        Current pre-defined symbol names are ones involving circles:
            'dot': very small filled circle
            'dotex': very small filled circle and outer lines of an ex mark
            'dotplus': very small filled circle and outer lines of a plus mark
            'circle': unfilled circle
            'circex': small unfilled circle and outer lines of an ex mark
            'circplus': small unfilled circle and outer lines of a plus mark

        If symbolinfo is a dictionary, the following key/value pairs are 
        recognized:
            'name' : (string) symbol name (required)
            'pts'  : (sequence of pairs of floats) vertex coordinates
            'fill' : (bool) color-fill symbol?

        If 'pts' is given, the value is coordinates that define the symbol 
        as multiline subpaths in a [-50,50] square.  The location of the 
        point this symbol represents will be at the center of the square. 
        An invalid coordinate (outside [-50,50]) will terminate the current 
        subpath, and the next valid coordinate will start a new subpath. 
        This definition will replace an existing symbol with the given name.

        If 'pts' is not given, the symbol must already be defined, either as 
        a pre-defined symbol (see above) or from a previous symbol definition.

        Raises:
             TypeError  - if symbolinfo is neither a string nor a dictionary
             KeyError   - if symbolinfo is a dictionary and 
                          the key 'name' is not given 
             ValueError - if there are problems generating the symbol
        '''
        # get the information about the symbol
        if isinstance(symbolinfo, str):
            symbol = symbolinfo
            pts = None
            fill = False
        elif isinstance(symbolinfo, dict):
            symbol = symbolinfo['name']
            pts = symbolinfo.get('pts', None)
            fill = symbolinfo.get('fill', False)
        else:
            raise TypeError('symbolinfo must either be a dictionary or a string')

        if pts is None:
            # no path given; check if already defined
            sympath = self.__symbolpaths.get(symbol)
            if sympath is not None:
                return sympath
            # symbol not defined - if well known, create a SymbolPath for it
            if symbol == 'dot':
                path = QPainterPath()
                path.addEllipse(-10.0, -10.0, 20.0, 20.0)
                sympath = SymbolPath(path, True)
            elif symbol == 'dotplus':
                path = QPainterPath()
                path.addEllipse(-10.0, -10.0, 20.0, 20.0)
                # filled path, so need to draw "lines" as rectangles
                path.addRect( -5.0, -50.0, 10.0, 20.0)
                path.addRect( -5.0,  30.0, 10.0, 20.0)
                path.addRect(-50.0,  -5.0, 20.0, 10.0)
                path.addRect( 30.0,  -5.0, 20.0, 10.0)
                sympath = SymbolPath(path, True)
            elif symbol == 'dotex':
                path = QPainterPath()
                path.addEllipse(-10.0, -10.0, 20.0, 20.0)
                # filled path, so need to draw "lines" as rectangles
                path.moveTo(-38.5, -31.5)
                path.lineTo(-31.5, -38.5)
                path.lineTo(-17.5, -24.5)
                path.lineTo(-24.5, -17.5)
                path.closeSubpath()
                path.moveTo(-38.5,  31.5)
                path.lineTo(-31.5,  38.5)
                path.lineTo(-17.5,  24.5)
                path.lineTo(-24.5,  17.5)
                path.closeSubpath()
                path.moveTo( 38.5, -31.5)
                path.lineTo( 31.5, -38.5)
                path.lineTo( 17.5, -24.5)
                path.lineTo( 24.5, -17.5)
                path.closeSubpath()
                path.moveTo( 38.5,  31.5)
                path.lineTo( 31.5,  38.5)
                path.lineTo( 17.5,  24.5)
                path.lineTo( 24.5,  17.5)
                path.closeSubpath()
                sympath = SymbolPath(path, True)
            elif symbol == 'circle':
                path = QPainterPath()
                path.addEllipse(-40.0, -40.0, 80.0, 80.0)
                sympath = SymbolPath(path, False)
            elif symbol == 'circplus':
                path = QPainterPath()
                path.addEllipse(-20.0, -20.0, 40.0, 40.0)
                # not a filled path, so just draw the lines
                path.moveTo(  0.0, -47.5)
                path.lineTo(  0.0, -21.0)
                path.moveTo(  0.0,  47.5)
                path.lineTo(  0.0,  21.0)
                path.moveTo(-47.5,   0.0)
                path.lineTo(-21.0,   0.0)
                path.moveTo( 47.5,   0.0)
                path.lineTo( 21.0,   0.0)
                sympath = SymbolPath(path, False)
            elif symbol == 'circex':
                path = QPainterPath()
                path.addEllipse(-20.0, -20.0, 40.0, 40.0)
                # not a filled path, so just draw the lines
                path.moveTo(-35.0, -35.0)
                path.lineTo(-20.0, -20.0)
                path.moveTo(-35.0,  35.0)
                path.lineTo(-20.0,  20.0)
                path.moveTo( 35.0, -35.0)
                path.lineTo( 20.0, -20.0)
                path.moveTo( 35.0,  35.0)
                path.lineTo( 20.0,  20.0)
                sympath = SymbolPath(path, False)
            else:
                raise ValueError("Unknown symbol '%s'" % str(symbol))
        else:
            # define (or redefine) a symbol from the given path
            try:
                coords = [ [ float(val) for val in coord ] for coord in pts ]
                if not coords:
                    raise ValueError
                for crd in coords:
                    if len(crd) != 2:
                        raise ValueError
            except Exception:
                raise ValueError('pts, if given, must be a sequence of pairs of numbers')
            path = QPainterPath()
            somethingdrawn = False
            newstart = True
            for (xval, yval) in coords:
                # flip so positive y is up
                yval *= -1.0
                if (xval < -50.0) or (xval > 50.0) or (yval < -50.0) or (yval > 50.0):
                    # end the current subpath
                    newstart = True
                elif newstart:
                    # start a new subpath; Qt closes the previous subpath automatically
                    path.moveTo(xval, yval)
                    newstart = False
                else:
                    # continue the current subpath
                    path.lineTo(xval, yval)
                    somethingdrawn = True
            if not somethingdrawn:
                del path
                raise ValueError('symbol definition does not contain any drawn lines')
            # Qt closes the (sub)path automatically
            sympath = SymbolPath(path, fill)
        # save and return the SymbolPath
        self.__symbolpaths[symbol] = sympath
        return sympath

    def getSizeFromCmnd(self, sizeinfo):
        '''
        Returns a QSizeF based on the information in the dictionary
        sizeinfo.  Recognized keys are "width" and "height", and
        correspond to those float values in the QSizeF.  Values not
        given in sizeinfo are assigned as zero in the returned QSizeF.
        '''
        myrect = QSizeF(0.0, 0.0)
        try:
            myrect.setWidth(float(sizeinfo["width"]))
        except KeyError:
            pass
        try:
            myrect.setHeight(float(sizeinfo["height"]))
        except KeyError:
            pass
        return myrect

    def getSidesFromCmnd(self, rectinfo):
        '''
        Returns a SidesQRectF based on the information in the dictionary
        rectinfo.  Recognized keys are "left", "top", "right", and "bottom",
        and correspond to those float values in the SidesQRectF.  Default
        values: "left": 0.0, "top": 0.0, "right":1.0, "bottom":1.0
        '''
        myrect = SidesRectF(left=0.0, top=0.0, right=1.0, bottom=1.0)
        try:
            myrect.setLeft(float(rectinfo["left"]))
        except KeyError:
            pass
        try:
            myrect.setTop(float(rectinfo["top"]))
        except KeyError:
            pass
        try:
            myrect.setRight(float(rectinfo["right"]))
        except KeyError:
            pass
        try:
            myrect.setBottom(float(rectinfo["bottom"]))
        except KeyError:
            pass
        return myrect

    def getColorFromCmnd(self, colorinfo):
        '''
        Returns a QColor based on the information in the dictionary
        colorinfo.  Raises a KeyError if the "color" key is not given.

        Recognized keys are:
            "color": color name or 24-bit RGB integer value
                         (eg, 0xFF0088)
            "alpha": alpha value from 0 (transparent) to 255 (opaque)
                     if viewer.ignoreAlpha True, this value is ignored
        '''
        colordata = colorinfo["color"]
        mycolor = QColor(colordata)
        if not mycolor.isValid():
            raise ValueError("Invalid color '%s'" % str(colordata))
        if not self.__viewer.ignoreAlpha():
            try:
                mycolor.setAlpha(int(colorinfo["alpha"]))
            except KeyError:
                pass
        return mycolor

    def computeARGB32PreMultInt(self, color):
        '''
        Returns the Format_ARGB32_Premultiplied integer value
        of the given QColor.
        '''
        (redint, greenint, blueint, alphaint) = color.getRgb()
        if self.__viewer.ignoreAlpha():
            alphaint = 255
        elif (alphaint < 255):
            # Scale the RGB values by the alpha value
            alphafactor = alphaint / 255.0
            redint = int( redint * alphafactor + 0.5 )
            if redint > alphaint:
                redint = alphaint
            greenint = int( greenint * alphafactor + 0.5 )
            if greenint > alphaint:
                greenint = alphaint
            blueint = int( blueint * alphafactor + 0.5 )
            if blueint > alphaint:
                blueint = alphaint
        fillint = ((alphaint * 256 + redint) * 256 + \
                   greenint) * 256 + blueint
        return fillint

