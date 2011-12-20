'''
The PyFerretBindings class is a base class providing common
methods in PipedViewer bindings for PyFerret graphics methods.

The PyQtViewPyFerretBindings class is a subclass of PyFerretBindings
using PyQtPipedViewer as the viewer.

The PyQtImagePyFerretBindings class is a subclass of PyFerretBindings
using PyQtImageViewer as the viewer. 

This package was developed by the Thermal Modeling and Analysis Project
(TMAP) of the National Oceanographic and Atmospheric Administration's (NOAA)
Pacific Marine Environmental Lab (PMEL).
'''

from pyferret.graphbind.abstractpyferretbindings import AbstractPyFerretBindings
from pipedviewer import PipedViewer


class PyFerretBindings(AbstractPyFerretBindings):
    '''
    Common methods in PipedViewer bindings for PyFerret graphical
    functions.  The createWindow method should be defined in a
    subclass in order to create a valid bindings class for PyFerret.
    '''

    def __init__(self):
        '''
        Create an instance of the the PipedViewer bindings for PyFerret
        graphical functions.  The createWindow method should be called
        to associate a new PipedViewer with these bindings.
        '''
        super(PyFerretBindings, self).__init__()
        self.__window = None

    def createPipedViewerWindow(self, viewertype, title, visible):
        '''
        Creates a PipedViewer of viewertype as the window of this
        instance of the bindings.

        Arguments:
            viewertype: type of PipedViewer to use 
            title: display title for the Window
            visible: display Window on start-up?

        Raises a RuntimeError if an active window is already associated
        with these bindings, or if there were problems with creating
        the window.

        Returns True.
        '''
        if self.__window != None:
            raise RuntimeError("createWindow called from bindings " \
                               "with an active window")
        self.__window = PipedViewer(viewertype)
        self.__window.submitCommand( { "action":"setTitle",
                                      "title":str(title) } )
        if visible:
            self.__window.submitCommand( {"action":"show"} )
        self.checkForErrorResponse()
        return True

    def checkForErrorResponse(self):
        '''
        Checks the response pipe for a message.  If anything is found,
        a RuntimeError is raised with the string of the full response.
        '''
        fullresponse = None
        response = self.__window.checkForResponse()
        while response:
            if fullresponse:
                fullresponse += '\n'
                fullresponse += str(response)
            else:
                fullresponse = str(response)
            response = self.__window.checkForResponse()
        if fullresponse:
            raise RuntimeError(fullresponse)

    # The remaining methods are common implementations of the required binding methods

    def deleteWindow(self):
        '''
        Shuts down the PyQtPipedViewer.

        Returns True.
        '''
        try:
            self.__window.submitCommand( { "action":"exit" } )
            self.checkForErrorResponse()
            self.__window.waitForViewerExit()
        finally:
            self.__window = None
        return True

    def setAntialias(self, antialias):
        '''
        Turns on (antilaias True) or off (antialias False) anti-aliasing
        in future drawing commands. 
        '''
        cmnd = { "action":"antialias",
                 "antialias":bool(antialias) }
        self.__window.submitCommand(cmnd)
        self.checkForErrorResponse()
        
    def beginView(self, leftfrac, bottomfrac, rightfrac, topfrac, clipit):
        '''
        Start a view in the PyQtPipedViewer Window.  The view fractions
        start at (0.0, 0.0) in the left top corner and increase to
        (1.0, 1.0) in the right bottom corner; thus leftfrac must be less
        than rightfrac and topfrac must be less than bottomfrac.

        Arguments:
            leftfrac:    [0,1] fraction of the Window width
                         for the left side of the View
            bottomfrac:  [0,1] fraction of the Window height
                         for the bottom side of the View
            rightfrac:   [0,1] fraction of the Window width
                         for the right side of the View
            topfrac:     [0,1] fraction of the Window height
                         for the top side of the View
           clipit:      clip drawings to this View?
        '''
        leftfracflt = float(leftfrac)
        bottomfracflt = float(bottomfrac)
        rightfracflt = float(rightfrac)
        topfracflt = float(topfrac)
        if (0.0 > leftfracflt) or (leftfracflt >= rightfracflt) or (rightfracflt > 1.0):
            raise ValueError("leftfrac (%f) and rightfrac (%f) must be in [0.0, 1.0] " \
                             "with leftfrac < rightfrac" % (leftfracflt, rightfracflt))
        if (0.0 > topfracflt) or (topfracflt >= bottomfracflt) or (bottomfracflt > 1.0):
            raise ValueError("topfrac (%f) and bottomfrac (%f) must be in [0.0, 1.0] " \
                             "with topfrac < bottomfrac" % (topfracflt, bottomfracflt))
        cmnd = { "action":"beginView",
                 "viewfracs":{"left":leftfracflt, "right":rightfracflt,
                              "top":topfracflt, "bottom":bottomfracflt }, 
                 "clip":bool(clipit) }
        self.__window.submitCommand(cmnd)
        self.checkForErrorResponse()

    def clipView(self, clipit):
        '''
        Enable or disable clipping to the current View.

        Arguments:
            clipit: clip drawings to the current View?
        '''
        cmnd = { "action":"clipView",
                 "clip":bool(clipit) }
        self.__window.submitCommand(cmnd)
        self.checkForErrorResponse()

    def endView(self):
        '''
        Close a View in the PyQtPipedViewer Window
        '''
        self.__window.submitCommand( { "action":"endView" } )
        self.checkForErrorResponse()

    def updateWindow(self):
        '''
        Indicates the viewer should update the graphics displayed.
        '''
        cmnd = { "action":"update" }
        self.__window.submitCommand(cmnd)
        self.checkForErrorResponse()

    def clearWindow(self, fillcolor):
        '''
        Clears the Window of all drawings.  The Window is filled
        (initialized) with fillcolor.

        Arguments:
            fillcolor: Color to fill (initialize) the Window
        '''
        if fillcolor:
            # Make a copy of the fillcolor dictionary
            cmnd = dict(fillcolor)
        else:
            cmnd = { }
        cmnd["action"] = "clear"
        self.__window.submitCommand(cmnd)
        self.checkForErrorResponse()

    def resizeWindow(self, width, height):
        '''
        Sets the current size of the Window.

        Arguments:
            width: width of the Window, in "device units"
            height: height of the window in "device units"

        "device units" is pixels at the current window DPI
        '''
        cmnd = { "action":"resize",
                 "width":width,
                 "height":height }
        self.__window.submitCommand(cmnd)
        self.checkForErrorResponse()

    def windowDpi(self):
        '''
        Returns a two-tuple containing the screen resolution of
        the Window, in dots (pixels) per inch, in the horizontal
        (X) and vertical (Y) directions.
        '''
        cmnd = { "action":"dpi" }
        self.__window.submitCommand(cmnd)
        response = None
        try:
            # Wait indefinitely for a response
            # Make sure it is a valid response
            response = self.__window.checkForResponse(None)
            if (type(response) != tuple) or (len(response) != 2):
                raise ValueError
            dpix = float(response[0])
            dpiy = float(response[1])
            if (dpix <= 0.0) or (dpiy <= 0.0):
                raise ValueError
        except Exception:
            if not response:
                # error raised before a response obtained
                raise
            fullresponse = str(response)
            response = self.__window.checkForResponse()
            while response:
                fullresponse += '\n'
                fullresponse += response
                response = self.__window.checkForResponse()
            raise RuntimeError(fullresponse)
        return (dpix, dpiy)

    def showWindow(self, visible):
        '''
        Display or hide a Window.

        Arguments:
            visible: display (if True) or
                     hide (if False) the Window
        '''
        if visible:
            cmnd = { "action":"show" }
        else:
            cmnd = { "action":"hide" }
        self.__window.submitCommand(cmnd)
        self.checkForErrorResponse()

    def saveWindow(self, filename, fileformat, transparentbkg):
        '''
        Save the contents of the window to a file.

        Arguments:
            filename: name of the file to create
            fileformat: name of the format to use
            transparentbkg: should the background be transparent?

        If fileformat is None or empty, the fileformat
        is guessed from the filename extension.
        '''
        cmnd = { "action":"save",
                 "filename":filename,
                 "transparentbkg": transparentbkg }
        if fileformat:
            cmnd["fileformat"] = fileformat
        self.__window.submitCommand(cmnd)
        self.checkForErrorResponse()

    def createColor(self, redfrac, greenfrac, bluefrac, opaquefrac):
        '''
        Returns a Color object from fractional [0.0, 1.0]
        intensities of the red, green, and blue channels.
        The opaquefrac is used to set the alpha channel.

        Arguments:
            redfrac: fractional [0.0, 1.0] red intensity
            greenfrac: fractional [0.0, 1.0] green intensity
            bluefrac: fractional [0.0, 1.0] blue intensity
            opaquefrac: fractional [0.0, 1.0] opaqueness
                (0.0 is transparent; 1.0 is opaque) of the color.
                For output that does not support an alpha channel,
                this will be silently ignored and the color will
                be completely opaque.

        Raises an error if unable to create the Color object.
        '''
        if (redfrac < 0.0) or (redfrac > 1.0):
            raise ValueError("redfrac must be a value in [0.0, 1.0]")
        if (greenfrac < 0.0) or (greenfrac > 1.0):
            raise ValueError("greenfrac must be a value in [0.0, 1.0]")
        if (bluefrac < 0.0) or (bluefrac > 1.0):
            raise ValueError("bluefrac must be a value in [0.0, 1.0]")
        if (opaquefrac < 0.0) or (opaquefrac > 1.0):
            raise ValueError("opaquefrac must be a value in [0.0, 1.0]")
        redint = int( 256.0 * redfrac )
        if redint == 256:
            redint = 255
        greenint = int( 256.0 * greenfrac )
        if greenint == 256:
            greenint = 255
        blueint = int( 256.0 * bluefrac )
        if blueint == 256:
            blueint = 255
        colorint = (redint * 256 + greenint) * 256 + blueint
        opaqueint = int( 256.0 * opaquefrac )
        if opaqueint == 256:
            opaqueint = 255
        return { "color":colorint, "alpha":opaqueint }

    def deleteColor(self, color):
        '''
        Delete a Color object created by createColor

        Arguments:
            color: Color to be deleted
        '''
        del color

    def createFont(self, familyname, fontsize, italic, bold, underlined):
        '''
        Returns a Font object.

        Arguments:
            familyname: name of the font family (e.g., "Helvetica", "Times");
                        None or an empty string uses the default font
            fontsize: desired size of the font (scales with view size)
            italic: use the italic version of the font?
            bold: use the bold version of the font?
            underlined: use the underlined version of the font?

        Raises an error if unable to create the Font object.
        '''
        fontdict = { "size":fontsize,
                     "italic":italic,
                     "bold":bold,
                     "underlined":underlined }
        if familyname:
            fontdict["family"] = familyname
        return fontdict

    def deleteFont(self, font):
        '''
        Delete a Font object created by createFont

        Arguments:
            font: Font to be deleted
        '''
        del font

    def createPen(self, color, width, style, capstyle, joinstyle):
        '''
        Returns a Pen object.

        Arguments:
            color: Color to use
            width: line width (scales with view size)
            style: line style name (e.g., "solid", "dash")
            capstyle: end-cap style name (e.g., "square")
            joinstyle: join style name (e.g., "bevel")

        Raises an error if unable to create the Pen object.
        '''
        if color:
            pen = dict(color)
        else:
            pen = { }
        if width:
            pen["width"] = width
        if style:
            pen["style"] = style
        if capstyle:
            pen["capstyle"] = capstyle
        if joinstyle:
            pen["joinstyle"] = joinstyle
        return pen

    def deletePen(self, pen):
        '''
        Delete a Pen object created by createPen

        Arguments:
            pen: Pen to be deleted
        '''
        del pen

    def createBrush(self, color, style):
        '''
        Returns a Brush object.

        Arguments:
            color: Color to use
            style: fill style name (e.g., "solid", "cross")

        Raises an error if unable to create the Brush object.
        '''
        if color:
            brush = dict(color)
        else:
            brush = { }
        if style:
            brush["style"] = style
        return brush

    def deleteBrush(self, brush):
        '''
        Delete a Brush object created by createBrush

        Arguments:
            brush: Brush to be deleted
        '''
        del brush

    def createSymbol(self, symbolname):
        '''
        Returns a Symbol object.

        Arguments:
            symbolname: name of the symbol.
                Currently supported values are:
                '.' (period): filled circle
                'o' (lowercase oh): unfilled circle
                '+': plus mark
                'x' (lowercase ex): x mark
                '*': asterisk
                '^': triangle
                "#": square

        Raises an error if unable to create the Symbol object.
        '''
        return symbolname

    def deleteSymbol(self, symbol):
        '''
        Delete a Symbol object created by createSymbol

        Arguments:
            symbol: Symbol to be deleted
        '''
        del symbol

    def drawMultiline(self, ptsx, ptsy, pen):
        '''
        Draws connected line segments.

        Arguments:
            ptsx: X-coordinates of the endpoints
            ptsy: Y-coordinates of the endpoints
            pen: the Pen to use to draw the line segments

        Coordinates are measured from the upper left corner
        in "device units" (pixels at the current window DPI).
        '''
        if len(ptsx) != len(ptsy):
            raise ValueError("the lengths of ptsx and ptsy are not the same")
        points = list(zip(ptsx, ptsy))
        cmnd = { "action":"drawMultiline",
                 "points":points,
                 "pen":pen }
        self.__window.submitCommand(cmnd)
        self.checkForErrorResponse()

    def drawPoints(self, ptsx, ptsy, symbol, color, ptsize):
        '''
        Draws discrete points.

        Arguments:
            ptsx: X-coordinates of the points
            ptsy: Y-coordinates of the points
            symbol: the Symbol to use to draw a point
            color: color of the Symbol (default color if None or empty)
            ptsize: size of the symbol (scales with view size)

        Coordinates are measured from the upper left corner
        in "device units" (pixels at the current window DPI).
        '''
        if len(ptsx) != len(ptsy):
            raise ValueError("the lengths of ptsx and ptsy are not the same")
        points = list(zip(ptsx, ptsy))
        if color:
            # make a copy of the color dictionary
            cmnd = dict(color)
        else:
            cmnd = { }
        cmnd["action"] = "drawPoints"
        cmnd["points"] = points
        cmnd["symbol"] = symbol
        cmnd["size"] = ptsize
        self.__window.submitCommand(cmnd)
        self.checkForErrorResponse()

    def drawPolygon(self, ptsx, ptsy, brush, pen):
        '''
        Draws a polygon.

        Arguments:
            ptsx: X-coordinates of the vertices
            ptsy: Y-coordinates of the vertices
            brush: the Brush to use to fill the polygon; if None
                    the polygon will not be filled
            pen: the Pen to use to outline the polygon; if None
                    the polygon will not be outlined

        Coordinates are measured from the upper left corner
        in "device units" (pixels at the current window DPI).
        '''
        if len(ptsx) != len(ptsy):
            raise ValueError("the lengths of ptsx and ptsy are not the same")
        points = list(zip(ptsx, ptsy))
        cmnd = { "action":"drawPolygon", "points":points }
        if brush:
            cmnd["fill"] = brush
        if pen:
            cmnd["outline"] = pen
        self.__window.submitCommand(cmnd)
        self.checkForErrorResponse()

    def drawRectangle(self, left, bottom, right, top, brush, pen):
        '''
        Draws a rectangle.

        Arguments:
            left: X-coordinate of the left edge
            bottom: Y-coordinate of the bottom edge
            right: X-coordinate of the right edge
            top: Y-coordinate of the top edge
            brush: the Brush to use to fill the polygon; if None
                    the polygon will not be filled
            pen: the Pen to use to outline the polygon; if None
                    the polygon will not be outlined

        Coordinates are measured from the upper left corner
        in "device units" (pixels at the current window DPI).
        '''
        cmnd = { "action":"drawRectangle",
                 "left":left, "bottom":bottom,
                 "right":right, "top": top }
        if brush:
            cmnd["fill"] = brush
        if pen:
            cmnd["outline"] = pen
        self.__window.submitCommand(cmnd)
        self.checkForErrorResponse()

    def drawMulticolorRectangle(self, left, bottom, right, top,
                                numrows, numcols, colors):
        '''
        Draws a filled rectangle using an array of solid colors.
        The rectangle is divided into a given number of equally
        spaced rows and a number of equally spaced columns.  Each
        of these cells is then filled with a color (using a solid
        brush) from the corresponding element in an array of colors.

        Arguments:
            left: X-coordinate of the left edge
            bottom: Y-coordinate of the bottom edge
            right: X-coordinate of the right edge
            top: Y-coordinate of the top edge
            numrows: the number of equally spaced rows
                    to subdivide the rectangle into
            numcols: the number of equally spaced columns
                    to subdivide the rectangle into
            colors: a flattened column-major 2-D list of colors
                    specifying the color of the corresponding cell.
                    The first row is at the top, the first column
                    is on the left.

        Coordinates are measured from the upper left corner
        in "device units" (pixels at the current window DPI).
        '''
        cmnd = { "action":"drawMulticolorRectangle",
                 "left":left, "bottom":bottom,
                 "right":right, "top": top,
                 "numrows":numrows, "numcols":numcols,
                 "colors":colors }
        self.__window.submitCommand(cmnd)
        self.checkForErrorResponse()

    def drawText(self, text, startx, starty, font, color, rotate):
        '''
        Draws text.

        Arguments:
            text: the text string to draw
            startx: the X-coordinate of the beginning baseline
            starty: the Y-coordinate of the beginning baseline
            font: the font to use
            color: the color to use as a solid brush or pen
            rotate: the angle of the text baseline in degrees
                    clockwise from horizontal

        Coordinates are measured from the upper left corner
        in "device units" (pixels at the current window DPI).
        '''
        cmnd = { "action":"drawText", "text":text,
                 "location":(startx,starty) }
        if font:
            cmnd["font"] = font
        if color:
            pen = dict(color)
            pen["style"] = "solid"
            cmnd["fill"] = pen
        if rotate != 0.0:
            cmnd["rotate"] = rotate
        self.__window.submitCommand(cmnd)
        self.checkForErrorResponse()


class PyQtViewPyFerretBindings(PyFerretBindings):
    '''
    PyFerretBindings using PyQtPipedViewer as the viewer.
    '''

    def createWindow(self, title, visible):
        '''
        Creates PyFerret bindings using a PyQtPipedViewer.

        Arguments:
            title: display title for the Window
            visible: display Window on start-up?

        Raises a RuntimeError if an active window is already associated
        with these bindings, or if there were problems with creating
        the window.

        Returns True.
        '''
        result = self.createPipedViewerWindow("PyQtPipedViewer",
                                              title, visible)
        return result


class PyQtImagePyFerretBindings(PyFerretBindings):
    '''
    PyFerretBindings using PyQtPipedImager as the viewer.
    '''

    def createWindow(self, title, visible):
        '''
        Creates PyFerret bindings using a PyQtPipedImager.

        Arguments:
            title: display title for the Window
            visible: display Window on start-up?

        Raises a RuntimeError if an active window is already associated
        with these bindings, or if there were problems with creating
        the window.

        Returns True.
        '''
        result = self.createPipedViewerWindow("PyQtPipedImager",
                                              title, visible)
        return result


if __name__ == "__main__":
    import pyferret
    import pyferret.graphbind

    # x and y coordinates of the vertices of a pentagon
    # (roughly) centered in a 1000 x 1000 square
    pentaptsx = ( 504.5, 100.0, 254.5, 754.5, 909.0, )
    pentaptsy = ( 100.0, 393.9, 869.4, 869.4, 393.9, )
    mypentax = [ 0.25 * ptx for ptx in pentaptsx ]
    mypentay = [ 0.25 * pty + 250 for pty in pentaptsy ]

    # RGBA tuples of the colors to create
    colorvals = ( (0.0, 0.0, 0.0, 1.0),   #  0 opaque black
                  (1.0, 1.0, 1.0, 1.0),   #  1 opaque white
                  (1.0, 0.0, 0.0, 1.0),   #  2 opaque red
                  (0.6, 0.5, 0.0, 1.0),   #  3 opaque yellowish
                  (0.0, 1.0, 0.0, 1.0),   #  4 opaque green
                  (0.0, 0.5, 0.5, 1.0),   #  5 opaque cyan
                  (0.0, 0.0, 1.0, 1.0),   #  6 opaque blue
                  (0.5, 0.0, 0.5, 1.0),   #  7 opaque magenta
                  (0.0, 0.0, 0.0, 0.35),  #  8 translucent black
                  (1.0, 1.0, 1.0, 0.35),  #  9 translucent white
                  (1.0, 0.0, 0.0, 0.35),  # 10 translucent red
                  (0.6, 0.5, 0.0, 0.35),  # 11 translucent yellowish
                  (0.0, 1.0, 0.0, 0.35),  # 12 translucent green
                  (0.0, 0.5, 0.5, 0.35),  # 13 translucent cyan
                  (0.0, 0.0, 1.0, 0.35),  # 14 translucent blue
                  (0.5, 0.0, 0.5, 0.35),  # 15 translucent magenta
                  (1.0, 1.0, 1.0, 0.0),   # 16 transparent white
                )

    # Initiate pyferret, but stay in python
    pyferret.init(None, False)
    for viewertype in ( "PyQtPipedViewer", ):
        print "Testing bindings for %s" % viewertype
        # Create a viewer window
        title = viewertype + "Tester"
        bindinst = pyferret.graphbind.createWindow(viewertype, title, True)
        # Resize the window to 500 x 500 pixels
        bininst.resizeWindow(500, 500)
        # Turn on anti-aliasing
        bindinst.setAntialias(True)
        # Create the one font that will be used here
        myfont = bindinst.createFont(None, 50, False, False, False)
        # Create a list of colors that will be used here
        mycolors = [ bindinst.createColor(r, g, b, a) \
                     for (r, g, b, a) in colorvals ]
        # Clear the window in opaque white
        bindinst.clearWindow(mycolors[1])
        # Create a view in the bottom left corner
        bindinst.beginView(0.0, 1.0, 0.5, 0.0, True)
        # Draw a translucent black rectangle over most of the view
        mybrush = bindinst.createBrush(mycolors[8], "solid")
        bindinst.drawRectangle(5, 495, 245, 245, mybrush, None)
        bindinst.deleteBrush(mybrush)
        # Draw a opaque blue polygon with solid black outline
        mybrush = bindinst.createBrush(mycolors[6], "solid")
        mypen = bindinst.createPen(mycolors[0], 5, "solid", "round", "round")
        bindinst.drawPolygon(mypentax, mypentay, mybrush, mypen)
        bindinst.deletePen(mypen)
        bindinst.deleteBrush(mybrush)
        # Draw some red text strings
        bindinst.drawText("y=480", 50, 480, myfont, mycolors[2], 0)
        bindinst.drawText("y=430", 50, 430, myfont, mycolors[2], 0)
        bindinst.drawText("y=380", 50, 380, myfont, mycolors[2], 0)
        bindinst.drawText("y=330", 50, 330, myfont, mycolors[2], 0)
        # End of this view
        bindinst.endView()
        # Window should already be shown, but just to make sure
        bindinst.showWindow(True)
        raw_input("Press Enter to continue")
        # Create a view of almost the whole window
        bindinst.beginView(0.25, 0.75, 1.0, 0.0, True)
        # Draw a translucent multicolor rectangle covering most of the window
        bindinst.drawMulticolorRectangle(130, 370, 495, 5, 2, 3, mycolors[10:])
        # Draw letters indicating the expected colors
        bindinst.drawText("R", 190, 120, myfont, mycolors[0], -45)
        bindinst.drawText("Y", 190, 300, myfont, mycolors[0], -45)
        bindinst.drawText("G", 310, 120, myfont, mycolors[0], -45)
        bindinst.drawText("C", 310, 300, myfont, mycolors[0], -45)
        bindinst.drawText("B", 430, 120, myfont, mycolors[0], -45)
        bindinst.drawText("M", 430, 300, myfont, mycolors[0], -45)
        # End of this view
        bindinst.endView()
        # Window should already be shown, but just to make sure
        bindinst.showWindow(True)
        raw_input("Press Enter to continue")
        # Create a view of the whole window
        bindinst.beginView(0.0, 1.0, 1.0, 0.0, True)
        # Draw points using various symbols
        ptsy = (50, 150, 250, 350, 450)
        ptsx = (100, 100, 100, 100, 100)
        mysymbol = bindinst.createSymbol(".")
        bindinst.drawPoints(ptsx, ptsy, mysymbol, mycolors[0], 20)
        bindinst.deleteSymbol(mysymbol)
        ptsx = (150, 150, 150, 150, 150)
        mysymbol = bindinst.createSymbol("o")
        bindinst.drawPoints(ptsx, ptsy, mysymbol, mycolors[0], 20)
        bindinst.deleteSymbol(mysymbol)
        ptsx = (200, 200, 200, 200, 200)
        mysymbol = bindinst.createSymbol("+")
        bindinst.drawPoints(ptsx, ptsy, mysymbol, mycolors[6], 20)
        bindinst.deleteSymbol(mysymbol)
        ptsx = (250, 250, 250, 250, 250)
        mysymbol = bindinst.createSymbol("x")
        bindinst.drawPoints(ptsx, ptsy, mysymbol, mycolors[0], 20)
        bindinst.deleteSymbol(mysymbol)
        ptsx = (300, 300, 300, 300, 300)
        mysymbol = bindinst.createSymbol("*")
        bindinst.drawPoints(ptsx, ptsy, mysymbol, mycolors[0], 20)
        bindinst.deleteSymbol(mysymbol)
        ptsx = (350, 350, 350, 350, 350)
        mysymbol = bindinst.createSymbol("^")
        bindinst.drawPoints(ptsx, ptsy, mysymbol, mycolors[6], 20)
        bindinst.deleteSymbol(mysymbol)
        ptsx = (400, 400, 400, 400, 400)
        mysymbol = bindinst.createSymbol("#")
        bindinst.drawPoints(ptsx, ptsy, mysymbol, mycolors[0], 20)
        bindinst.deleteSymbol(mysymbol)
        # Draw a white dash line between some of the points
        mypen = bindinst.createPen(mycolors[1], 3, "dash", "round", "round")
        ptsx = (350, 200, 400, 300, 150, 100)
        ptsy = ( 50, 150, 250, 350, 250, 450)
        bindinst.drawMultiline(ptsx, ptsy, mypen)
        bindinst.deletePen(mypen)
        # End of this view
        bindinst.endView()
        # Window should already be shown, but just to make sure
        bindinst.showWindow(True)
        raw_input("Press Enter to continue")
        try:
            while 1:
                bindinst.deleteColor(mycolors.pop())
        except IndexError:
            pass
        bindinst.deleteFont(myfont)
        bindinst.deleteWindow()
        print "Done with bindings for %s" % viewertype

