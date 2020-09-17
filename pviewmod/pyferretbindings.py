'''
The PyFerretBindings class is a base class providing common
methods in PipedViewer bindings for PyFerret graphics methods.

The PViewerPQPyFerretBindings class is a subclass of PyFerretBindings
using PipedViewerPQ as the viewer.

The PImagerPQPyFerretBindings class is a subclass of PyFerretBindings
using PipedImagerPQ as the viewer.  Note that PipedImagerPQ only
displays completed images and does not understand many of the commands
(including all the drawing commands) given here.

This package was developed by the Thermal Modeling and Analysis Project
(TMAP) of the National Oceanographic and Atmospheric Administration's (NOAA)
Pacific Marine Environmental Lab (PMEL).
'''

from __future__ import print_function

import sys

from pyferret.graphbind.abstractpyferretbindings import AbstractPyFerretBindings
from pipedviewer import PipedViewer, WINDOW_CLOSED_MESSAGE


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

    def createPipedViewerWindow(self, viewertype, title, visible, noalpha):
        '''
        Creates a PipedViewer of viewertype as the window of this
        instance of the bindings.

        Arguments:
            viewertype: type of PipedViewer to use
            title: display title for the Window
            visible: display Window on start-up?
            noalpha: do not use the alpha channel in colors?

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
        if noalpha:
            self.__window.submitCommand( {"action":"noalpha"} )
        return True


    # The remaining methods are common implementations of the required binding methods

    def deleteWindow(self):
        '''
        Shuts down the PipedViewer.

        Returns True.
        '''
        try:
            closingremarks = self.__window.shutdownViewer()
        finally:
            self.__window = None
        return True

    def setImageName(self, imagename, formatname):
        '''
        Assigns the name and format of the image file to be created.

        Arguments:
            imagename  - name for the image file (can be NULL)
            imgnamelen - actual length of imagename (zero if NULL)
            formatname - name of the image format (case insensitive,
                         can be NULL)
            fmtnamelen - actual length of formatname (zero if NULL)

        If formatname is empty or NULL, the filename extension of
        imagename, if it exists and is recognized, will determine
        the format.

        This method only gives the default name of the image file
        to be created by the saveWindow method.  The saveWindow
        method must be called to save the image.
        '''
        cmnd = { "action":"imgname" }
        if imagename:
            cmnd["name"] = imagename
        if formatname:
            cmnd["format"] = formatname
        self.__window.submitCommand(cmnd)

    def setAntialias(self, antialias):
        '''
        Turns on (antilaias True) or off (antialias False) anti-aliasing
        in future drawing commands.
        '''
        cmnd = { "action":"antialias",
                 "antialias":bool(antialias) }
        self.__window.submitCommand(cmnd)

    def beginView(self, leftfrac, bottomfrac, rightfrac, topfrac, clipit):
        '''
        Start a view in the PipedViewer Window.  The view fractions
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

    def clipView(self, clipit):
        '''
        Enable or disable clipping to the current View.

        Arguments:
            clipit: clip drawings to the current View?
        '''
        cmnd = { "action":"clipView",
                 "clip":bool(clipit) }
        self.__window.submitCommand(cmnd)

    def endView(self):
        '''
        Close a View in the PipedViewer Window
        '''
        self.__window.submitCommand( { "action":"endView" } )

    def beginSegment(self, segid):
        '''
        Creates a "Segment object" for the given Window.
        A Segment is just a group of drawing commands.

        Arguments:
            segid: ID for the Segment
        '''
        cmnd = { "action":"beginSegment",
                 "segid":segid }
        self.__window.submitCommand(cmnd)

    def endSegment(self):
        '''
        End the current "Segment" for the Window.
        '''
        cmnd = { "action":"endSegment" }
        self.__window.submitCommand(cmnd)

    def deleteSegment(self, segid):
        '''
        Deletes the drawing commands in the indicated Segment.

        Arguments:
            segid: ID for the Segment to be deleted
        '''
        cmnd = { "action":"deleteSegment",
                 "segid":segid }
        self.__window.submitCommand(cmnd)

    def updateWindow(self):
        '''
        Indicates the viewer should update the graphics displayed.
        '''
        cmnd = { "action":"update" }
        self.__window.submitCommand(cmnd)

    def clearWindow(self, bkgcolor):
        '''
        Clears the Window of all drawings.  The window is
        initialized to all bkgcolor (the background color).

        Arguments:
            bkgcolor: initialize (fill) the Window with this Color
        '''
        if bkgcolor:
            # Make a copy of the bkgcolor dictionary
            cmnd = dict(bkgcolor)
        else:
            cmnd = { }
        cmnd["action"] = "clear"
        self.__window.submitCommand(cmnd)

    def redrawWindow(self, bkgcolor):
        '''
        Redraw the current drawing except using bkgcolor as the
        background color (the initialization color for the Window).

        Arguments:
            bkgcolor: initialize (fill) the Window with this Color
                      before redrawing the current drawing.
        '''
        if bkgcolor:
            # Make a copy of the bkgcolor dictionary
            cmnd = dict(bkgcolor)
        else:
            cmnd = { }
        cmnd["action"] = "redraw"
        self.__window.submitCommand(cmnd)

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

    def scaleWindow(self, scale):
        '''
        Sets the current scaling factor for the Window.

        Arguments:
            scale: scaling factor to use
        '''
        cmnd = { "action":"rescale",
                 "factor":scale }
        self.__window.submitCommand(cmnd)

    def windowScreenInfo(self):
        '''
        Returns the four-tuple (dpix, dpiy, screenwidth, screenheight) for
        the default screen (display) of this Window
           dpix: dots (pixels) per inch, in the horizontal (X) direction
           dpiy: dots (pixels) per inch, in the vertical (Y) direction
           screenwidth: width of the screen (display) in pixels (dots)
           screenheight: height of the screen (display) in pixels (dots)
        '''
        self.__window.blockErrMonitor()
        try:
            cmnd = { "action":"screenInfo" }
            self.__window.submitCommand(cmnd)
            response = None
            try:
                # Wait indefinitely for a response
                # Make sure it is a valid response
                response = self.__window.checkForResponse(None)
                if (type(response) != tuple) or (len(response) != 4):
                    raise ValueError
                dpix = float(response[0])
                dpiy = float(response[1])
                screenwidth = int(response[2])
                screenheight = int(response[3])
                if (dpix <= 0.0) or (dpiy <= 0.0) or \
                   (screenwidth <= 0) or (screenheight <= 0):
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
        finally:
            self.__window.resumeErrMonitor()
        return (dpix, dpiy, screenwidth, screenheight)

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

    def saveWindow(self, filename, fileformat, transparent,
                   xinches, yinches, xpixels, ypixels, annotations):
        '''
        Save the contents of the window to a file.  This might be called
        when there is no image to save; in this case the call should be
        ignored.

        Arguments:
            filename: name of the file to create
            fileformat: name of the format to use
            transparent: use a transparent background?
            xinches: horizontal size of vector image in inches
            yinches: vertical size of vector image in inches
            xpixels: horizontal size of raster image in pixels
            ypixels: vertical size of raster image in pixels
            annotations: tuple of annotation strings

        If fileformat is None or empty, the fileformat
        is guessed from the filename extension.

        If transparent is False, the entire scene is initialized
        to the last clearing color.  If transparent is True, the
        entire scene is initialized as transparent.

        If annotations is not None, the strings given in the tuple
        are to be displayed above the image.  These annotations add
        height, as needed, to the saved image (i.e., yinches or
        ypixels is the height of the image below these annotations).
        '''
        cmnd = { }
        cmnd["action"] = "save"
        cmnd["filename"] = filename
        if fileformat:
            cmnd["fileformat"] = fileformat
        cmnd["transparent"] = transparent
        cmnd["vectsize"] = { "width":xinches, "height":yinches }
        cmnd["rastsize"] = { "width":xpixels, "height":ypixels }
        cmnd["annotations"] = annotations
        self.__window.submitCommand(cmnd)

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

    def replacePenColor(self, pen, newcolor):
        '''
        Replaces the color in pen with newcolor.

        Arguments:
            pen: Pen object to modify
            newcolor: Color to use

        Raises an error if unable to replace the Color in the Pen.
        '''
        pen.update(newcolor)

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

    def replaceBrushColor(self, brush, newcolor):
        '''
        Replaces the color in brush with newcolor.

        Arguments:
            brush: Brush object to modify
            newcolor: Color to use

        Raises an error if unable to replace the Color in the Brush.
        '''
        brush.update(newcolor)

    def deleteBrush(self, brush):
        '''
        Delete a Brush object created by createBrush

        Arguments:
            brush: Brush to be deleted
        '''
        del brush

    def createSymbol(self, name, pts=None, fill=False):
        '''
        Returns a Symbol object associated with the given name.

        If pts is not given, the symbol name must already be known,
        either as a pre-defined symbol or from a previous call to
        this method.

        Current pre-defined symbol names are ones involving circles:
            'dot': very small filled circle
            'dotex': very small filled circle and outer lines of an ex mark
            'dotplus': very small filled circle and outer lines of a plus mark
            'circle': unfilled circle
            'circex': small unfilled circle and outer lines of an ex mark
            'circplus': small unfilled circle and outer lines of a plus mark

        If pts is given, the value is coordinates that define the symbol
        as multiline subpaths in a [-50,50] square.  The location of the
        point this symbol represents will be at the center of the square.
        An invalid coordinate (outside [-50,50]) will terminate the current
        subpath, and the next valid coordinate will start a new subpath.
        This definition will replace an existing symbol with the given name.

        Arguments:
            name: (string) name of the symbol
            pts:  (sequence of pairs of floats) vertex coordinates
            fill: (bool) color-fill symbol?

        Raises an error
            if name is not a string,
            if pts, if not None, is not a sequence of pairs of numbers, or
            if unable to create the Symbol object for any other reason.
        '''
        # If no points, assume the symbol is already defined but do not waste
        # time validating this.  The symbol object is just the symbol name.
        if not isinstance(name, str):
            raise RuntimeException('symbol name is not a string')
        if pts is None:
            return name
        # Send the symbol definition to the viewer engine
        self.__window.blockErrMonitor()
        try:
            cmnd = { 'action': 'createSymbol',
                     'name': name,
                     'pts': pts,
                     'fill': fill }
            self.__window.submitCommand(cmnd)
            response = None
            try:
                # Wait indefinitely for a response.
                # The valid response is the name of the symbol.
                response = self.__window.checkForResponse(None)
                if response != name:
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
        finally:
            self.__window.resumeErrMonitor()
        # Since this symbol is now defined in the viewer engine,
        # all we need is the name of this new symbol.
        return name

    def deleteSymbol(self, symbol):
        '''
        Delete a Symbol object created by createSymbol

        Arguments:
            symbol: Symbol to be deleted
        '''
        # TODO: delete the definition in the viewer engine
        del symbol

    def setWidthFactor(self, widthfactor):
        '''
        Assigns the scaling factor to be used for pen widths,
        symbols sizes, and font sizes

        Arguments:
            widthfactor: positive float giving the new scaling factor to use
        '''
        newfactor = float(widthfactor)
        if newfactor <= 0.0:
            raise ValueError("Width scaling factor must be positive")
        cmnd = { "action":"setWidthFactor",
                 "factor":widthfactor }
        self.__window.submitCommand(cmnd)

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

    def drawPoints(self, ptsx, ptsy, symbol, color, ptsize, highlight):
        '''
        Draws discrete points.

        Arguments:
            ptsx: X-coordinates of the points
            ptsy: Y-coordinates of the points
            symbol: the Symbol to use to draw a point
            color: color of the Symbol (default color if None)
            ptsize: size of the symbol (scales with view size)
            highlight: color to outline the symbol; not outlined if None

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
        if highlight:
            cmnd["highlight"] = highlight
        self.__window.submitCommand(cmnd)

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

    def textSize(self, text, font):
        '''
        Returns the width and height of the text if drawn in the given font.
        The width is such that continuing text should be positioned at the
        start of this text plus this width.  The height will always be the
        ascent plus descent for the font and is independent of the text.

        Arguments:
            text: the text string to draw
            font: the font to use

        Returns: (width, height) of the text in "device units"
              (pixels at the current window DPI)
        '''
        self.__window.blockErrMonitor()
        try:
            cmnd = { "action":"textSize", "text":text }
            if font:
                cmnd["font"] = font
            self.__window.submitCommand(cmnd)
            response = None
            try:
                # Wait indefinitely for a response
                # Make sure it is a valid response
                response = self.__window.checkForResponse(None)
                if (type(response) != tuple) or (len(response) != 2):
                    raise ValueError
                width = float(response[0])
                height = float(response[1])
                if (width <= 0.0) or (height <= 0.0):
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
        finally:
            self.__window.resumeErrMonitor()
        return (width, height)

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

    def setWaterMark(self, filename, len_filename, xloc, yloc, scalefrac, opacity):
        '''
        Overlays water mark.

        Arguments:
            filename:     path to water mark image
            len_filename: number of characters in filename
            xloc:         horizontal position of upper left corner of watermark image
            yloc:         vertical position of upper left corner of watermark image
            scalefrac:    multiple of original image size to display plot as
            opacity:      image visibility in range [0.0,1.0] where 0->invisible, 1->opaque
        '''
        cmnd = { "action":"setWaterMark",
                 "filename":filename,
                 "xloc":xloc,
                 "yloc":yloc,
                 "scalefrac":scalefrac,
                 "opacity":opacity }
        self.__window.submitCommand(cmnd)


class PViewerPQPyFerretBindings(PyFerretBindings):
    '''
    PyFerretBindings using a PipedViewerPQ as the viewer.
    '''

    def createWindow(self, title, visible, noalpha, rasteronly):
        '''
        Creates PyFerret bindings using a PipedViewerPQ.

        Arguments:
            title: display title for the Window
            visible: display Window on start-up?
            noalpha: do not use the alpha channel in colors?
            rasteronly: ignored

        Raises a RuntimeError if an active window is already associated
        with these bindings, or if there were problems with creating
        the window.

        Returns True.
        '''
        result = self.createPipedViewerWindow("PipedViewerPQ",
                                       title, visible, noalpha)
        return result


class PImagerPQPyFerretBindings(PyFerretBindings):
    '''
    PyFerretBindings using PipedImagerPQ as the viewer.

    Note that PipedImagerPQ only displays completed images
    and at this time does not understand many of the commands
    (including all the drawing commands) given in the base
    class PyFerretBindings.  However, the associated methods
    were left as-is in case future versions did implement
    these commands.

    The additional method newSceneImage sends image data
    for the new scene to be displayed.
    '''

    def createWindow(self, title, visible, noalpha, rasteronly):
        '''
        Creates PyFerret bindings using a PipedImagerPQ.

        Arguments:
            title: display title for the Window
            visible: display Window on start-up?
            noalpha: do not use the alpha channel in colors?
            rasteronly: ignored

        Raises a RuntimeError if an active window is already associated
        with these bindings, or if there were problems with creating
        the window.

        Returns True.
        '''
        result = self.createPipedViewerWindow("PipedImagerPQ",
                                       title, visible, noalpha)
        return result


    def newSceneImage(self, width, height, stride, imagedata):
        '''
        Create and display the scene created from the given
        image data.

        Arguments:
            width: width of the scene in pixels
            height: height of the scene in pixels
            stride: number of bytes in a single row of the image
            imagedata: a bytearray of the image data given in
                       premultiplied ARGB32 format in native
                       btye order
        '''
        lenimgdata = stride * height
        if len(imagedata) < lenimgdata:
            raise RuntimeError("newSceneImage: imagedata is too short")
        cmnd = { "action":"newImage",
                 "width":width,
                 "height":height,
                 "stride":stride }
        self.__window.submitCommand(cmnd)
        blocksize = 8192
        numblocks = (lenimgdata + blocksize - 1) // blocksize
        for k in range(numblocks):
            if k < (numblocks - 1):
                blkdata = imagedata[k*blocksize:(k+1)*blocksize]
            else:
                blkdata = imagedata[k*blocksize:]
            cmnd = { "action":"newImage",
                     "blocknum":k+1,
                     "numblocks":numblocks,
                     "startindex":k*blocksize,
                     "blockdata":blkdata }
            self.__window.submitCommand(cmnd)


#
# The following is for testing this module
#

def _test_pyferretbindings():
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
                  (1.0, 1.0, 0.0, 1.0),   #  3 opaque yellowish
                  (0.0, 1.0, 0.0, 1.0),   #  4 opaque green
                  (0.0, 1.0, 1.0, 1.0),   #  5 opaque cyan
                  (0.0, 0.0, 1.0, 1.0),   #  6 opaque blue
                  (1.0, 0.0, 1.0, 1.0),   #  7 opaque magenta
                  (0.0, 0.0, 0.0, 0.5),   #  8 translucent black
                  (1.0, 1.0, 1.0, 0.5),   #  9 translucent white
                  (1.0, 0.0, 0.0, 0.5),   # 10 translucent red
                  (1.0, 1.0, 0.0, 0.5),   # 11 translucent yellowish
                  (0.0, 1.0, 0.0, 0.5),   # 12 translucent green
                  (0.0, 1.0, 1.0, 0.5),   # 13 translucent cyan
                  (0.0, 0.0, 1.0, 0.5),   # 14 translucent blue
                  (1.0, 0.0, 1.0, 0.5),   # 15 translucent magenta
                  (1.0, 1.0, 1.0, 0.0),   # 16 transparent "white"
                )

    # Initiate pyferret, but stay in python
    pyferret.init(arglist=['-nojnl'], enterferret=False)
    for viewertype in ( "PipedViewerPQ", ):
        print("Testing bindings for %s" % viewertype)
        # Create a viewer window
        wintitle = viewertype + "Tester"
        bindinst = pyferret.graphbind.createWindow(engine_name=viewertype, title=wintitle, visible=True, noalpha=False, rasteronly=False)
        # Resize the window to 500 x 500 pixels
        bindinst.resizeWindow(500, 500)
        # Turn on anti-aliasing
        bindinst.setAntialias(True)
        # Create the one font that will be used here
        myfont = bindinst.createFont(None, 50, False, False, False)
        # Create a list of colors that will be used here
        mycolors = [ bindinst.createColor(r, g, b, a) \
                     for (r, g, b, a) in colorvals ]
        # Clear the window in black
        bindinst.clearWindow(mycolors[0])
        bindinst.showWindow(True)
        if sys.version_info[0] > 2:
            input("Press Enter to continue")
        else:
            raw_input("Press Enter to continue")
        # Create a view in the bottom left corner
        bindinst.beginView(0.0, 1.0, 0.5, 0.0, True)
        # Draw a translucent green rectangle over most of the view
        mybrush = bindinst.createBrush(mycolors[12], "solid")
        bindinst.drawRectangle(5, 495, 245, 245, mybrush, None)
        bindinst.deleteBrush(mybrush)
        # Draw a blue polygon with solid black outline
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
        if sys.version_info[0] > 2:
            input("Press Enter to continue")
        else:
            raw_input("Press Enter to continue")
        # Create a view of the whole window
        bindinst.beginView(0.0, 1.0, 1.0, 0.0, True)
        # Draw magenta points using various symbols
        ptsy = (50, 150, 250, 350, 450)
        ptsx = (100, 100, 100, 100, 100)
        mysymbol = bindinst.createSymbol("dot")
        bindinst.drawPoints(ptsx, ptsy, mysymbol, mycolors[7], 20, None)
        bindinst.deleteSymbol(mysymbol)
        ptsx = (150, 150, 150, 150, 150)
        mysymbol = bindinst.createSymbol("circle")
        bindinst.drawPoints(ptsx, ptsy, mysymbol, mycolors[7], 20, None)
        bindinst.deleteSymbol(mysymbol)
        ptsx = (200, 200, 200, 200, 200)
        mysymbol = bindinst.createSymbol("dotplus")
        bindinst.drawPoints(ptsx, ptsy, mysymbol, mycolors[7], 20, None)
        bindinst.deleteSymbol(mysymbol)
        ptsx = (250, 250, 250, 250, 250)
        mysymbol = bindinst.createSymbol("circplus")
        bindinst.drawPoints(ptsx, ptsy, mysymbol, mycolors[7], 20, None)
        bindinst.deleteSymbol(mysymbol)
        ptsx = (300, 300, 300, 300, 300)
        mysymbol = bindinst.createSymbol("dotex")
        bindinst.drawPoints(ptsx, ptsy, mysymbol, mycolors[7], 20, None)
        bindinst.deleteSymbol(mysymbol)
        ptsx = (350, 350, 350, 350, 350)
        mysymbol = bindinst.createSymbol("circex")
        bindinst.drawPoints(ptsx, ptsy, mysymbol, mycolors[7], 20, None)
        bindinst.deleteSymbol(mysymbol)
        ptsx = (400, 400, 400, 400, 400)
        mysymbol = bindinst.createSymbol(name="filledtriangle",
                            pts=( (-40.0, -30.0), (0.0, 39.282), (40.0, -30.0), (-40.0, -30.0), ),
                            fill=True)
        bindinst.drawPoints(ptsx, ptsy, mysymbol, mycolors[7], 20, None)
        bindinst.deleteSymbol(mysymbol)
        ptsx = (450, 450, 450, 450, 450)
        mysymbol = bindinst.createSymbol("bararrow",
                            ( (-50,50), (-10,10),
                              (-999, -999),
                              (50,0), (50,50), (0,50),
                              (-999, -999),
                              (0,-10), (20,-30), (10,-30), (10,-50), (-10,-50), (-10,-30), (-20,-30), (0,-10), ),
                            False)
        bindinst.drawPoints(ptsx, ptsy, mysymbol, mycolors[7], 20, None)
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
        if sys.version_info[0] > 2:
            input("Press Enter to continue")
        else:
            raw_input("Press Enter to continue")
        try:
            while 1:
                bindinst.deleteColor(mycolors.pop())
        except IndexError:
            pass
        bindinst.deleteFont(myfont)
        bindinst.deleteWindow()
        print("Done with bindings for %s" % viewertype)

if __name__ == "__main__":
    _test_pyferretbindings()
