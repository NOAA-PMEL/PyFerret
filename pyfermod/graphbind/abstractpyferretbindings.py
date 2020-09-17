'''
Defines the abstract base class for bindings to the graphics calls
from PyFerret.  Specific engine bindings should create a subclass of
AbstractPyFerretBindings and re-implement the methods to call the
methods or functions for that engine.  This class definition should
then be registered with PyFerret (pyferret.graphbind.addPyFerretBindings).

When PyFerret creates a Window for an engine, it first creates an
instance of the appropriate bindings class and then calls the
"createWindow" method on that instance.  Thus instance variables
will be for the one window associated with the bindings instance.
'''

class AbstractPyFerretBindings(object):
    '''
    Abstract base class for providing bindings to graphics calls
    from PyFerret for a graphics engine.  The methods defined in
    this class should all be re-implemented in a subclass for
    proper PyFerret behavior.
    '''

    def __init__(self):
        '''
        When PyFerret creates a Window for an engine, it creates
        an instance of the appropriate bindings class, then calls
        the createWindow method of this instance created.  Thus
        instance variables will be for the one window associated
        with the bindings instance.
        '''
        super(AbstractPyFerretBindings, self).__init__()

    def createWindow(self, title, visible, noalpha, rasteronly):
        '''
        Creates a "Window object" for this graphics engine.  Here,
        a Window is the complete drawing area.  However, no drawing
        will be performed on a Window, only on Views (see beginView).
        Initializes the graphics engine if needed.

        The rasteronly option is for possible faster drawing by
        drawing directly to an image surface.  If true, deleting
        segments may not be supported.

        Arguments:
            title: display title for the Window
            visible: display Window on start-up?
            noalpha: do not use the alpha channel in colors?
            rasteronly: only raster images will be used ?

        Returns True if a Window was successfully created.
        '''
        return False

    def deleteWindow(self):
        '''
        Deletes the Window associated with this instance of the bindings.
        When this call returns True, the Window should not be visible,
        all resources associated with this Window should have been released,
        After calling this function, this instance of the bindings should
        be deleted and considered no longer usable.
        '''
        return False

    def setImageName(self, imagename, imgnamelen, formatname, fmtnamelen):
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

        This method only suggests the name of the image file to
        be created.  A file using the given name may or may not
        be open from this call.

        If a file was opened from this call (image data written
        to file as it is being drawn), the saveWindow method may
        not be supported.

        If a file was not opened from this call, the saveWindow
        method must be called to save the image.  Thus, the
        filename provided here may only be used as a default
        filename.
        '''
        raise AttributeError('not implemented')

    def setAntialias(self, antialias):
        '''
        Turns on (antilaias True) or off (antialias False) anti-aliasing
        in future drawing commands.  May not be implemented and thus raise
        an AttributeError.
        '''
        raise AttributeError('not implemented')

    def beginView(self, leftfrac, bottomfrac, rightfrac, topfrac,
                        clipit):
        '''
        Creates a "View object" for the given Window.  The view fractions
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
            clipit:      clip drawing to this View?
        '''
        raise AttributeError('not implemented')

    def clipView(self, clipit):
        '''
        Enable or disable clipping to the current View.

        Arguments:
            clipit: clip drawings to the current View?
        '''
        raise AttributeError('not implemented')

    def endView(self):
        '''
        Closes the current View.  When this call returns, the graphics
        drawn to the View should be visible in its Window.
        '''
        raise AttributeError('not implemented')

    def beginSegment(self, segid):
        '''
        Creates a "Segment object" for the given Window.
        A Segment is just a group of drawing commands.

        Arguments:
            segid: ID for the Segment
        '''
        raise AttributeError('not implemented')

    def endSegment(self):
        '''
        End the current "Segment" for the Window.
        '''
        raise AttributeError('not implemented')

    def deleteSegment(self, segid):
        '''
        Deletes the drawing commands in the indicated Segment.

        Arguments:
            segid: ID for the Segment to be deleted
        '''
        raise AttributeError('not implemented')

    def updateWindow(self):
        '''
        Indicates the viewer should update the graphics displayed.
        '''
        raise AttributeError('not implemented')

    def clearWindow(self, bkgcolor):
        '''
        Clears the Window of all drawings.  The window is
        initialized to all bkgcolor (the background color).

        Arguments:
            bkgcolor: initialize (fill) the Window with this Color
        '''
        raise AttributeError('not implemented')

    def redrawWindow(self, bkgcolor):
        '''
        Redraw the current drawing except using bkgcolor as the
        background color (the initialization color for the Window).

        Arguments:
            bkgcolor: initialize (fill) the Window with this Color
                      before redrawing the current drawing.
        '''
        raise AttributeError('not implemented')

    def windowScreenInfo(self):
        '''
        Returns the four-tuple (dpix, dpiy, screenwidth, screenheight) for
        the default screen (display) of this Window
           dpix: dots (pixels) per inch, in the horizontal (X) direction
           dpiy: dots (pixels) per inch, in the vertical (Y) direction
           screenwidth: width of the screen (display) in pixels (dots)
           screenheight: height of the screen (display) in pixels (dots)
        '''
        raise AttributeError('not implemented')

    def resizeWindow(self, width, height):
        '''
        Sets the current size of the Window.

        Arguments:
            width: width of the Window, in "device units"
            height: height of the window in "device units"

        "device units" is pixels at the current window DPI
        '''
        raise AttributeError('not implemented')

    def scaleWindow(self, scale):
        '''
        Sets the scaling factor for the Window.  If zero, switch to
        auto-scaling (automatically scales to best fit window size
        without changing aspect ratio).  If negative, scale using
        the absolute value and then switch to auto-scaling.

        Arguments:
            scale: scaling factor to use
        '''
        raise AttributeError('not implemented')

    def showWindow(self, visible):
        '''
        Display or hide a Window.  A graphics engine that does not
        have the ability to display a Window should ignore this call.

        Arguments:
            visible: display (if True) or hide (if False) the Window
        '''
        raise AttributeError('not implemented')

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

        If fileformat is NULL, the fileformat is guessed from the
        filename extension.

        If transparent is False, the entire scene is initialized
        to the last clearing color.  If transparent is True, the
        entire scene is initialized as transparent.

        If annotations is not None, the strings given in the tuple
        are to be displayed above the image.  These annotations add
        height, as needed, to the saved image (i.e., yinches or
        ypixels is the height of the image below these annotations).
        '''
        raise AttributeError('not implemented')

    def createColor(self, redfrac, greenfrac, bluefrac, opaquefrac):
        '''
        Returns a Color object from fractional [0.0, 1.0] intensities
        of the red, green, and blue channels.

        Arguments:
            redfrac: fractional [0.0, 1.0] red intensity
            greenfrac: fractional [0.0, 1.0] green intensity
            bluefrac: fractional [0.0, 1.0] blue intensity
            opaquefrac: fractional [0.0, 1.0] opaqueness
                (0.0 is transparent; 1.0 is opaque) of the color.
                If the graphics engine does not support this
                feature (alpha channel), this may be silently
                ignored and the color be completely opaque.

        Raises an error if unable to create the Color object.
        '''
        raise AttributeError('not implemented')

    def deleteColor(self, color):
        '''
        Delete a Color object created by createColor

        Arguments:
            color: Color to be deleted
        '''
        raise AttributeError('not implemented')

    def createFont(self, familyname, fontsize, italic, bold, underlined):
        '''
        Returns a Font object.

        Arguments:
            familyname: name of the font family (e.g., "Helvetica", "Times")
            fontsize: desired size of the font (scales with view size)
            italic: use the italic version of the font?
            bold: use the bold version of the font?
            underlined: use the underlined version of the font?

        Raises an error if unable to create the Font object.
        '''
        raise AttributeError('not implemented')

    def deleteFont(self, font):
        '''
        Delete a Font object created by createFont

        Arguments:
            font: Font to be deleted
        '''
        raise AttributeError('not implemented')

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
        raise AttributeError('not implemented')

    def replacePenColor(self, pen, newcolor):
        '''
        Replaces the color in pen with newcolor.

        Arguments:
            pen: Pen object to modify
            newcolor: Color to use

        Raises an error if unable to replace the Color in the Pen.
        '''
        raise AttributeError('not implemented')

    def deletePen(self, pen):
        '''
        Delete a Pen object created by createPen

        Arguments:
            pen: Pen to be deleted
        '''
        raise AttributeError('not implemented')

    def createBrush(self, color, style):
        '''
        Returns a Brush object.

        Arguments:
            color: Color to use
            style: fill style name (e.g., "solid", "cross")

        Raises an error if unable to create the Brush object.
        '''
        raise AttributeError('not implemented')

    def replaceBrushColor(self, brush, newcolor):
        '''
        Replaces the color in brush with newcolor.

        Arguments:
            brush: Brush object to modify
            newcolor: Color to use

        Raises an error if unable to replace the Color in the Brush.
        '''
        raise AttributeError('not implemented')

    def deleteBrush(self, brush):
        '''
        Delete a Brush object created by createBrush

        Arguments:
            brush: Brush to be deleted
        '''
        raise AttributeError('not implemented')

    def createSymbol(self, name, pts=None, fill=False):
        '''
        Returns a Symbol object associated with the given name.

        If pts is not given, the symbol name must already be known,
        either as a pre-defined symbol or from a previous call to
        this method.

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
        Returns a Symbol object.
        '''
        raise AttributeError('not implemented')

    def deleteSymbol(self, symbol):
        '''
        Delete a Symbol object created by createSymbol

        Arguments:
            symbol: Symbol to be deleted
        '''
        raise AttributeError('not implemented')

    def setWidthFactor(self, widthfactor):
        '''
        Assigns the scaling factor to be used for pen widths and symbols sizes

        Arguments:
            widthfactor: positive float giving the new scaling factor to use
        '''
        raise AttributeError('not implemented')

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
        raise AttributeError('not implemented')

    def drawPoints(self, ptsx, ptsy, symbol, color, ptsize, highlight):
        '''
        Draws discrete points.

        Arguments:
            ptsx: X-coordinates of the points
            ptsy: Y-coordinates of the points
            symbol: the Symbol to use to draw a point
            color: color of the symbol (default color if None or empty)
            ptsize: size of the symbol (scales with view size)
            highlight: outline color of the symbol (do not outline if None or empty)

        Coordinates are measured from the upper left corner
        in "device units" (pixels at the current window DPI).
        '''
        raise AttributeError('not implemented')

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
        raise AttributeError('not implemented')

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
        raise AttributeError('not implemented')

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
        raise AttributeError('not implemented')

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
        raise AttributeError('not implemented')

    def setWaterMark(self, filename, len_filename, xloc, yloc, scalefrac, opacity):
        '''
        Overlays watermark.

        Arguments:
            filename:     path to water mark image
            len_filename: number of characters in filename
            xloc:         horizontal position of upper left corner of watermark image
            yloc:         vertical position of upper left corner of watermark image
            scalefrac:    multiple of original image size to display plot as
            opacity:      image visibility in range [0.0,1.0] where 0->invisible, 1->opaque
        '''
        print(filename)
        print(xloc)
        print(yloc)
        print(scalefrac)
        print(opacity)
        raise AttributeError('not implented')
