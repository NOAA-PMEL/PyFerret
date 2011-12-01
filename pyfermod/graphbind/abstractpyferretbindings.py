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

    def createWindow(self, title, width, height, visible):
        '''
        Creates a "Window object" for this graphics engine.  Here,
        a Window is the complete drawing area.  However, no drawing
        will be performed on a Window, only on Views (see beginView).
        Initializes the graphics engine if needed.

        Arguments:
           title: display title for the Window
           width: width of the Window, in inches
           height: height of the Window, in inches
           visible: display Window on start-up?
        
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

    def beginView(self, leftfrac, bottomfrac, rightfrac, topfrac,
                        leftcoord, bottomcoord, rightcoord, topcoord,
                        clipit):
        '''
        Creates a "View object" for the given Window.  Here, a View is a
        rectangular region in the drawing area with its own coordinate
        system.  The origin for windows and views is the bottom left corner
        (NOT the top left as in most modern GUIs).

        Arguments:
            leftfrac:    [0,1] fraction of the Window width
                         for the left side of the View
            bottomfrac:  [0,1] fraction of the Window height
                         for the bottom side of the View
            rightfrac:   [0,1] fraction of the Window width
                         for the right side of the View
            topfrac:     [0,1] fraction of the Window height
                         for the top side of the View
            leftcoord:   user coordinate
                         for the left side of the View
            bottomcoord: user coordinate
                         for the bottom side of the View
            rightcoord:  user coordinate
                         for the right side of the View
            topcoord:    user coordinate
                         for the top side of the View
            clipit:      clip drawing to this View?

        Note that leftfrac < rightfrac, bottomfrac < topfrac since the
        origin of Windows and Views is the bottom left corner.
        '''
        raise AttributeError()

    def clipView(self, clipit):
        '''
        Enable or disable clipping to the current View.

        Arguments:
            clipit: clip drawings to the current View?
        '''
        raise AttributeError()

    def endView(self):
        '''
        Closes the current View.  When this call returns, the graphics 
        drawn to the View should be visible in its Window.
        '''
        raise AttributeError()

    def updateWindow(self):
        '''
        Indicates the viewer should update the graphics displayed.
        '''
        raise AttributeError()

    def clearWindow(self, fillcolor):
        '''
        Clears the Window of all drawings.  The Window is filled
        (initialized) with fillcolor.
 
        Arguments:
            fillcolor: Color to fill (initialize) the Window
        '''
        raise AttributeError()

    def resizeWindow(self, width, height):
        '''
        Sets the current size of the Window.

        Arguments:
            width: width of the Window, in units of 0.001 inches
            height: height of the window in units of 0.001 inches
        '''
        raise AttributeError()

    def windowDpi(self):
        '''
        Returns a two-tuple containing the screen resolution 
        of the Window, in dots per inch, in the horizontal (X)
        and vertical (Y) directions.
        '''
        raise AttributeError()

    def showWindow(self, visible):
        '''
        Display or hide a Window.  A graphics engine that does not
        have the ability to display a Window will always raise an
        AttributeError.

        Arguments:
            visible: display (if True) or hide (if False) the Window
        '''
        raise AttributeError()

    def saveWindow(self, filename, fileformat, transparentbkg):
        '''
        Save the contents of the window to a file.

        Arguments:
            filename: name of the file to create
            fileformat: name of the format to use
            transparentbkg: should the background be transparent?

        If fileformat is NULL, the fileformat is guessed from the
        filename extension.
        '''
        raise AttributeError()

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
        raise AttributeError()

    def deleteColor(self, color):
        '''
        Delete a Color object created by createColor

        Arguments:
            color: Color to be deleted
        '''
        raise AttributeError()

    def createFont(self, familyname, fontsize, italic, bold, underlined):
        '''
        Returns a Font object.

        Arguments:
            familyname: name of the font family (e.g., "Helvetica", "Times")
            fontsize: desired size of the font
            italic: use the italic version of the font?
            bold: use the bold version of the font?
            underlined: use the underlined version of the font?

        Raises an error if unable to create the Font object.
        '''
        raise AttributeError()

    def deleteFont(self, font):
        '''
        Delete a Font object created by createFont

        Arguments:
            font: Font to be deleted
        '''
        raise AttributeError()

    def createPen(self, color, width, style, capstyle, joinstyle):
        '''
        Returns a Pen object.

        Arguments:
            color: Color to use
            width: line width (scales with the size of the View)
            style: line style name (e.g., "solid", "dash")
            capstyle: end-cap style name (e.g., "square")
            joinstyle: join style name (e.g., "bevel")

        Raises an error if unable to create the Pen object.
        '''
        raise AttributeError()

    def deletePen(self, pen):
        '''
        Delete a Pen object created by createPen

        Arguments:
            pen: Pen to be deleted
        '''
        raise AttributeError()

    def createBrush(self, color, style):
        '''
        Returns a Brush object.

        Arguments:
            color: Color to use
            style: fill style name (e.g., "solid", "cross")

        Raises an error if unable to create the Brush object.
        '''
        raise AttributeError()

    def deleteBrush(self, brush):
        '''
        Delete a Brush object created by createBrush

        Arguments:
            brush: Brush to be deleted
        '''
        raise AttributeError()

    def createSymbol(self, symbolname):
        '''
        Returns a Symbol object.

        Arguments:
            symbolname: name of the symbol (e.g., ".", "+")

        Raises an error if unable to create the Symbol object.
        '''
        raise AttributeError()

    def deleteSymbol(self, symbol):
        '''
        Delete a Symbol object created by createSymbol

        Arguments:
            symbol: Symbol to be deleted
        '''
        raise AttributeError()

    def drawMultiline(self, ptsx, ptsy, pen):
        '''
        Draws connected line segments.

        Arguments:
            ptsx: the X-coordinates of the points in View units
            ptsy: the Y-coordinates of the points in View units
            pen: the Pen to use to draw the line segments
        '''
        raise AttributeError()

    def drawPoints(self, ptsx, ptsy, symbol, color, ptsize):
        '''
        Draws discrete points.

        Arguments:
            ptsx: the X-coordinates of the points in View units
            ptsy: the Y-coordinates of the points in View units
            symbol: the Symbol to use to draw a point
            color: color of the Symbol
            ptsize: size of the Symbol (scales with the size of the View)
        '''
        raise AttributeError()

    def drawPolygon(self, ptsx, ptsy, brush, pen):
        '''
        Draws a polygon.

        Arguments:
            ptsx: the X-coordinates of the points in View units
            ptsy: the Y-coordinates of the points in View units
            brush: the Brush to use to fill the polygon; if None
                    the polygon will not be filled
            pen: the Pen to use to outline the polygon; if None
                    the polygon will not be outlined
        '''
        raise AttributeError()

    def drawRectangle(self, left, bottom, right, top, brush, pen):
        '''
        Draws a rectangle.

        Arguments:
            left: the X-coordinate of the left edge in View units
            bottom: the Y-coordinate of the bottom edge in View units
            right: the X-coordinate of the right edge in View units
            top: the Y-coordinate of the top edge in View units
            brush: the Brush to use to fill the polygon; if None
                    the polygon will not be filled
            pen: the Pen to use to outline the polygon; if None
                    the polygon will not be outlined
         '''
        raise AttributeError()

    def drawMulticolorRectangle(self, left, bottom, right, top,
                                numrows, numcols, colors):
        '''
        Draws a filled rectangle using an array of solid colors.
        The rectangle is divided into a given number of equally
        spaced rows and a number of equally spaced columns.  Each
        of these cells is then filled with a color (using a solid
        brush) from the corresponding element in an array of colors.

        Arguments:
            left: the X-coordinate of the left edge in View units
            bottom: the Y-coordinate of the bottom edge in View units
            right: the X-coordinate of the right edge in View units
            top: the Y-coordinate of the top edge in View units
            numrows: the number of equally spaced rows
                    to subdivide the rectangle into
            numcols: the number of equally spaced columns
                    to subdivide the rectangle into
            colors: iterable representing a flattened column-major
                    2-D array of colors specifying the color of the
                    corresponding cell.  The first row is at the top;
                    the first column is on the left.
        '''
        raise AttributeError()

    def drawText(self, text, startx, starty, font, color, rotate):
        '''
        Draws text.

        Arguments:
            text: the text string to draw
            startx: the X-coordinate of the beginning baseline
                    of the text in View units
            starty: the Y-coordinate of the beginning baseline
                    of the text in View units
            font: the font to use for the text
            color: the color to use (as a solid brush or pen)
                    for the text
            rotate: the angle of the baseline in degrees
                    clockwise from horizontal
        '''
        raise AttributeError()

