'''
PyGtkCmndHelper is a helper class for dealing with commands
sent to a PyGtk piped viewer.

This package was developed by the Thermal Modeling and Analysis
Project (TMAP) of the National Oceanographic and Atmospheric
Administration's (NOAA) Pacific Marine Environmental Lab (PMEL).
'''

import pygtk
pygtk.require('2.0')
import gtk.gdk


class SizeF(object):
    '''
    Trivial helper class for defining floating point width and height values.
    '''
    def __init__(self, width, height):
        '''
        Create a SizeF with the given width and height
        '''
        super(SizeF, self).__init__()
        self.__width = float(width)
        self.__height = float(height)

    def width(self):
        '''
        Return the width as a float
        '''
        return self.__width

    def setWidth(self, val):
        '''
        Set the width as a float value of the argument.
        '''
        self.__width = float(val)

    def height(self):
        '''
        Return the height as a float
        '''
        return self.__height

    def setHeight(self, val):
        '''
        Set the height as a float value of the argument.
        '''
        self.__height = float(val)


class RectF(object):
    '''
    Trivial helper class for defining a rectangle with floating point
    values for the left, top, width, and height.
    '''
    def __init__(self, left, top, width, height):
        '''
        Create a RectF with the given left, top, width,
        and height as float values.
        '''
        super(RectF, self).__init__()
        self.__left = float(left)
        self.__top = float(top)
        self.__width = float(width)
        self.__height = float(height)

    def left(self):
        '''
        Return the left value as a float.
        '''
        return self.__left

    def setLeft(self, val):
        '''
        Set the RectF left as a float value of the argument.
        '''
        self.__left = float(val)

    def top(self):
        '''
        Return the top value as a float.
        '''
        return self.__top

    def setTop(self, val):
        '''
        Set the RectF top as a float value of the argument.
        '''
        self.__top = float(val)

    def width(self):
        '''
        Return the width value as a float.
        '''
        return self.__width

    def setWidth(self, val):
        '''
        Set the RectF right as a float value of the argument.
        '''
        self.__width = float(val)

    def height(self):
        '''
        Return the height value as a float.
        '''
        return self.__height

    def setHeight(self, val):
        '''
        Set the RectF height as a float value of the argument.
        '''
        self.__height = float(val)


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


class SimpleTransform(object):
    '''
    Helper class to perform simple coordinate transformations
    '''
    def __init__(self, sx, sy, dx, dy):
        super(SimpleTransform, self).__init__()
        self.__sx = sx
        self.__sy = sy
        self.__dx = dx
        self.__dy = dy

    def transform(self, userx, usery):
        devx = self.__sx * userx + self.__dx
        devy = self.__sy * usery + self.__dy
        return (devx, devy)


class PyGtkCmndHelper(object):
    '''
    Helper class of static methods for dealing with commands
    sent to a PyGtk piped viewer.
    '''
    def __init__(self, viewer, colormap):
        '''
        Creates a cmndpipe command helper.  The viewer argument is
        used for determining default resource values.  The colormap
        argument is the gtk.gdk.Colormap used for allocating colors.
        '''
        self.__viewer = viewer
        self.__colormap = colormap

    def getSizeFromCmnd(self, sizeinfo):
        '''
        Returns a SizeF based on the information in the dictionary
        sizeinfo.  Recognized keys are "width" and "height", and
        correspond to those float values in the SizeF.  Values not
        given in sizeinfo are assigned as zero in the returned SizeF.
        '''
        myrect = SizeF(0.0, 0.0)
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
        Returns a SidesRectF based on the information in the dictionary
        rectinfo.  Recognized keys are "left", "top", "right", and "bottom",
        and correspond to those float values in the SidesRectF.  Values not
        given in rectinfo are assigned as zero in the returned SidesRectF.
        '''
        myrect = SidesRectF(0.0, 0.0, 0.0, 0.0)
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
        Returns an allocated Color based on the information
        in the dictionary colorinfo.
        
        Raises:
            KeyError if the "color" key is not given
            ValueError if the "color" key specification is invalid

        Recognized keys are:
            "color": color name (eg, "white", "#FF0088") or 
                     a 24-bit RGB integer value (eg, 0xFF0088)
        '''
        colordata = colorinfo["color"]
        if isinstance(colordata, str):
            mycolor = self.__colormap.alloc_color(colordata)
        elif isinstance(colordata, int):
            (colordata, blueint) = divmod(colordata, 256)
            (colordata, greenint) = divmod(colordata, 256)
            (colordata, redint) = divmod(colordata, 256)
            # Gtk color components range 0 -> 65535
            mycolor = self.__colormap.alloc_color( gtk.gdk.Color(
                            redint*256, greenint*256, blueint*256) )
        else:
            raise ValueError("color value is neither a string nor an integer")
        return mycolor

    def getDefaultScreenDpis(self):
        '''
        Returns the resolution of the system default screen in dots
         (pixels) per inch in the horizontal and vertical directions.
        '''
        pixwidth = float( gtk.gdk.screen_width() )
        mmwidth = float( gtk.gdk.screen_width_mm() )
        dpix = 25.4 * pixwidth / mmwidth 
        pixheight = float( gtk.gdk.screen_height() )
        mmheight = float(gtk.gdk.screen_height_mm() )
        dpiy = 25.4 * pixheight / mmheight
        return (dpix, dpiy)
