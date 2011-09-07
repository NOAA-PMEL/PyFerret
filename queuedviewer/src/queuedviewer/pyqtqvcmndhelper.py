'''
PyQtQVCmndHelper is a helper class for dealing with command coming from
the queue of a PyQtQueuedViewer.

This package was developed by the Thermal Modeling and Analysis Project
(TMAP) of the National Oceanographic and Atmospheric Administration's (NOAA)
Pacific Marine Environmental Lab (PMEL).
'''

import sip
sip.setapi('QVariant', 2)

from PyQt4.QtCore import QPointF, QRectF, Qt
from PyQt4.QtGui import QBrush, QColor, QFont, QPen, QPolygonF

class PyQtQVCmndHelper(object):
    '''
    Helper class for dealing with commands
    coming from the queue of a PyQtQueuedViewer.
    '''

    def __init__(self, scene, gritems):
        '''
        Create the helper for the PyQtViewer which has the
        QGraphicsScene scene and dictionary gritems whose
        values are lists of QGraphicsItems.
        '''
        self.__scene = scene
        self.__gritems = gritems

    def addSimpleText(self, cmnd):
        '''
        Add a "simple" text item to the viewer.  Raises a KeyError if either
        the "text" or "id" key is not given.

        Recognized keys from cmnd:
            "id": identification string for this graphics item
            "text": string to displayed
            "font": dictionary describing the font to use; 
                    see PyQtQVCmndHelper.getFontFromCommand
            "fill": dictionary describing the brush used to draw the text; 
                    see PyQtQVCmndHelper.getBrushFromCommand
            "outline": dictionary describing the pen used to outline the text; 
                       see PyQtQVCmndHelper.getPenFromCommand
            "location": (x,y) location for the start of text in pixels 
                        from the upper left corner of the scene
        '''
        myid = cmnd["id"]
        # If this is a new ID, create an empty list of graphics items
        # associated with this ID.
        if self.__gritems.get(myid) == None:
            self.__gritems[myid] = [ ]
        # Create the simple text graphics item
        try:
            myfont = self.getFontFromCommand(cmnd, "font")
        except KeyError:
            myfont = QFont()
        mygrtext = self.__scene.addSimpleText(cmnd["text"], myfont)
        try:
            mybrush = self.getBrushFromCommand(cmnd, "fill")
            mygrtext.setBrush(mybrush)
        except KeyError:
            pass
        try:
            mypen = self.getPenFromCommand(cmnd, "outline")
            mygrtext.setPen(mypen)
        except KeyError:
            pass
        try:
            (x, y) = cmnd["location"] 
            mygrtext.translate(x, y)
        except KeyError:
            pass
        # Add this graphics item to list associated with the given ID
        self.__gritems[myid].append(mygrtext)

    def addPolygon(self, cmnd):
        '''
        Adds a polygon item to the viewer.  Raises a KeyError if either
        the "points" or the "id" key is not given.

        Recognized keys from cmnd:
            "id": identification string for this graphics item
            "points": the vertices of the polygon as a list of (x,y) points
                      of pixel values from the upper left corner of the scene
            "fill": dictionary describing the brush used to fill the polygon; 
                    see PyQtQVCmndHelper.getBrushFromCommand
            "outline": dictionary describing the pen used to outline the polygon; 
                       see PyQtQVCmndHelper.getPenFromCommand
            "offset": (x,y) offset, in pixels from the upper left corner of 
                      the scene, for the polygon
        '''
        myid = cmnd["id"]
        # If this is a new ID, create an empty list of graphics items
        # associated with this ID.
        if self.__gritems.get(myid) == None:
            self.__gritems[myid] = [ ]
        # Create the polygon graphics item
        mypoints = cmnd["points"]
        mypolygon = QPolygonF([ QPointF(x,y) for (x,y) in mypoints ])
        try:
            (x, y) = cmnd["offset"]
            mypolygon.translate(x, y)
        except KeyError:
            pass
        try:
            mypen = self.getPenFromCommand(cmnd, "outline")
        except KeyError:
            mypen = QPen()
        try:
            mybrush = self.getBrushFromCommand(cmnd, "fill")
        except KeyError:
            mybrush = QBrush()
        mygrpolygon = self.__scene.addPolygon(mypolygon, mypen, mybrush)
        # Add this graphics item to list associated with the given ID
        self.__gritems[myid].append(mygrpolygon)

    def getFontFromCommand(self, cmnd, keywd):
        '''
        Returns a QFont based on the information in the dictionary given by
        cmnd[keywd].  A KeyError is raised if the keywd key is not present.
        
        Recognized keys in the font dictionary are:
            "family": font family name (string)
            "size": point size (integer)
            "italic": italicize? (False/True)
            "bold": make bold? (False/True)
            "underline": underline?  (False/True)
        '''
        fontinfo = cmnd[keywd]
        # Customize the font as described in the font dictionary
        try:
            myfont = QFont(fontinfo["family"])
        except KeyError:
            myfont = QFont()
        try:
            myfont.setPointSize(fontinfo["size"])
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

    def getBrushFromCommand(self, cmnd, keywd):
        '''
        Returns a QBrush based on the information in the dictionary given by
        cmnd[keywd].  A KeyError is raise if the keywd key is not present.
        
        Recognized keys in the fill dictionary are:
            "color": color name (string)
            "alpha": alpha value from 0 (transparent) to 255 (opaque)
            "style": brush style name ("solid", "dense1" to "dense7", "none",
                         "hor", "ver", "cross", "bdiag", "fdiag", "diagcross")
        '''
        brushinfo = cmnd[keywd]
        try:
            mycolor = self.getColor(brushinfo)
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
                raise ValueError( self.__scene.tr("Unknown brush style %1").arg(str(mystyle)) )
            mybrush.setStyle(mystyle)
        except KeyError:
            pass
        return mybrush

    def getPenFromCommand(self, cmnd, keywd):
        '''
        Returns a QPen based on the information in the dictionary given by
        cmnd[keywd].  A KeyError is raised if the keywd key is not present.
        
        Recognized keys in the outline dictionary are:
            "color": color name (string)
            "alpha": alpha value from 0 (transparent) to 255 (opaque)
            "width": pen width (integer)
            "style": pen style name ("solid", "dash", "dot", "dashdot", "dashdotdot")
            "capstyle": pen cap style name ("square", "flat", "round")
            "joinstyle": pen join style name ("bevel", "miter", "round")
        '''
        peninfo = cmnd[keywd]
        try:
            mycolor = self.getColor(peninfo)
            mypen = QPen(mycolor)
        except KeyError:
            mypen = QPen()
        try:
            mypen.setWidth(peninfo["width"])
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
                raise ValueError( self.__scene.tr("Unknown pen style %1").arg(str(mystyle)) )
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
                raise ValueError( self.__scene.tr("Unknown pen cap style %1").arg(str(mystyle)) )
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
                raise ValueError( self.__scene.tr("Unknown pen join style %1").arg(str(mystyle)) )
            mypen.setJoinStyle(mystyle)
        except KeyError:
            pass
        return mypen

    def getRect(self, info):
        '''
        Returns a QRectF based on the information in the dictionary info.
        Recognized keys are "x", "y", "width", and "height", and correspond
        to those float values in the QRectF.  Values not given in info are 
        assigned as zero in the returned QRectF.
        '''
        myrect = QRectF(0.0, 0.0, 0.0, 0.0)
        try:
            myrect.setX(float(info["x"]))
        except KeyError:
            pass
        try:
            myrect.setY(float(info["y"]))
        except KeyError:
            pass
        try:
            myrect.setWidth(float(info["width"]))
        except KeyError:
            pass
        try:
            myrect.setHeight(float(info["height"]))
        except KeyError:
            pass
        return myrect

    def getColor(self, info):
        '''
        Returns a QColor based on the information in the dictionary info.
        Raises a KeyError if the "color" key is not given. 
        
        Recognized keys are:
            "color": color name or 24-bit hex RGB value
            "alpha": alpha value from 0 (transparent) to 255 (opaque)
        '''
        mycolor = QColor(info["color"])
        if not mycolor.isValid():
            raise ValueError( self.__scene.tr("Invalid color '%1'").arg(str(info["color"])) )
        try:
            mycolor.setAlpha(int(info["alpha"]))
        except KeyError:
            pass
        return mycolor
