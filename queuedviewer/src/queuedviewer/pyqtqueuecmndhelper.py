import sip
sip.setapi('QVariant', 2)

from PyQt4.QtCore import Qt, QRectF
from PyQt4.QtGui import QBrush, QColor, QFont, QPen

class PyQtQueueCmndHelper(object):
    '''
    Helper methods for dealing with commands coming out of the queue of QueuedViewer.
    '''
    @staticmethod
    def getFontFromCommand(cmnd, keywd):
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

    @staticmethod
    def getBrushFromCommand(cmnd, keywd):
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
            mycolor = PyQtQueueCmndHelper.getColor(brushinfo)
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
                raise ValueError("Unknown brush style %s" % str(mystyle))
            mybrush.setStyle(mystyle)
        except KeyError:
            pass
        return mybrush

    @staticmethod
    def getPenFromCommand(cmnd, keywd):
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
            mycolor = PyQtQueueCmndHelper.getColor(peninfo)
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
                raise ValueError("Unknown pen style %s" % str(mystyle))
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
                raise ValueError("Unknown pen cap style %s" % str(mystyle))
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
                raise ValueError("Unknown pen join style %s" % str(mystyle))
            mypen.setJoinStyle(mystyle)
        except KeyError:
            pass
        return mypen

    @staticmethod
    def getRect(info):
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

    @staticmethod
    def getColor(info):
        '''
        Returns a QColor based on the information in the dictionary info.
        Raises a KeyError if the "color" key is not given. 
        
        Recognized keys are:
            "color": color name (string)
            "alpha": alpha value from 0 (transparent) to 255 (opaque)
        '''
        mycolor = QColor(info["color"])
        try:
            mycolor.setAlpha(int(info["alpha"]))
        except KeyError:
            pass
        return mycolor
