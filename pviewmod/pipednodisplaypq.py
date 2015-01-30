'''
PipedNoDisplayPQ is a graphics application written in PyQt4
that receives its drawing and other commands primarily from another
application through a pipe.  A limited number of commands are
provided by the viewer itself to allow saving and some manipulation
of the displayed image.  The controlling application, however, may
be unaware of these modifications made to the image.

PipedNoDisplayPQProcess is used to create and run a PipedNoDisplayPQ.

This package was developed by the Thermal Modeling and Analysis
Project (TMAP) of the National Oceanographic and Atmospheric
Administration's (NOAA) Pacific Marine Environmental Lab (PMEL).
'''

import sip
from time import sleep
try:
    sip.setapi('QVariant', 2)
except AttributeError:
    pass

from PyQt4.QtCore import Qt, QCoreApplication, QObject, QPointF, QRect, QRectF, \
                         QSize, QSizeF, QTimer
from PyQt4.QtGui  import QBrush, QColor, QImage, QPainter, QPen, \
                         QPicture, QPolygonF, QPrinter, QTextDocument

try:
    from PyQt4.QtSvg  import QSvgGenerator
    HAS_QSvgGenerator = True
except ImportError:
    HAS_QSvgGenerator = False

from cmndhelperpq import CmndHelperPQ
from multiprocessing import Pipe, Process
import sys
import time
import os
import signal
import math

class PipedNoDisplayPQ(QObject):
    '''
    A PyQt graphics engine that receives generic drawing commands
    through a pipe.  Uses a list of QPictures to record the drawings
    which are then used to manipulate and save the image.

    A drawing command is a dictionary with string keys that will be
    interpreted into the appropriate PyQt command(s).  For example,
        { "action":"drawText",
          "text":"Hello",
          "font":{"family":"Times", "size":100, "italic":True},
          "fill":{"color":0x880000, "style":"cross"},
          "outline":{"color":"black"},
          "location":(250,350) }

    The command { "action":"exit" } will shutdown the viewer and is
    the only way the viewer can be closed.
    '''

    def __init__(self, cmndpipe, rspdpipe):
        '''
        Create a PyQt no-display graphics engine which reads commands 
        from the Pipe cmndpipe and writes responses back to rspdpipe.
        '''
        super(PipedNoDisplayPQ, self).__init__()
        self.__cmndpipe = cmndpipe
        self.__rspdpipe = rspdpipe
        # The following causes crashes, thus assuming the answer is False
        # from PyQt4.QtGui import QFontDatabase
        # self.__supportsthreddedfontrendering = QFontDatabase.supportsThreadedFontRendering()
        self.__supportsthreddedfontrendering = False
        # ignore Ctrl-C
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        # default DPI used for this no-display graphics engine
        self.__physicalDpiX = 96.0
        self.__physicalDpiY = 96.0
        # default "display screen" size for this no-display graphics engine
        self.__screenwidth = 1920.0
        self.__screenheight = 1028.0
        # default scene size
        self.__scenewidth = int(10.5 * self.__physicalDpiX)
        self.__sceneheight = int(8.5 * self.__physicalDpiY)
        # scaling factor for line widths and symbol sizes
        self.__widthfactor = None
        self.setWidthScalingFactor(0.75)
        # initial default color for the background (opaque white)
        self.__lastclearcolor = QColor(0xFFFFFF)
        self.__lastclearcolor.setAlpha(0xFF)
        # List of QPictures creating the current scene
        self.__viewpics = [ ]
        self.__segid = None
        # QPicture/QPainter pair for the current view
        self.__activepicture = None
        self.__activepainter = None
        # Antialias when drawing?
        self.__antialias = True
        # data for recreating the current view
        self.__fracsides = None
        self.__clipit = True
        # number of drawing commands in the active painter
        self.__drawcount = 0
        # Limit the number of drawing commands per picture
        # to avoid the appearance of being "stuck"
        self.__maxdraws = 1024
        # scaling factor for creating the final scene
        self.__scalefactor = 1.0
        # Calculations of modified rectangular regions in QPictures
        # currently do not account for width and height of QPictures
        # played inside them.  So keep a expansion value.
        self.__maxsymbolwidth = 0.0
        self.__maxsymbolheight = 0.0
        self.__minsize = 128
        # default file name and format for saving the image
        self.__lastfilename = "ferret.png"
        self.__lastformat = "png"
        self.__addedannomargin = 12
        # command helper object
        self.__helper = CmndHelperPQ(self)
        # check the command queue any time there are no window events to deal with
        self.__timer = QTimer(self)
        self.__timer.timeout.connect(self.checkCommandPipe)
        self.__timer.setInterval(0)
        self.__timer.start()

    def exitViewer(self):
        '''
        stop the graphics engine
        '''
        self.__timer.stop()
        self.__cmndpipe.close()
        self.__rspdpipe.close()
        QCoreApplication.exit(0)

    def paintScene(self, painter, first, leftx, uppery, scalefactor, returnregion):
        '''
        Draws the pictures self.__viewpics[first:] using the QPainter
        painter.  This QPainter should have been initialized
        appropriately for the QPaintDevice to which it is painting
        (e.g., QImage.fill with the desired background color).

        The point (leftx, uppery) is the offset of the origin after
        scaling using scalefactor.  (All are floating point values.)

       If returnregion is True, a list of QRect objects describing
        the modified regions will be computed and returned.  If
        returnregion is False, the modified region will not be computed
        and an empty list will be returned.

        The call to painter.end() will need to be made after calling
        this function.
        '''
        # get the origin for drawing the pictures after scaling
        myorigin = QPointF(leftx, uppery)
        # set the scaling factor for the pictures
        painter.scale(scalefactor, scalefactor)
        modrects = [ ]
        # draw the appropriate pictures
        for (viewpic, _) in self.__viewpics[first:]:
            # draw the picture
            painter.drawPicture(myorigin, viewpic)
            if returnregion:
                picrect = viewpic.boundingRect()
                if picrect.isValid():
                    # Expand the region to account for possible symbols
                    xval = picrect.x() - 0.5 * self.__maxsymbolwidth
                    yval = picrect.y() - 0.5 * self.__maxsymbolheight
                    width = picrect.width() + self.__maxsymbolwidth
                    height = picrect.height() + self.__maxsymbolheight
                    # Scale and translate the region, then convert to integer
                    xval = int( math.floor(xval * scalefactor + leftx) )
                    width = int( math.ceil(width * scalefactor) )
                    yval = int( math.floor(yval * scalefactor + uppery) )
                    height = int( math.ceil(height * scalefactor) )
                    # Add this rectangle to the list
                    modrects.append( QRect(xval, yval, width, height) )
        return modrects

    def clearScene(self, bkgcolor):
        '''
        Removes all view pictures, and fills the scene with bkgcolor.
        If bkgcolor is None or an invalid color, the color used is 
        the one used from the last clearScene or redrawScene call 
        with a valid color (or opaque white if a color has never 
        been specified).
        '''
        # If there is an active View with content,
        # end it now, but do not update the scene
        if self.__activepainter and (self.__drawcount > 0):
            self.endView()
            restartview = True
        else:
            restartview = False
        # get the color to use for clearing (the background color)
        if bkgcolor:
            if bkgcolor.isValid():
                self.__lastclearcolor = bkgcolor
        # Delete all the pictures from the list
        self.__viewpics[:] = [ ]
        self.__maxsymbolwidth = 0.0
        self.__maxsymbolheight = 0.0
        # If there was an non-empty active View, restart it
        if restartview:
            self.beginViewFromSides(self.__fracsides, self.__clipit)

    def redrawScene(self, bkgcolor=None):
        '''
        Clear the scene using the given background color and redraw all 
        the pictures to the displayed scene.  If bkgcolor is None or an 
        invalid color, the color used is the one used from the last 
        clearScene or redrawScene call with a valid color (or opaque 
        white if a color has never been specified).
        '''
        # If there is an active View, end it now, but do not update the scene
        if self.__activepainter:
            self.endView()
            hadactiveview = True
        else:
            hadactiveview = False
        if bkgcolor:
            if bkgcolor.isValid():
                self.__lastclearcolor = bkgcolor
        # If there was an active View, restart it in this new system
        if hadactiveview:
            self.beginViewFromSides(self.__fracsides, self.__clipit)

    def resizeScene(self, width, height):
        '''
        Resize the scene to the given width and height in units of pixels.
        '''
        newwidth = int(width + 0.5)
        if newwidth < self.__minsize:
            newwidth = self.__minsize
        newheight = int(height + 0.5)
        if newheight < self.__minsize:
            newheight = self.__minsize
        if (newwidth != self.__scenewidth) or (newheight != self.__sceneheight):
            self.__scenewidth = newwidth
            self.__sceneheight = newheight
            # Redraw the scene from the beginning using the scaling factor
            self.redrawScene()


    def scaleScene(self, factor):
        '''
        Scales both the horizontal and vertical directions by factor.
        Scaling factors are not accumulative.  So if the scene was
        already scaled, that scaling is "removed" before this scaling
        factor is applied.
        '''
        newfactor = float(factor)
        newlabwidth = int(newfactor * self.__scenewidth + 0.5)
        newlabheight = int(newfactor * self.__sceneheight + 0.5)
        if (newlabwidth < self.__minsize) or (newlabheight < self.__minsize):
            # Set to minimum size
            if self.__scenewidth <= self.__sceneheight:
                newfactor = float(self.__minsize) / float(self.__scenewidth)
            else:
                newfactor = float(self.__minsize) / float(self.__sceneheight)
            newlabwidth = int(newfactor * self.__scenewidth + 0.5)
            newlabheight = int(newfactor * self.__sceneheight + 0.5)
        oldlabwidth = int(self.__scalefactor * self.__scenewidth + 0.5)
        oldlabheight = int(self.__scalefactor * self.__sceneheight + 0.5)
        if (newlabwidth != oldlabwidth) or (newlabheight != oldlabheight):
            # Set the new scaling factor
            self.__scalefactor = newfactor
            # Redraw the scene from the beginning
            self.redrawScene()

    def saveSceneToFile(self, filename, imageformat, transparent, 
                        vectsize, rastsize, annotations):
        '''
        Save the current scene to the named file.  If imageformat
        is empty or None, the format is guessed from the filename
        extension.

        If transparent is False, the entire scene is initialized
        to the last clearing color used, using a filled rectangle 
        for vector images.

        If given, vectsize is the size in inches of a saved vector 
        image.  If vectsize is not given, a vector image will be 
        saved at the current displayed scaled image size, unless
        specified otherwise if showPrintDialog is True.  

        If given, rastsize is the pixels size of a saved raster 
        image.  If rastsize is not given, a raster image will be 
        saved at the current displayed scaled image size.  

        If annotations is not None, the strings given in the tuple
        are to be displayed above the image.  These annotations add 
        height, as needed, to the saved image (i.e., vectsize or 
        rastsize gives the height of the image below these annotations).
        '''
        # This could be called when there is no scene present.
        # If this is the case, ignore the call.
        if len(self.__viewpics) == 0:
            return
        if not imageformat:
            # Guess the image format from the filename extension
            # to determine if it is a vector type, and if so,
            # which type. All the raster types use a QImage, which
            # does this guessing of format when saving the image.
            fileext = ( os.path.splitext(filename)[1] ).lower()
            if fileext == '.pdf':
                # needs a PDF QPrinter
                myformat = 'pdf'
            elif fileext == '.ps':
                # needs a PS QPrinter
                myformat = 'ps'
            elif fileext == '.svg':
                # needs a QSvgGenerator
                myformat = 'svg'
            elif fileext == '.plt':
                # check for plt (gks metafile) - needs to be changed to pdf
                myformat = 'plt'
            elif fileext == '.gif':
                # check for gif - needs to be changed to png
                myformat = 'gif'
            else:
                # use a QImage and let it figure out the format
                myformat = None
        else:
            myformat = imageformat.lower()

        if myformat == 'plt':
            # Silently convert plt filename and format to pdf
            myformat = 'pdf'
            myfilename = os.path.splitext(filename)[0] + ".pdf"
        elif myformat == 'gif':
            # Silently convert gif filename and format to png
            myformat = 'png'
            myfilename = os.path.splitext(filename)[0] + ".png"
        else:
            myfilename = filename

        # The RHEL5 distribution of Qt4 does not have a QSvgGenerator
        if (not HAS_QSvgGenerator) and (myformat == 'svg'):
            raise ValueError("Your version of Qt does not support generation of SVG files")

        if annotations and self.__supportsthreddedfontrendering:
            annopicture = QPicture()
            annopainter = QPainter(annopicture)
            annotextdoc = QTextDocument()
            # Leave room for the added margins to the width
            annotextdoc.setTextWidth(self.__scenewidth - 2.0 * self.__addedannomargin)
            annotextdoc.setHtml("<p>" + "<br />".join(annotations) + "</p>")
            annotextdoc.drawContents(annopainter)
            annopainter.end()
            annosize = annotextdoc.documentLayout().documentSize()
        else:
            annopicture = None
            annosize = None

        if (myformat == 'ps') or (myformat == 'pdf'):
            # Setup the QPrinter that will be used to create the PS or PDF file
            printer = QPrinter(QPrinter.HighResolution)
            printer.setOutputFileName(myfilename)
            # The print format is automatically set from the
            # filename extension; so the following is actually
            # only needed for absent or strange extensions
            if myformat == 'ps':
                printer.setOutputFormat(QPrinter.PostScriptFormat)
            else:
                printer.setOutputFormat(QPrinter.PdfFormat)
            # Print to file in color
            printer.setColorMode(printer.Color)
            # get the width and height in inches of the image to be produced
            if vectsize:
                imagewidth = vectsize.width()
                imageheight = vectsize.height()
            else:
                imagewidth = self.__scenewidth * self.__scalefactor \
                             / float(self.__physicalDpiX)
                imageheight = self.__sceneheight * self.__scalefactor \
                              / float(self.__physicalDpiY)
            # Add in any height needed for the annotations
            if annopicture:
                annoheight = (annosize.height() + 2 * self.__addedannomargin) * \
                             imageheight / self.__sceneheight
                imageheight += annoheight
            # Set the image size
            try:
                # Set custom paper size to just fit around the image
                if ( imagewidth > imageheight ):
                    printer.setPaperSize(QSizeF(imageheight, imagewidth), QPrinter.Inch)
                else:
                    printer.setPaperSize(QSizeF(imagewidth, imageheight), QPrinter.Inch)
                # The above has issues with Qt 4.6 at GFDL - 
                # still puts it on the default letter size page.
                # So just always use a letter size page.
                # printer.setPaperSize(QPrinter.Letter)
            except AttributeError:
                # setPaperSize introduced in 4.4 and made setPageSize 
                # obsolete; but RHEL5 Qt4 is 4.2, so set to letter size
                printer.setPageSize(QPrinter.Letter)
            # No margins (setPageMargins introduced in 4.4)
            printer.setFullPage(True)
            # Default orientation
            if ( imagewidth > imageheight ):
                printer.setOrientation(QPrinter.Landscape)
            else:
                printer.setOrientation(QPrinter.Portrait)
            # also get the image size in units of printer dots
            printres = printer.resolution()
            printwidth = int(imagewidth * printres + 0.5)
            printheight = int(imageheight * printres + 0.5)
            # Set up to draw to the QPrinter
            painter = QPainter(printer)
            if not transparent:
                # Draw a rectangle filling the entire scene
                # with the last clearing color.
                # Only draw if not completely transparent
                if (self.__lastclearcolor.getRgb())[3] > 0:
                    painter.fillRect(QRectF(0, 0, printwidth, printheight), 
                                     self.__lastclearcolor)
            # Scaling printfactor for the scene to the saved image
            widthscalefactor = imagewidth * self.__physicalDpiX / float(self.__scenewidth)
            # Check if there are annotations to add
            if annopicture:
                # Scale the scene now for the annotations
                painter.scale(widthscalefactor, widthscalefactor)
                # factor that makes it work after scaling (12.5 = 1200 / 96)
                printfactor = printres / self.__physicalDpiX
                # Draw a solid white rectangle with black outline for the annotations
                painter.setBrush(QBrush(Qt.white, Qt.SolidPattern))
                painter.setPen(QPen(QBrush(Qt.black, Qt.SolidPattern), 
                                    2.0 * printfactor, Qt.SolidLine, Qt.SquareCap, Qt.MiterJoin))
                painter.drawRect(QRectF(1.0 * printfactor, 1.0 * printfactor, 
                    (self.__scenewidth - 2.0) * printfactor, 
                    ((annosize.height() + 2.0 * self.__addedannomargin) - 2.0) * printfactor))
                # And add the annotations within this box
                painter.drawPicture(QPointF(self.__addedannomargin * printfactor,
                                            self.__addedannomargin * printfactor), 
                                    annopicture)
                # Draw the scene to the printer - scaling already in effect
                self.paintScene(painter, 0, 0.0, 
                        (annosize.height() + 2.0 * self.__addedannomargin) * printfactor, 
                        1.0, False)
            else:
                # No annotations so just do the normal drawing
                self.paintScene(painter, 0, 0.0, 0.0, 
                                widthscalefactor, False)                
            painter.end()
        elif myformat == 'svg':
            # if HAS_QSvgGenerator is False, it should never get here
            generator = QSvgGenerator()
            generator.setFileName(myfilename)
            if vectsize:
                imagewidth = int(vectsize.width() * self.__physicalDpiX + 0.5)
                imageheight = int(vectsize.height() * self.__physicalDpiY + 0.5)
            else:
                imagewidth = int(self.__scenewidth * self.__scalefactor + 0.5)
                imageheight = int(self.__sceneheight * self.__scalefactor + 0.5)
            # Add in any height needed for the annotations
            if annopicture:
                annoheight = (annosize.height() + 2 * self.__addedannomargin) * \
                             imageheight / self.__sceneheight
                imageheight += annoheight
            # Set the image size
            generator.setResolution( 
                int(0.5 * (self.__physicalDpiX + self.__physicalDpiY + 0.5)) )
            generator.setSize( QSize(imagewidth, imageheight) )
            generator.setViewBox( QRect(0, 0, imagewidth, imageheight) )
            # paint the scene to this QSvgGenerator
            painter = QPainter(generator)
            if not transparent:
                # Draw a rectangle filling the entire scene
                # with the last clearing color.
                # Only draw if not completely transparent
                if (self.__lastclearcolor.getRgb())[3] > 0:
                    painter.fillRect( QRectF(0, 0, imagewidth, imageheight),
                                      self.__lastclearcolor )
            # Scaling printfactor for the scene to the saved image
            widthscalefactor = imagewidth / float(self.__scenewidth)
            # Check if there are annotations to add
            if annopicture:
                # Scale the scene now for the annotations
                painter.scale(widthscalefactor, widthscalefactor)
                # Draw a solid white rectangle with black outline for the annotations
                painter.setBrush(QBrush(Qt.white, Qt.SolidPattern))
                painter.setPen(QPen(QBrush(Qt.black, Qt.SolidPattern), 
                                    2.0, Qt.SolidLine, Qt.SquareCap, Qt.BevelJoin))
                painter.drawRect(QRectF(1.0, 1.0, 
                            self.__scenewidth - 2.0, 
                            annosize.height() + 2.0 * self.__addedannomargin - 2.0))
                # And add the annotations within this box
                painter.drawPicture(QPointF(self.__addedannomargin,self.__addedannomargin), 
                                    annopicture)
                # Draw the scene to the printer - scaling already in effect
                self.paintScene(painter, 0, 
                                0.0, annosize.height() + 2.0 * self.__addedannomargin, 
                                1.0, False)
            else:
                # No annotations so just do the normal drawing
                self.paintScene(painter, 0, 0.0, 0.0, 
                                widthscalefactor, False)                
            painter.end()
        else:
            if rastsize:
                imagewidth = int(rastsize.width() + 0.5)
                imageheight = int(rastsize.height() + 0.5)
            else:
                imagewidth = int(self.__scenewidth * self.__scalefactor + 0.5)
                imageheight = int(self.__sceneheight * self.__scalefactor + 0.5)
            # Add in any height needed for the annotations
            if annopicture:
                annoheight = (annosize.height() + 2 * self.__addedannomargin) * \
                             imageheight / self.__sceneheight
                imageheight += annoheight
            # Create the image
            image = QImage( QSize(imagewidth, imageheight),
                            QImage.Format_ARGB32_Premultiplied )
            # Initialize the image
            # Note that completely transparent gives black for formats not supporting 
            # the alpha channel (JPEG) whereas ARGB32 with 0x00FFFFFF gives white
            if not transparent:
                # Clear the image with self.__lastclearcolor
                fillint = self.__helper.computeARGB32PreMultInt(self.__lastclearcolor)
            else:
                fillint = 0
            image.fill(fillint)
            # paint the scene to this QImage
            painter = QPainter(image)
            # Scaling printfactor for the scene to the saved image
            widthscalefactor = imagewidth / float(self.__scenewidth)
            # Check if there are annotations to add
            if annopicture:
                # Scale the scene now for the annotations
                painter.scale(widthscalefactor, widthscalefactor)
                # Draw a solid white rectangle with black outline for the annotations
                painter.setBrush(QBrush(Qt.white, Qt.SolidPattern))
                painter.setPen(QPen(QBrush(Qt.black, Qt.SolidPattern), 
                                    2.0, Qt.SolidLine, Qt.SquareCap, Qt.BevelJoin))
                painter.drawRect(QRectF(1.0, 1.0, 
                            self.__scenewidth - 2.0, 
                            annosize.height() + 2.0 * self.__addedannomargin - 2.0))
                # And add the annotations within this box
                painter.drawPicture(QPointF(self.__addedannomargin,self.__addedannomargin), 
                                    annopicture)
                # Draw the scene to the printer - scaling already in effect
                self.paintScene(painter, 0, 
                                0.0, annosize.height() + 2.0 * self.__addedannomargin, 
                                1.0, False)
            else:
                # No annotations so just do the normal drawing
                self.paintScene(painter, 0, 0.0, 0.0, 
                                widthscalefactor, False)                
            painter.end()
            # save the image to file
            image.save(myfilename, myformat)

    def checkCommandPipe(self):
        '''
        Get and perform commands waiting in the pipe.
        Stop when no more commands or if more than 50
        milliseconds have passed.
        '''
        try:
            starttime = time.clock()
            # Wait up to 2 milliseconds waiting for a command.
            # This prevents unchecked spinning when there is
            # nothing to do (Qt immediately calling this method
            # again only for this method to immediately return).
            while self.__cmndpipe.poll(0.002):
                cmnd = self.__cmndpipe.recv()
                self.processCommand(cmnd)
                # Continue to try to process commands until
                # more than 50 milliseconds have passed.
                # This reduces Qt overhead when there are lots
                # of commands waiting in the queue.
                if (time.clock() - starttime) > 0.050:
                    break
        except Exception:
            # EOFError should never arise from recv since
            # the call is after poll returns True
            (exctype, excval) = sys.exc_info()[:2]
            if excval:
                self.__rspdpipe.send("**ERROR %s: %s" % (str(exctype), str(excval)))
            else:
                self.__rspdpipe.send("**ERROR %s" % str(exctype))
            self.exitViewer()

    def processCommand(self, cmnd):
        '''
        Examine the action of cmnd and call the appropriate
        method to deal with this command.  Raises a KeyError
        if the "action" key is missing.
        '''
        try:
            cmndact = cmnd["action"]
        except KeyError:
            raise ValueError("Unknown command %s" % str(cmnd))

        if cmndact == "clear":
            try:
                bkgcolor = self.__helper.getColorFromCmnd(cmnd)
            except KeyError:
                bkgcolor = None
            self.clearScene(bkgcolor)
        elif cmndact == "exit":
            self.exitViewer()
        elif cmndact == "hide":
            pass
        elif cmndact == "screenInfo":
            info = ( self.__physicalDpiX, self.__physicalDpiY, 
                     self.__screenwidth, self.__screenheight )
            self.__rspdpipe.send(info)
        elif cmndact == "antialias":
            self.__antialias = bool(cmnd.get("antialias", True))
        elif cmndact == "update":
            self.updateScene()
        elif cmndact == "redraw":
            try:
                bkgcolor = self.__helper.getColorFromCmnd(cmnd)
            except KeyError:
                bkgcolor = None
            self.redrawScene(bkgcolor)
        elif cmndact == "rescale":
            newscale = float(cmnd["factor"])
            if newscale <= 0.0:
                raise ValueError("invalid scaling factor")
            self.scaleScene(newscale)
        elif cmndact == "resize":
            mysize = self.__helper.getSizeFromCmnd(cmnd)
            self.resizeScene(mysize.width(), mysize.height())
        elif cmndact == "save":
            filename = cmnd["filename"]
            fileformat = cmnd.get("fileformat", None)
            transparent = cmnd.get("transparent", False)
            vectsize = self.__helper.getSizeFromCmnd(cmnd["vectsize"])
            rastsize = self.__helper.getSizeFromCmnd(cmnd["rastsize"])
            try:
                annotations = cmnd["annotations"]
            except KeyError:
                annotations = None
            self.saveSceneToFile(filename, fileformat, transparent, 
                                 vectsize, rastsize, annotations)
        elif cmndact == "setWidthFactor":
            newfactor = float(cmnd.get("factor", -1.0))
            if newfactor <= 0.0:
                raise ValueError("Invalid width factor")
            self.setWidthScalingFactor(newfactor)
        elif cmndact == "setTitle":
            pass
        elif cmndact == "imgname":
            value = cmnd.get("name", None)
            if value:
                self.__lastfilename = value
            value = cmnd.get("format", None)
            if value:
                self.__lastformat = value.lower()
        elif cmndact == "show":
            pass
        elif cmndact == "beginView":
            self.beginView(cmnd)
        elif cmndact == "clipView":
            self.clipView(cmnd)
        elif cmndact == "endView":
            self.endView()
        elif cmndact == "beginSegment":
            self.beginSegment(cmnd["segid"])
        elif cmndact == "endSegment":
            self.endSegment()
        elif cmndact == "deleteSegment":
            self.deleteSegment(cmnd["segid"])
        elif cmndact == "drawMultiline":
            self.drawMultiline(cmnd)
        elif cmndact == "drawPoints":
            self.drawPoints(cmnd)
        elif cmndact == "drawPolygon":
            self.drawPolygon(cmnd)
        elif cmndact == "drawRectangle":
            self.drawRectangle(cmnd)
        elif cmndact == "drawText":
            self.drawSimpleText(cmnd)
        else:
            raise ValueError("Unknown command action %s" % str(cmndact))

    def beginView(self, cmnd):
        '''
        Setup a new viewport and window for drawing on a portion
        (possibly all) of the scene.  Recognized keys from cmnd
        are:
            "viewfracs": a dictionary of sides positions (see
                    CmndHelperPQ.getSidesFromCmnd) giving the
                    fractions [0.0, 1.0] of the way through the
                    scene for the sides of the new View.
            "clip": clip to the new View? (default: True)

        Note that the view fraction values are based on (0,0) being the
        top left corner and (1,1) being the bottom right corner.  Thus,
        left < right and top < bottom.

        Raises a KeyError if the "viewfracs" key is not given.
        '''
        # Get the view rectangle in fractions of the full scene
        fracsides = self.__helper.getSidesFromCmnd(cmnd["viewfracs"])
        # Should graphics be clipped to this view?
        try:
            clipit = cmnd["clip"]
        except KeyError:
            clipit = True
        self.beginViewFromSides(fracsides, clipit)

    def beginViewFromSides(self, fracsides, clipit):
        '''
        Setup a new viewport and window for drawing on a portion
        (possibly all) of the scene.  The view in fractions of
        the full scene are given in fracsides.  Sets the clipping
        rectangle to this view.  If clipit is True, graphics
        will be clipped to this view.
        '''
        # If a view is still active, automatically end it
        if self.__activepainter:
            self.endView()
        # Get the location for the new view in terms of scene pixels.
        width = float(self.__scenewidth)
        height = float(self.__sceneheight)
        leftpixel = fracsides.left() * width
        rightpixel = fracsides.right() * width
        bottompixel = fracsides.bottom() * height
        toppixel = fracsides.top() * height
        # perform the checks after turning into units of pixels
        # to make sure the values are significantly different
        if (0.0 > leftpixel) or (leftpixel >= rightpixel) or (rightpixel > width):
            raise ValueError( "Invalid left, right view fractions: " \
                              "left in pixels = %s, right in pixels = %s" \
                              % (str(leftpixel), str(rightpixel)) )
        if (0.0 > toppixel) or (toppixel >= bottompixel) or (bottompixel > height):
            raise ValueError( "Invalid bottom, top view fractions: " \
                              "top in pixels = %s, bottom in pixels = %s" \
                              % (str(toppixel), str(bottompixel)) )
        # Create the view rectangle in device coordinates
        vrectf = QRectF(leftpixel, toppixel,
                       rightpixel - leftpixel, bottompixel - toppixel)
        # Create the new picture and painter
        self.__activepicture = QPicture()
        self.__activepainter = QPainter(self.__activepicture)
        # Set the clip rectangle to that of the view; this also activates clipping
        self.__activepainter.setClipRect(vrectf, Qt.ReplaceClip)
        # Disable clipping if not desired at this time
        if not clipit:
            self.__activepainter.setClipping(False)
        # Note that __activepainter has to end before __activepicture will
        # draw anything.  So no need to add it to __viewpics until then.
        self.__drawcount = 0
        # Save the current view sides and clipit setting for recreating the view.
        # Just save the original objects (assume calling functions do not keep them)
        self.__fracsides = fracsides
        self.__clipit = clipit

    def clipView(self, cmnd):
        '''
        If cmnd["clip"] is True, activates clipping to the
        current view rectangle.  If cmnd["clip"] is False,
        disable clipping in this view.

        Raises a KeyError if the "clip" key is not given.
        '''
        if cmnd["clip"]:
            self.__activepainter.setClipping(True)
            self.__clipit = True
        else:
            self.__activepainter.setClipping(False)
            self.__clipit = False

    def endView(self):
        '''
        Ends the current view and appends it to the list of pictures
        drawn in the scene.
        '''
        self.__activepainter.end()
        self.__activepainter = None
        # Only save the active picture if it contains something
        if self.__drawcount > 0:
            self.__viewpics.append((self.__activepicture, self.__segid))
            self.__drawcount = 0
        self.__activepicture = None

    def beginSegment(self, segid):
        '''
        Associates a segment ID with the current empty view
        (picture) and all future views until endSegment is 
        called.  If the current view is not empty, the current 
        view is ended and a new view started.  If there is not
        a active view, the segment ID is just saved for the
        next active view.
        '''
        if self.__activepainter and (self.__drawcount > 0):
            self.endView()
            self.beginViewFromSides(self.__fracsides, self.__clipit)
        self.__segid = segid
        
    def endSegment(self):
        '''
        Ends the current active view and starts a new view.
        Removes the current segment ID associated with views.
        '''
        if self.__activepainter and (self.__drawcount > 0):
            self.endView()
            self.beginViewFromSides(self.__fracsides, self.__clipit)
        self.__segid = None

    def deleteSegment(self, segid):
        '''
        Removes all pictures associated with the given segment ID
        '''
        # if deleting the current segment, end the current segment
        if segid == self.__segid:
            self.endSegment()
        # Go through all the pictures, determining which to save
        newpicts = [ ]
        for (viewpic, vsegid) in self.__viewpics:
            if vsegid != segid:
                newpicts.append((viewpic, vsegid))
        self.__viewpics[:] = newpicts
        # Do NOT update since there may be more segments to be deleted
        # Rely on the receiving an update or redraw command at the end 

    def updateScene(self):
        '''
        If there is a current view with something drawn, ends this view 
        and then starts a new view with the same limits.  Used for 
        limiting the number of drawing elements in a view.
        '''
        if self.__drawcount > 0:
            self.endView()
            self.beginViewFromSides(self.__fracsides, self.__clipit)

    def drawMultiline(self, cmnd):
        '''
        Draws a collection of connected line segments.

        Recognized keys from cmnd:
            "points": consecutive endpoints of the connected line
                    segments as a list of (x, y) coordinates
            "pen": dictionary describing the pen used to draw the
                    segments (see CmndHelperPQ.getPenFromCmnd)

        The coordinates are device coordinates from the upper left corner.

        Raises:
            KeyError if the "points" or "pen" key is not given
            ValueError if there are fewer than two endpoints given
        '''
        ptcoords = cmnd["points"]
        if len(ptcoords) < 2:
            raise ValueError("fewer that two endpoints given")
        endpts = QPolygonF( [ QPointF(xypair[0], xypair[1]) \
                                  for xypair in ptcoords ] )
        mypen = self.__helper.getPenFromCmnd(cmnd["pen"])
        self.__activepainter.setRenderHint(QPainter.Antialiasing,
                                           self.__antialias)
        self.__activepainter.setBrush(Qt.NoBrush)
        self.__activepainter.setPen(mypen)
        self.__activepainter.drawPolyline(endpts)
        self.__drawcount += 1
        # Limit the number of drawing commands per picture
        if self.__drawcount >= self.__maxdraws:
            self.updateScene()

    def drawPoints(self, cmnd):
        '''
        Draws a collection of discrete points using a single symbol
        for each point.

        Recognized keys from cmnd:
            "points": point centers as a list of (x,y) coordinates
            "symbol": name of the symbol to use
                    (see CmndHelperPQ.getSymbolFromCmnd)
            "size": size of the symbol in points (1/72 inches); possibly
                    further scaled by the width scaling factor
            "color": color name or 24-bit RGB integer value (eg, 0xFF0088)
            "alpha": alpha value from 0 (transparent) to 255 (opaque)

        The coordinates are device coordinates from the upper left corner.

        Raises a KeyError if the "symbol", "points", or "size" key
        is not given.
        '''
        ptcoords = cmnd["points"]
        ptsize = cmnd["size"]
        sympath = self.__helper.getSymbolFromCmnd(cmnd["symbol"])
        self.__activepainter.setRenderHint(QPainter.Antialiasing,
                                           self.__antialias)
        try:
            mycolor = self.__helper.getColorFromCmnd(cmnd)
            mybrush = QBrush(mycolor, Qt.SolidPattern)
        except KeyError:
            mybrush = QBrush(Qt.SolidPattern)
        if sympath.isFilled():
            self.__activepainter.setBrush(mybrush)
            self.__activepainter.setPen(Qt.NoPen)
        else:
            self.__activepainter.setBrush(Qt.NoBrush)
            # pen width is 15% of the width of the symbol
            mypen = QPen(mybrush, 15.0, Qt.SolidLine,
                         Qt.SquareCap, Qt.BevelJoin)
            self.__activepainter.setPen(mypen)
        # Unmodified symbols are 100x100 pixels 
        scalefactor = ptsize * self.widthScalingFactor() / 100.0
        if self.__maxsymbolwidth < 100.0 * scalefactor:
            self.__maxsymbolwidth = 100.0 * scalefactor
        if self.__maxsymbolheight < 100.0 * scalefactor:
            self.__maxsymbolheight = 100.0 * scalefactor
        for xyval in ptcoords:
            # save so the translation and scale are not permanent
            self.__activepainter.save()
            try:
                self.__activepainter.translate( QPointF(xyval[0], xyval[1]) )
                self.__activepainter.scale(scalefactor, scalefactor)
                self.__activepainter.drawPath(sympath.painterPath())
            finally:
                self.__activepainter.restore()
        self.__drawcount += len(ptcoords)
        # Limit the number of drawing commands per picture
        if self.__drawcount >= self.__maxdraws:
            self.updateScene()

    def drawPolygon(self, cmnd):
        '''
        Draws a polygon item to the viewer.

        Recognized keys from cmnd:
            "points": the vertices of the polygon as a list of (x,y)
                    coordinates
            "fill": dictionary describing the brush used to fill the
                    polygon; see CmndHelperPQ.getBrushFromCmnd
                    If not given, the polygon will not be filled.
            "outline": dictionary describing the pen used to outline
                    the polygon; see CmndHelperPQ.getPenFromCmnd
                    If not given, the border will be drawn with a
                    cosmetic pen identical to the brush used to fill
                    the polygon.

        The coordinates are device coordinates from the upper left corner.

        Raises a KeyError if the "points" key is not given.
        '''
        mypoints = cmnd["points"]
        mypolygon = QPolygonF( [ QPointF(xypair[0], xypair[1]) \
                                     for xypair in mypoints ] )
        self.__activepainter.setRenderHint(QPainter.Antialiasing,
                                           False)
        try:
            mybrush = self.__helper.getBrushFromCmnd(cmnd["fill"])
        except KeyError:
            mybrush = Qt.NoBrush
        try:
            mypen = self.__helper.getPenFromCmnd(cmnd["outline"])
        except KeyError:
            if ( mybrush == Qt.NoBrush ):
                raise ValueError('drawPolygon called without a Brush or Pen')
            # Use a "cosmetic" Pen matching the brush
            mypen = QPen(mybrush, 1.0, Qt.SolidLine, Qt.SquareCap, Qt.BevelJoin)
        self.__activepainter.setBrush(mybrush)
        self.__activepainter.setPen(mypen)
        self.__activepainter.drawPolygon(mypolygon)
        self.__drawcount += 1
        # Limit the number of drawing commands per picture
        if self.__drawcount >= self.__maxdraws:
            self.updateScene()

    def drawRectangle(self, cmnd):
        '''
        Draws a rectangle in the current view using the information
        in the dictionary cmnd.

        Recognized keys from cmnd:
            "left": x-coordinate of left edge of the rectangle
            "bottom": y-coordinate of the bottom edge of the rectangle
            "right": x-coordinate of the right edge of the rectangle
            "top": y-coordinate of the top edge of the rectangle
            "fill": dictionary describing the brush used to fill the
                    rectangle; see CmndHelperPQ.getBrushFromCmnd
                    If not given, the rectangle will not be filled.
            "outline": dictionary describing the pen used to outline
                    the rectangle; see CmndHelperPQ.getPenFromCmnd
                    If not given, the border will be drawn with a
                    cosmetic pen identical to the brush used to fill
                    the rectangle.

        The coordinates are device coordinates from the upper left corner.

        Raises a ValueError if the width or height of the rectangle
        is not positive.
        '''
        # get the left, bottom, right, and top values
        # any keys not given get a zero value
        sides = self.__helper.getSidesFromCmnd(cmnd)
        width = sides.right() - sides.left()
        if width <= 0.0:
            raise ValueError("width of the rectangle in not positive")
        height = sides.bottom() - sides.top()
        if height <= 0.0:
            raise ValueError("height of the rectangle in not positive")
        myrect = QRectF(sides.left(), sides.top(), width, height)
        self.__activepainter.setRenderHint(QPainter.Antialiasing,
                                           False)
        try:
            mybrush = self.__helper.getBrushFromCmnd(cmnd["fill"])
        except KeyError:
            mybrush = Qt.NoBrush
        try:
            mypen = self.__helper.getPenFromCmnd(cmnd["outline"])
        except KeyError:
            if ( mybrush == Qt.NoBrush ):
                raise ValueError('drawPolygon called without a Brush or Pen')
            # Use a "cosmetic" Pen matching the brush
            mypen = QPen(mybrush, 1.0, Qt.SolidLine, Qt.SquareCap, Qt.BevelJoin)
        self.__activepainter.setBrush(mybrush)
        self.__activepainter.setPen(mypen)
        self.__activepainter.drawRect(myrect)
        self.__drawcount += 1
        # Limit the number of drawing commands per picture
        if self.__drawcount >= self.__maxdraws:
            self.updateScene()

    def drawSimpleText(self, cmnd):
        '''
        Draws a "simple" text item in the current view.
        Raises a KeyError if the "text" or "location" key is not given.

        Recognized keys from cmnd:
            "text": string to displayed
            "font": dictionary describing the font to use;  see
                    CmndHelperPQ.getFontFromCmnd.  If not given
                    the default font for this viewer is used.
            "fill": dictionary describing the pen used to draw the
                    text; see CmndHelperPQ.getPenFromCmnd.
                    If not given, the default pen for this viewer
                    is used.
            "rotate": clockwise rotation of the text in degrees
            "location": (x,y) location for the baseline of the
                    start of text.  The coordinates are device
                    coordinates from the upper left corner.
        '''
        # from http://qt-project.org/doc/qt-4.8/threads-modules.html#painting-in-threads
        # "Note that on X11 systems without FontConfig support, Qt cannot render text outside of the GUI thread"
        if not self.__supportsthreddedfontrendering:
            return
        mytext = cmnd["text"]
        startpt = cmnd["location"]
        self.__activepainter.setRenderHint(QPainter.Antialiasing,
                                           self.__antialias)
        self.__activepainter.setBrush(Qt.NoBrush)
        try:
            mypen = self.__helper.getPenFromCmnd(cmnd["fill"])
            self.__activepainter.setPen(mypen)
        except KeyError:
            pass
        # save so the font, translation, and rotation are not permanent
        self.__activepainter.save()
        try:
            try:
                myfont = self.__helper.getFontFromCmnd(cmnd["font"])
                self.__activepainter.setFont(myfont)
            except KeyError:
                pass
            # Move the coordinate system so the origin is at the start
            # of the text so that rotation is about this point
            self.__activepainter.translate(startpt[0], startpt[1])
            try:
                rotdeg = cmnd["rotate"]
                self.__activepainter.rotate(rotdeg)
            except KeyError:
                pass
            mystring = QCoreApplication.translate("PipedNoDisplayPQ", mytext)
            self.__activepainter.drawText(0, 0, mystring)
            self.__drawcount += 1
        finally:
            # return the painter to the default state
            self.__activepainter.restore()
        # Limit the number of drawing commands per picture
        if self.__drawcount >= self.__maxdraws:
            self.updateScene()

    def setWidthScalingFactor(self, factor):
        '''
        Assign the scaling factor for line widths and symbol sizes 
        to convert from points (1/72 inches) to pixels, and to apply 
        any additional width scaling specified by factor. 
        '''
        self.__widthfactor  = (self.__physicalDpiX + self.__physicalDpiY) / 144.0
        self.__widthfactor *= factor
        
    def widthScalingFactor(self):
        '''
        Return the scaling factor for line widths and symbol sizes 
        to convert from points (1/72 inches) to pixels, and to apply 
        any additional width scaling specified by setWidthFactor. 
        '''
        return self.__widthfactor 


class PipedNoDisplayPQProcess(Process):
    '''
    A Process specifically tailored for creating a PipedNoDisplayPQ.
    '''
    def __init__(self, cmndpipe, rspdpipe):
        '''
        Create a Process that will produce a PipedNoDisplayPQ
        attached to the given Pipes when run.
        '''
        Process.__init__(self)
        self.__cmndpipe = cmndpipe
        self.__rspdpipe = rspdpipe
        self.__app = None
        self.__viewer = None

    def run(self):
        '''
        Create a PipedNoDisplayPQ that is attached
        to the Pipe of this instance.
        '''
        self.__app = QCoreApplication(["PipedNoDisplayPQ"])
        self.__viewer = PipedNoDisplayPQ(self.__cmndpipe, self.__rspdpipe)
        myresult = self.__app.exec_()
        self.__cmndpipe.close()
        self.__rspdpipe.close()
        SystemExit(myresult)


#
# The following is for testing this and the cmndhelperpq modules
#

if __name__ == "__main__":
    # vertices of a pentagon (roughly) centered in a 1000 x 1000 square
    pentagonpts = ( (504.5, 100.0), (100.0, 393.9),
                    (254.5, 869.4), (754.5, 869.4),
                    (909.0, 393.9),  )

    # create the list of commands to submit
    drawcmnds = []
    drawcmnds.append( { "action":"setTitle", "title":"Tester" } )
    drawcmnds.append( { "action":"show" } )
    drawcmnds.append( { "action":"clear", "color":"black"} )
    drawcmnds.append( { "action":"screenInfo"} )
    drawcmnds.append( { "action":"antialias", "antialias":True } )
    drawcmnds.append( { "action":"resize",
                        "width":500,
                        "height":500 } )
    drawcmnds.append( { "action":"beginView",
                        "viewfracs":{"left":0.0, "right":0.5,
                                     "top":0.5, "bottom":1.0},
                        "clip":True } )
    drawcmnds.append( { "action":"drawRectangle",
                        "left": 5, "right":245, 
                        "top":245, "bottom":495,
                        "fill":{"color":"green", "alpha":128} } )
    mypentapts = [ (.25 * ptx, .25 * pty + 250) for (ptx, pty) in pentagonpts ]
    drawcmnds.append( { "action":"drawPolygon",
                        "points":mypentapts,
                        "fill":{"color":"blue"},
                        "outline":{"color":"black",
                                   "width": 5,
                                   "style":"solid",
                                   "capstyle":"round",
                                   "joinstyle":"round" } } )
    drawcmnds.append( { "action":"endView" } )
    drawcmnds.append( { "action":"show" } )
    drawcmnds.append( { "action":"beginView",
                        "viewfracs":{"left":0.0, "right":1.0,
                                     "top":0.0, "bottom":1.0},
                        "clip":True } )
    drawcmnds.append( { "action":"drawPoints",
                        "points":( (100,  50),
                                   (100, 150),
                                   (100, 250),
                                   (100, 350),
                                   (100, 450) ),
                        "symbol":".",
                        "size":20,
                        "color":"magenta" })
    drawcmnds.append( { "action":"drawPoints",
                        "points":( (150,  50),
                                   (150, 150),
                                   (150, 250),
                                   (150, 350),
                                   (150, 450) ),
                        "symbol":"o",
                        "size":20,
                        "color":"magenta" })
    drawcmnds.append( { "action":"drawPoints",
                        "points":( (200,  50),
                                   (200, 150),
                                   (200, 250),
                                   (200, 350),
                                   (200, 450) ),
                        "symbol":"+",
                        "size":20,
                        "color":"magenta" })
    drawcmnds.append( { "action":"drawPoints",
                        "points":( (250,  50),
                                   (250, 150),
                                   (250, 250),
                                   (250, 350),
                                   (250, 450) ),
                        "symbol":"x",
                        "size":20,
                        "color":"magenta" })
    drawcmnds.append( { "action":"drawPoints",
                        "points":( (300,  50),
                                   (300, 150),
                                   (300, 250),
                                   (300, 350),
                                   (300, 450) ),
                        "symbol":"*",
                        "size":20,
                        "color":"magenta" })
    drawcmnds.append( { "action":"drawPoints",
                        "points":( (350,  50),
                                   (350, 150),
                                   (350, 250),
                                   (350, 350),
                                   (350, 450) ),
                        "symbol":"^",
                        "size":20,
                        "color":"magenta" })
    drawcmnds.append( { "action":"drawPoints",
                        "points":( (400,  50),
                                   (400, 150),
                                   (400, 250),
                                   (400, 350),
                                   (400, 450) ),
                        "symbol":"#",
                        "size":20,
                        "color":"magenta" })
    drawcmnds.append( { "action":"drawMultiline",
                        "points":( (350,  50),
                                   (200, 150),
                                   (400, 250),
                                   (300, 350),
                                   (150, 250),
                                   (100, 450) ),
                        "pen": {"color":"white",
                                "width":3,
                                "style":"dash",
                                "capstyle":"round",
                                "joinstyle":"round"} } )
    drawcmnds.append( { "action":"endView" } )
    drawcmnds.append( { "action":"show" } )
    drawcmnds.append( { "action":"save",
                        "filename":"test.png",
                        "vectsize":{"width":7.0, "height":7.0},
                        "rastsize":{"width":750, "height":750} } )
    drawcmnds.append( { "action":"exit" } )

    cmndrecvpipe, cmndsendpipe = Pipe(False)
    rspdrecvpipe, rspdsendpipe = Pipe(False)
    proc = PipedNoDisplayPQProcess(cmndrecvpipe, rspdsendpipe)
    proc.start()

    for testcmnd in drawcmnds:
        print "Command: %s" % str(testcmnd)
        cmndsendpipe.send(testcmnd)
        sleep(0.5)
        while rspdrecvpipe.poll():
            print "Response: %s" % str(rspdrecvpipe.recv())
    
    cmndsendpipe.close()
    sleep(1)
    while rspdrecvpipe.poll():
        print "Response: %s" % str(rspdrecvpipe.recv())
    rspdrecvpipe.close()
    cmndrecvpipe.close()
    rspdsendpipe.close()

    proc.join(None)
    SystemExit(proc.exitcode)
