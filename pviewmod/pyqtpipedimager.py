'''
PyQtPipedImager is a graphics viewer application written in PyQt4
that receives its drawing and other commands primarily from another
application through a pipe.  A limited number of commands are
provided by the viewer itself to allow saving and some manipulation
of the displayed image.  The controlling application, however, may
be unaware of these modifications made to the image.

PyQtPipedImagerProcess is used to create and run a PyQtPipedImager.

This package was developed by the Thermal Modeling and Analysis
Project (TMAP) of the National Oceanographic and Atmospheric
Administration's (NOAA) Pacific Marine Environmental Lab (PMEL).
'''

import sip
try:
    sip.setapi('QVariant', 2)
except AttributeError:
    pass

from PyQt4.QtCore import Qt, QPointF, QRectF, QSize, QString, QTimer
from PyQt4.QtGui  import QAction, QApplication, QBrush, QColor, QDialog, \
                         QFileDialog, QImage, QLabel, QMainWindow, \
                         QMessageBox, QPainter, QPalette, QPen, QPixmap, \
                         QPolygonF, QPushButton, QScrollArea

try:
    from PyQt4.QtGui import QTransform
    HAS_QTransform = True
except ImportError:
    from PyQt4.QtGui import QMatrix
    HAS_QTransform = False

from pyqtcmndhelper import PyQtCmndHelper
from pyqtscaledialog import PyQtScaleDialog
from multiprocessing import Pipe, Process
import sys
import time
import os


class PyQtPipedImager(QMainWindow):
    '''
    A PyQt graphics viewer that receives generic drawing commands
    through a pipe.  The image is recorded in a QImage, which is
    then used for displaying, manipulating, and saving.

    A drawing command is a dictionary with string keys that will be
    interpreted into the appropriate PyQt command(s).  For example,
        { "action":"drawText",
          "text":"Hello",
          "font":{"family":"Times", "size":100, "italic":True},
          "fill":{"color":0x880000, "style":"cross"},
          "outline":{"color":"black"},
          "location":(250,350) }

    The command { "action":"exit" } will shutdown the viewer and is
    the only way the viewer can be closed.  GUI actions can only hide
    the viewer.
    '''

    def __init__(self, cmndpipe, rspdpipe):
        '''
        Create a PyQt viewer which reads commands from the Pipe
        cmndpipe and writes responses back to rspdpipe.
        '''
        super(PyQtPipedImager, self).__init__()
        self.__cmndpipe = cmndpipe
        self.__rspdpipe = rspdpipe
        # default scene size = sqrt(0.75) * (1280, 1024)
        self.__scenewidth = 1110
        self.__sceneheight = 890
        # initial default color for clearScene (transparent white, but
        # with premultiplied alpha, the color here does not matter)
        self.__lastclearcolor = QColor(0xFFFFFF)
        self.__lastclearcolor.setAlpha(0)
        # QImage containing the scene
        self.__sceneimage = None
        # QPainter attached to the scene QImage
        self.__activepainter = None
        # Antialias when drawing?
        self.__antialias = False
        # data for recreating the current view
        self.__fracsides = None
        self.__usersides = None
        self.__clipit = True
        # number of drawing commands in the active painter
        self.__drawcount = 0
        # Limit the number of drawing commands between updates
        # to the displayed scene (if there is one) to avoid the
        # appearance of being "stuck".
        self.__maxdraws = 512
        # maximum user Y coordinate - used by adjustPoint
        self.__userymax = 1.0
        # scaling (zoom) factor for creating the displayed scene
        self.__scalefactor = 1.0
        # used to decide if the display needs to be updated 
        self.__displayisstale = False
        # create the label, that will serve as the canvas, in a scrolled area
        self.__scrollarea = QScrollArea(self)
        self.__label = QLabel(self.__scrollarea)
        # set the initial label size and other values for the scrolled area
        self.__label.setMinimumSize(self.__scenewidth, self.__sceneheight)
        self.__label.resize(self.__scenewidth, self.__sceneheight)
        # setup the scrolled area
        self.__scrollarea.setWidget(self.__label)
        self.__scrollarea.setBackgroundRole(QPalette.Dark)
        self.setCentralWidget(self.__scrollarea)
        self.__minsize = 128
        # command helper object
        self.__helper = PyQtCmndHelper(self)
        # create the scene image
        self.initializeScene()
        # Create the menubar
        self.createActions()
        self.createMenus()
        self.__lastfilename = ""
        self.__shuttingdown = False
        # Put in a default message in the statusbar
        self.statusBar().showMessage("Ready")
        # Set the initial size of the viewer
        mwwidth = self.__scenewidth + 8
        mwheight = self.__sceneheight + 8 + self.menuBar().height() + \
                                            self.statusBar().height()
        self.resize(mwwidth, mwheight)
        # check the command queue any time there are no window events to deal with
        self.__timer = QTimer(self)
        self.__timer.timeout.connect(self.checkCommandPipe)
        self.__timer.setInterval(0)
        self.__timer.start()

    def createActions(self):
        '''
        Create the actions used by the menus in this viewer.  Ownership
        of the actions are not transferred in addAction, thus the need
        to maintain references here.
        '''
        self.__saveact = QAction(self.tr("&Save"), self,
                                shortcut=self.tr("Ctrl+S"),
                                statusTip=self.tr("Save the scene to file"),
                                triggered=self.inquireSaveFilename)
        self.__scaleact = QAction(self.tr("Sc&ale"), self,
                                shortcut=self.tr("Ctrl+A"),
                                statusTip=self.tr("Scale the scene (canvas and drawn images change)"),
                                triggered=self.inquireSceneScale)
        self.__updateact = QAction(self.tr("&Update"), self,
                                shortcut=self.tr("Ctrl+U"),
                                statusTip=self.tr("Update the scene to the current content"),
                                triggered=self.updateScene)
        self.__hideact = QAction(self.tr("&Hide"), self,
                                shortcut=self.tr("Ctrl+H"),
                                statusTip=self.tr("Hide the viewer"),
                                triggered=self.hide)
        self.__aboutact = QAction(self.tr("&About"), self,
                                statusTip=self.tr("Show information about this viewer"),
                                triggered=self.aboutMsg)
        self.__aboutqtact = QAction(self.tr("About &Qt"), self,
                                statusTip=self.tr("Show information about the Qt library"),
                                triggered=self.aboutQtMsg)
        self.__exitact = QAction(self.tr("E&xit"), self,
                                statusTip=self.tr("Shut down the viewer"),
                                triggered=self.exitViewer)

    def createMenus(self):
        '''
        Create the menu items for the viewer
        using the previously created actions.
        '''
        menuBar = self.menuBar()
        sceneMenu = menuBar.addMenu(menuBar.tr("&Scene"))
        sceneMenu.addAction(self.__saveact)
        sceneMenu.addAction(self.__scaleact)
        sceneMenu.addAction(self.__updateact)
        sceneMenu.addSeparator()
        sceneMenu.addAction(self.__hideact)
        helpMenu = menuBar.addMenu(menuBar.tr("&Help"))
        helpMenu.addAction(self.__aboutact)
        helpMenu.addAction(self.__aboutqtact)
        helpMenu.addSeparator()
        helpMenu.addAction(self.__exitact)

    def showEvent(self, event):
        '''
        When the viewer is going to be shown, make sure
        the displayed scene is current.
        '''
        # update, ignoring the visibility flags
        self.displayScene(True)
        event.accept()

    def closeEvent(self, event):
        '''
        Override so the viewer cannot be closed from the
        user selecting the windowframe close ('X') button.
        Instead only hide the window.
        '''
        if self.__shuttingdown:
            event.accept()
        else:
            event.ignore()
            self.hide()

    def exitViewer(self):
        '''
        Close and exit the viewer.
        '''
        self.__timer.stop()
        self.__shuttingdown = True
        self.close()

    def aboutMsg(self):
        QMessageBox.about(self, self.tr("About PyQtPipedImager"),
            self.tr("\n" \
            "PyQtPipedImager is a graphics viewer application that " \
            "receives its drawing and other commands primarily from " \
            "another application through a pipe.  A limited number " \
            "of commands are provided by the viewer itself to allow " \
            "saving and some manipulation of the displayed scene.  " \
            "The controlling application, however, may be unaware " \
            "of these modifications made to the scene. " \
            "\n\n" \
            "Normally, the controlling program will exit the viewer " \
            "when it is no longer needed.  The Help -> Exit menu item " \
            "should not normally be used.  It is provided when problems " \
            "occur and the controlling program cannot shut down the " \
            "viewer properly. " \
            "\n\n" \
            "PyQtViewer was developed by the Thermal Modeling and Analysis " \
            "Project (TMAP) of the National Oceanographic and Atmospheric " \
            "Administration's (NOAA) Pacific Marine Environmental Lab (PMEL). "))

    def aboutQtMsg(self):
        QMessageBox.aboutQt(self, self.tr("About Qt"))

    def displayScene(self, ignorevis):
        '''
        Update the displayed scene from the scene image.
        If ignorevis is True, the update will be done
        even if the viewer is not visible; otherwise
        drawing to the scene label is only done if the
        viewer is visible.
        '''
        if not self.__displayisstale:
            # nothing new since the last display, ignore this call
            return
        if not ignorevis:
            if self.isHidden() or self.isMinimized():
                # not shown, so do not waste time drawing
                return
        if self.__scalefactor == 1.0:
            # use the scene image directly
            image = self.__sceneimage
        else:
            # scale the image
            image = self.__sceneimage.scaled(
                        int(self.__scenewidth * self.__scalefactor + 0.5),
                        int(self.__sceneheight * self.__scalefactor + 0.5),
                        Qt.KeepAspectRatio, Qt.SmoothTransformation )
        # set the label pixmap from the image
        self.__label.setPixmap(QPixmap.fromImage(image))
        # make sure the label knows to redraw
        self.__label.update()
        # mark that the displayed scene is current
        self.__displayisstale = False

    def initializeScene(self):
        '''
        Create and initialize the scene image
        '''
        # End any active View but do not update the scene
        if self.__activepainter:
            self.endView(False)
            restartview = True
        else:
            restartview = False
        self.__sceneimage = QImage(self.__scenewidth, self.__sceneheight,
                                   QImage.Format_ARGB32_Premultiplied)
        fillint = self.__helper.computeARGB32PreMultInt(self.__lastclearcolor)
        self.__sceneimage.fill(fillint)
        self.__displayisstale = True
        # If there was an active View, restart it
        if restartview:
            self.beginViewFromSides(self.__fracsides, self.__usersides,
                                    self.__clipit)

    def clearScene(self, colorinfo):
        '''
        Fills the scene with the color described in the colorinfo
        dictionary.  If colorinfo is None, or if no color or an
        invalid color is specified in this dictionary, the color
        used is the one used from the last clearScene call (or
        transparent white if a color has never been specified).
        '''
        # End any active View but do not update the scene
        if self.__activepainter:
            self.endView(False)
            restartview = True
        else:
            restartview = False
        # get the color to use for clearing (the background color)
        if colorinfo:
            try :
                mycolor = self.__helper.getColorFromCmnd(colorinfo)
                if mycolor.isValid():
                    self.__lastclearcolor = mycolor
            except KeyError:
                pass
        # Clear the image with the clearing color
        fillint = self.__helper.computeARGB32PreMultInt(self.__lastclearcolor)
        self.__sceneimage.fill(fillint)
        self.__displayisstale = True
        # Update the displayed scene if appropriate
        self.displayScene(False)
        # If there was an active View, restart it
        if restartview:
            self.beginViewFromSides(self.__fracsides, self.__usersides,
                                    self.__clipit)

    def resizeScene(self, width, height):
        '''
        Clear and resize the scene to the given width and height
        in units of 0.001 inches.
        '''
        newwidth = int(width * 0.001 * self.physicalDpiX() + 0.5)
        if newwidth < self.__minsize:
            newwidth = self.__minsize
        newheight = int(height * 0.001 * self.physicalDpiY() + 0.5)
        if newheight < self.__minsize:
            newheight = self.__minsize
        if (newwidth != self.__scenewidth) or (newheight != self.__sceneheight):
            # Resize the label and set label values
            # so the scrollarea knows of the new size
            labelwidth = int(newwidth * self.__scalefactor + 0.5)
            labelheight = int(newheight * self.__scalefactor + 0.5)
            self.__label.setMinimumSize(labelwidth, labelheight)
            self.__label.resize(labelwidth, labelheight)
            # Create and clear the new scene image
            self.__scenewidth = newwidth
            self.__sceneheight = newheight
            self.initializeScene()
            # Display the new scene, if appropriate
            self.displayScene(False)

    def inquireSceneScale(self):
        '''
        Prompt the user for the desired scaling factor for the scene.
        '''
        scaledlg = PyQtScaleDialog(self.tr("Scene Size Scaling"),
                       self.tr("Scaling factor (both horiz. and vert.) for the scene"),
                       self.__scalefactor, 
                       self.__scenewidth * self.__scalefactor,
                       self.__sceneheight * self.__scalefactor,
                       self.__minsize, self.__minsize, self)
        if scaledlg.exec_():
            (newscale, okay) = scaledlg.getValues()
            if okay:
                self.scaleScene(newscale)

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
            # Resize the label and set label values
            # so the scrollarea knows of the new size
            self.__label.setMinimumSize(newlabwidth, newlabheight)
            self.__label.resize(newlabwidth, newlabheight)
            # Redisplay the scene, if appropriate
            self.__displayisstale = True
            self.displayScene(False)

    def inquireSaveFilename(self):
        '''
        Prompt the user for the name of the file into which to save the scene.
        The file format will be determined from the filename extension.
        '''
        formattypes = [ ( "png",
                          self.tr("PNG - Portable Networks Graphics (*.png)") ),
                        ( "jpeg",
                          self.tr("JPEG - Joint Photographic Experts Group (*.jpeg *.jpg *.jpe)") ),
                        ( "tiff",
                          self.tr("TIFF - Tagged Image File Format (*.tiff *.tif)") ),
                        ( "bmp",
                          self.tr("BMP - Windows Bitmap (*.bmp)") ),
                        ( "ppm",
                          self.tr("PPM - Portable Pixmap (*.ppm)") ),
                        ( "xpm",
                          self.tr("XPM - X11 Pixmap (*.xpm)") ),
                        ( "xbm",
                          self.tr("XBM - X11 Bitmap (*.xbm)") ), ]
        # tr returns QStrings so the following does not work
        # filters = ";;".join( [ t[1] for t in formattypes ] )
        filters = QString(formattypes[0][1])
        for typePair in formattypes[1:]:
            filters.append(";;")
            filters.append(typePair[1])
        (fileName, fileFilter) = QFileDialog.getSaveFileNameAndFilter(self,
                                        self.tr("Save the current scene as "),
                                        self.__lastfilename, filters)
        if fileName:
            for (fmt, fmtQName) in formattypes:
                if fmtQName.compare(fileFilter) == 0:
                    fileFormat = fmt
                    break
            else:
                raise RuntimeError( self.tr("Unexpected file format name '%1'") \
                                        .arg(fileFilter) )
            if (fileFormat == "png") or \
               (fileFormat == "tiff") or \
               (fileFormat == "xpm"):
                transparentbkg = True
            else:
                transparentbkg = False
            self.saveSceneToFile(fileName, fileFormat, transparentbkg)
            self.__lastfilename = fileName

    def saveSceneToFile(self, filename, imageformat, transparentbkg):
        '''
        Save the current scene to the named file.  If imageformat
        is empty or None, the format is guessed from the filename
        extension.

        If transparentbkg is False, the last clearing color, as an
        opaque color, is drawn to the image before drawing the
        current scene.

        If transparentbkg is True and the scaling factor is 1.0,
        the current scene image is used directly for saving.
        '''
        # check for and explicitly mark gif format
        if not imageformat:
            fileext = ( os.path.splitext(filename)[1] ).lower()
            if fileext == '.gif':
                myformat = 'gif'
            else:
                # let QImage figure out the format
                myformat = None
        else:
            myformat = imageformat.lower()
        # Silently convert gif filename and format to png
        if myformat == 'gif':
            myformat = 'png'
            myfilename = os.path.splitext(filename)[0] + ".png"
        else:
            myfilename = filename
        # Get the image to save to file
        if transparentbkg and (self.__scalefactor == 1.0):
            # Can use the scene image directly
            image = self.__sceneimage
        else:
            # Create a new image from the scene image
            newsize = QSize(int(self.__scenewidth * self.__scalefactor + 0.5),
                            int(self.__sceneheight * self.__scalefactor + 0.5))
            image = QImage(newsize, QImage.Format_ARGB32_Premultiplied)
            if transparentbkg:
                # Initialize with transparent
                fillint = 0
            else:
                # Initialize with an opaque version of the last clearing color
                opaquecolor = QColor(self.__lastclearcolor)
                opaquecolor.setAlpha(255)
                fillint = self.__helper.computeARGB32PreMultInt(opaquecolor)
            image.fill(fillint)
            # paint the scene to this QImage
            painter = QPainter(image)
            trgrect = QRectF(0.0, 0.0, newsize.width(), newsize.height())
            srcrect = QRectF(0.0, 0.0, self.__scenewidth, self.__sceneheight)
            painter.drawImage(trgrect, self.__sceneimage, srcrect, Qt.AutoColor)
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
        cmndact = cmnd["action"]
        if cmndact == "clear":
            self.clearScene(cmnd)
        elif cmndact == "exit":
            self.exitViewer()
        elif cmndact == "hide":
            self.hide()
        elif cmndact == "dpi":
            windowdpi = ( self.physicalDpiX(), self.physicalDpiY() )
            self.__rspdpipe.send(windowdpi)
        elif cmndact == "update":
            self.updateScene()
        elif cmndact == "resize":
            mysize = self.__helper.getSizeFromCmnd(cmnd)
            self.resizeScene(mysize.width(), mysize.height())
        elif cmndact == "save":
            filename = cmnd["filename"]
            fileformat = cmnd.get("fileformat", None)
            transparentbkg = cmnd.get("transparentbkg", False)
            self.saveSceneToFile(filename, fileformat, transparentbkg)
        elif cmndact == "setTitle":
            self.setWindowTitle(cmnd["title"])
        elif cmndact == "show":
            self.showNormal()
        elif cmndact == "beginView":
            self.beginView(cmnd)
        elif cmndact == "clipView":
            self.clipView(cmnd)
        elif cmndact == "endView":
            self.endView(True)
        elif cmndact == "drawMultiline":
            self.drawMultiline(cmnd)
        elif cmndact == "drawPoints":
            self.drawPoints(cmnd)
        elif cmndact == "drawPolygon":
            self.drawPolygon(cmnd)
        elif cmndact == "drawRectangle":
            self.drawRectangle(cmnd)
        elif cmndact == "drawMulticolorRectangle":
            self.drawMulticolorRectangle(cmnd)
        elif cmndact == "drawText":
            self.drawSimpleText(cmnd)
        else:
            raise ValueError( self.tr("Unknown command action %1") \
                                  .arg(str(cmndact)) )

    def beginView(self, cmnd):
        '''
        Setup a new viewport and window for drawing on a portion
        (possibly all) of the scene.  Recognized keys from cmnd
        are:
            "viewfracs": a dictionary of sides positions (see
                    PyQtCmndHelper.getSidesFromCmnd) giving the
                    fractions [0.0, 1.0] of the way through the
                    scene for the sides of the new View.
            "usercoords": a dictionary of sides positions (see
                    PyQtCmndHelper.getSidesFromCmnd) giving the
                    user coordinates for the sides of the new View.
            "clip": clip to the new View? (default: True)

        Note that the view fraction values are based on (0,0) being the
        bottom left corner and (1,1) being the top right corner.  Thus,
        left < right and bottom < top.

        Raises a KeyError if either the "viewfracs" or the "usercoords"
        key is not given.
        '''
        # Get the view rectangle in fractions of the full scene
        fracsides = self.__helper.getSidesFromCmnd(cmnd["viewfracs"])
        # Get the user coordinates for this view rectangle
        usersides = self.__helper.getSidesFromCmnd(cmnd["usercoords"])
        # Should graphics be clipped to this view?
        try:
            clipit = cmnd["clip"]
        except KeyError:
            clipit = True
        self.beginViewFromSides(fracsides, usersides, clipit)

    def beginViewFromSides(self, fracsides, usersides, clipit):
        '''
        Setup a new viewport and window for drawing on a portion
        (possibly all) of the scene.  The view in fractions of
        the full scene are given in fracsides.  The user coordinates
        for this view are given in usersides.  Sets the clipping
        rectangle to this view.  If clipit is True, graphics
        will be clipped to this view.
        '''
        # If a view is still active, automatically end it
        if self.__activepainter:
            self.endView(True)
        # Get the location for the new view in terms of pixels.
        width = float(self.__scenewidth)
        height = float(self.__sceneheight)
        leftpixel = fracsides.left() * width
        rightpixel = fracsides.right() * width
        bottompixel = fracsides.bottom() * height
        toppixel = fracsides.top() * height
        # perform the checks after turning into units of pixels
        # to make sure the values are significantly different
        if (0.0 > leftpixel) or (leftpixel >= rightpixel) or (rightpixel > width):
            raise ValueError( self.tr("Invalid left, right view fractions: " \
                                      "left in pixels = %1, right in pixels = %2") \
                                  .arg(str(leftpixel)).arg(str(rightpixel)) )
        if (0.0 > bottompixel) or (bottompixel >= toppixel) or (toppixel > height):
            raise ValueError( self.tr("Invalid bottom, top view fractions: " \
                                      "bottom in pixels = %1, top in pixels = %2") \
                                  .arg(str(bottompixel)).arg(str(toppixel)) )
        # Create the view rectangle in device coordinates
        vrectf = QRectF(leftpixel, height - toppixel,
                       rightpixel - leftpixel, toppixel - bottompixel)
        # Get the user coordinates for this view rectangle
        leftcoord = usersides.left()
        rightcoord = usersides.right()
        bottomcoord = usersides.bottom()
        topcoord = usersides.top()
        if leftcoord >= rightcoord:
            raise ValueError( self.tr("Invalid left, right user coordinates: " \
                                      "left = %1, right = %2") \
                                  .arg(str(leftcoord)).arg(str(rightcoord)) )
        if bottomcoord >= topcoord:
            raise ValueError( self.tr("Invalid bottom, top user coordinates: " \
                                      "bottom = %1, top = %2") \
                                  .arg(str(bottomcoord)).arg(str(topcoord)) )
        # Create the view rectangle in user (world) coordinates
        # adjustPoint will correct for the flipped, zero-based Y coordinate
        wrectf = QRectF(leftcoord, 0.0, rightcoord - leftcoord, topcoord - bottomcoord)
        # Compute the entries in the transformation matrix
        m11 = vrectf.width() / wrectf.width()
        m12 = 0.0
        m21 = 0.0
        m22 = vrectf.height() / wrectf.height()
        dx = vrectf.left() - (m11 * wrectf.left())
        dy = vrectf.top() - (m22 * wrectf.top())
        # Create the new painter, and set the view transformation
        self.__activepainter = QPainter(self.__sceneimage)
        self.__activepainter.save()
        # Set the viewport and window just to be safe
        self.__activepainter.setViewport(0, 0, self.__scenewidth, self.__sceneheight)
        self.__activepainter.setWindow(0, 0, self.__scenewidth, self.__sceneheight)
        # Assign the transformation to take the user coordinates to device coordinates
        if HAS_QTransform:
            wvtrans = QTransform(m11, m12, m21, m22, dx, dy)
            self.__activepainter.setWorldTransform(wvtrans, True)
        else:
            wvtrans = QMatrix(m11, m12, m21, m22, dx, dy)
            self.__activepainter.setWorldMatrix(wvtrans, True)
        self.__activepainter.setWorldMatrixEnabled(True)
        # Set the clip rectangle to that of the view; this also activates clipping
        self.__activepainter.setClipRect(wrectf, Qt.ReplaceClip)
        # Disable clipping if not desired at this time
        if not clipit:
            self.__activepainter.setClipping(False)
        # Note that __activepainter has to end before drawings will show up.
        self.__drawcount = 0
        # Save the current view sides and clipit setting for recreating the view.
        # Just save the original objects (assume calling functions do not keep them)
        self.__fracsides = fracsides
        self.__usersides = usersides
        self.__clipit = clipit
        # Pull out the top coordinate since this is used a lot (via adjustPoint)
        self.__userymax = usersides.top()

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

    def endView(self, update):
        '''
        Ends the current view and appends it to the list of pictures
        drawn in the scene.  If update is True and something was drawn
        in this view, the new scene is displayed, if appropriate.
        '''
        self.__activepainter.restore()
        self.__activepainter.end()
        self.__activepainter = None
        # Only update if something was drawn and update requested
        if self.__drawcount > 0:
            self.__drawcount = 0
            self.__displayisstale = True
            if update:
                # update the scene
                self.displayScene(False)
        self.__activepicture = None

    def updateScene(self):
        '''
        Updates the displayed graphics to include all drawn elements.
        '''
        # If there is an active painter on the image,
        # end the view, thus updating the display,
        # then restart the view.
        if self.__drawcount > 0:
            self.endView(True)
            self.beginViewFromSides(self.__fracsides, self.__usersides,
                                    self.__clipit)

    def drawMultiline(self, cmnd):
        '''
        Draws a collection of connected line segments.

        Recognized keys from cmnd:
            "points": consecutive endpoints of the connected line
                    segments as a list of (x, y) coordinates
            "pen": dictionary describing the pen used to draw the
                    segments (see PyQtCmndHelper.getPenFromCmnd)

        The coordinates are user coordinates from the bottom left corner.

        Raises:
            KeyError if the "points" or "pen" key is not given
            ValueError if there are fewer than two endpoints given
        '''
        ptcoords = cmnd["points"]
        if len(ptcoords) < 2:
            raise ValueError("fewer that two endpoints given")
        adjpts = [ self.adjustPoint(xypair) for xypair in ptcoords ]
        endpts = QPolygonF( [ QPointF(xypair[0], xypair[1]) \
                                  for xypair in adjpts ] )
        mypen = self.__helper.getPenFromCmnd(cmnd["pen"])
        # save the default state of the painter
        self.__activepainter.save()
        try:
            self.__activepainter.setRenderHint(QPainter.Antialiasing,
                                               self.__antialias)
            self.__activepainter.setPen(mypen)
            self.__activepainter.drawPolyline(endpts)
            self.__drawcount += 1
        finally:
            # return the painter to the default state
            self.__activepainter.restore()
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
                    (see PyQtCmndHelper.getSymbolFromCmnd)
            "size": size of the symbol (scales with view size)
            "color": color name or 24-bit RGB integer value (eg, 0xFF0088)
            "alpha": alpha value from 0 (transparent) to 255 (opaque)

        The coordinates are user coordinates from the bottom left corner.

        Raises a KeyError if the "symbol", "points", or "size" key
        is not given.
        '''
        ptcoords = cmnd["points"]
        ptsize = cmnd["size"]
        sympath = self.__helper.getSymbolFromCmnd(cmnd["symbol"])
        # save the default state of the painter
        self.__activepainter.save()
        try:
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
                mypen = QPen(mybrush, 15.0, Qt.SolidLine,
                             Qt.RoundCap, Qt.RoundJoin)
                self.__activepainter.setPen(mypen)
            scalefactor = ptsize / 100.0
            for xyval in ptcoords:
                (adjx, adjy) = self.adjustPoint( xyval )
                self.__activepainter.save()
                try:
                    self.__activepainter.translate(adjx, adjy)
                    self.__activepainter.scale(scalefactor, scalefactor)
                    self.__activepainter.drawPath(sympath.painterPath())
                finally:
                    self.__activepainter.restore()
            self.__drawcount += len(ptcoords)
        finally:
            # return the painter to the default state
            self.__activepainter.restore()
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
                    polygon; see PyQtCmndHelper.getBrushFromCmnd
                    If not given, the polygon will not be filled.
            "outline": dictionary describing the pen used to outline
                    the polygon; see PyQtCmndHelper.getPenFromCmnd
                    If not given, the polygon border will not be drawn.

        The coordinates are user coordinates from the bottom left corner.

        Raises a KeyError if the "points" key is not given.
        '''
        mypoints = cmnd["points"]
        adjpoints = [ self.adjustPoint(xypair) for xypair in mypoints ]
        mypolygon = QPolygonF( [ QPointF(xypair[0], xypair[1]) \
                                     for xypair in adjpoints ] )
        # save the default state of the painter
        self.__activepainter.save()
        try:
            self.__activepainter.setRenderHint(QPainter.Antialiasing,
                                               self.__antialias)
            try:
                mybrush = self.__helper.getBrushFromCmnd(cmnd["fill"])
            except KeyError:
                mybrush = Qt.NoBrush
            try:
                mypen = self.__helper.getPenFromCmnd(cmnd["outline"])
            except KeyError:
                if ( mybrush == Qt.NoBrush ):
                    raise ValueError( self.tr('drawPolygon called without a Brush or Pen') )
                # Use a cosmetic Pen matching the brush
                mypen = QPen(mybrush, 0.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            self.__activepainter.setBrush(mybrush)
            self.__activepainter.setPen(mypen)
            self.__activepainter.drawPolygon(mypolygon)
            self.__drawcount += 1
        finally:
            # return the painter to the default state
            self.__activepainter.restore()
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
                    rectangle; see PyQtCmndHelper.getBrushFromCmnd
                    If not given, the rectangle will not be filled.
            "outline": dictionary describing the pen used to outline
                    the rectangle; see PyQtCmndHelper.getPenFromCmnd
                    If not given, the rectangle border will not be drawn.

        The coordinates are user coordinates from the bottom left corner.

        Raises a ValueError if the width or height of the rectangle
        is not positive.
        '''
        # get the left, bottom, right, and top values
        # any keys not given get a zero value
        sides = self.__helper.getSidesFromCmnd(cmnd)
        # adjust to actual view coordinates from the top left
        lefttop = self.adjustPoint( (sides.left(), sides.top()) )
        rightbottom = self.adjustPoint( (sides.right(), sides.bottom()) )
        width = rightbottom[0] - lefttop[0]
        if width <= 0.0:
            raise ValueError("width of the rectangle in not positive")
        height = rightbottom[1] - lefttop[1]
        if height <= 0.0:
            raise ValueError("height of the rectangle in not positive")
        myrect = QRectF(lefttop[0], lefttop[1], width, height)
        # save the default state of the painter
        self.__activepainter.save()
        try:
            self.__activepainter.setRenderHint(QPainter.Antialiasing,
                                               self.__antialias)
            try:
                mypen = self.__helper.getPenFromCmnd(cmnd["outline"])
                self.__activepainter.setPen(mypen)
            except KeyError:
                self.__activepainter.setPen(Qt.NoPen)
            try:
                mybrush = self.__helper.getBrushFromCmnd(cmnd["fill"])
                self.__activepainter.setBrush(mybrush)
            except KeyError:
                self.__activepainter.setBrush(Qt.NoBrush)
            self.__activepainter.drawRect(myrect)
            self.__drawcount += 1
        finally:
            # return the painter to the default state
            self.__activepainter.restore()
        # Limit the number of drawing commands per picture
        if self.__drawcount >= self.__maxdraws:
            self.updateScene()

    def drawMulticolorRectangle(self, cmnd):
        '''
        Draws a multi-colored rectangle in the current view using
        the information in the dictionary cmnd.

        Recognized keys from cmnd:
            "left": x-coordinate of left edge of the rectangle
            "bottom": y-coordinate of the bottom edge of the rectangle
            "right": x-coordinate of the right edge of the rectangle
            "top": y-coordinate of the top edge of the rectangle
            "numrows": the number of equally spaced rows
                    to subdivide the rectangle into
            "numcols": the number of equally spaced columns
                    to subdivide the rectangle into
            "colors": iterable representing a flattened column-major
                    2-D array of color dictionaries
                    (see PyQtCmndHelper.getcolorFromCmnd) which are
                    used to create solid brushes to fill each of the
                    cells.  The first row is at the top; the first
                    column is on the left.

        The coordinates are user coordinates from the bottom left corner.

        Raises:
            KeyError: if the "numrows", "numcols", or "colors" keys
                    are not given; if the "color" key is not given
                    in a color dictionary
            ValueError: if the width or height of the rectangle is
                    not positive; if the value of the "numrows" or
                    "numcols" key is not positive; if a color
                    dictionary does not produce a valid color
            IndexError: if not enough colors were given
        '''
        # get the left, bottom, right, and top values
        # any keys not given get a zero value
        sides = self.__helper.getSidesFromCmnd(cmnd)
        # adjust to actual view coordinates from the top left
        lefttop = self.adjustPoint( (sides.left(), sides.top()) )
        rightbottom = self.adjustPoint( (sides.right(), sides.bottom()) )
        width = rightbottom[0] - lefttop[0]
        if width <= 0.0:
            raise ValueError("width of the rectangle in not positive")
        height = rightbottom[1] - lefttop[1]
        if height <= 0.0:
            raise ValueError("height of the rectangle in not positive")
        numrows = int( cmnd["numrows"] + 0.5 )
        if numrows < 1:
            raise ValueError("numrows not a positive integer value")
        numcols = int( cmnd["numcols"] + 0.5 )
        if numcols < 1:
            raise ValueError("numcols not a positive integer value")
        colors = [ self.__helper.getColorFromCmnd(colorinfo) \
                                 for colorinfo in cmnd["colors"] ]
        if len(colors) < (numrows * numcols):
            raise IndexError("not enough colors given")

        # save the default state of the painter
        self.__activepainter.save()
        try:
            self.__activepainter.setRenderHint(QPainter.Antialiasing,
                                               self.__antialias)
            self.__activepainter.setPen(Qt.NoPen)
            width = width / float(numcols)
            height = height / float(numrows)
            myrect = QRectF(lefttop[0], lefttop[1], width, height)
            colorindex = 0
            for j in xrange(numcols):
                myrect.moveLeft(lefttop[0] + j * width)
                for k in xrange(numrows):
                    myrect.moveTop(lefttop[1] + k * height)
                    mybrush = QBrush(colors[colorindex], Qt.SolidPattern)
                    colorindex += 1
                    self.__activepainter.setBrush(mybrush)
                    self.__activepainter.drawRect(myrect)
            self.__drawcount += numcols * numrows
        finally:
            # return the painter to the default state
            self.__activepainter.restore()
        # Limit the number of drawing commands per picture
        if self.__drawcount >= self.__maxdraws:
            self.updateScene()

    def drawSimpleText(self, cmnd):
        '''
        Draws a "simple" text item in the current view.
        Raises a KeyError if the "text" key is not given.

        Recognized keys from cmnd:
            "text": string to displayed
            "font": dictionary describing the font to use;  see
                    PyQtCmndHelper.getFontFromCmnd.  If not given
                    the default font for this viewer is used.
            "fill": dictionary describing the pen used to draw the
                    text; see PyQtCmndHelper.getPenFromCmnd.
                    If not given, the default pen for this viewer
                    is used.
            "rotate": clockwise rotation of the text in degrees
            "location": (x,y) location (user coordinates) in the
                    current view window for the baseline of the
                    start of text.
        '''
        mytext = cmnd["text"]
        try:
            startpt = cmnd["location"]
            (xpos, ypos) = self.adjustPoint(startpt)
        except KeyError:
            # Almost certainly an error, so put it someplace
            # where it will be seen, hopefully as an error.
            winrect = self.__activepainter.window()
            xpos = winrect.width() / 2.0
            ypos = winrect.height() / 2.0
        # save the default state of the painter
        self.__activepainter.save()
        try:
            self.__activepainter.setRenderHint(QPainter.Antialiasing,
                                               self.__antialias)
            # Move the coordinate system so the origin is at the start
            # of the text so that rotation is about this point
            self.__activepainter.translate(xpos, ypos)
            try:
                myfont = self.__helper.getFontFromCmnd(cmnd["font"])
                self.__activepainter.setFont(myfont)
            except KeyError:
                pass
            try:
                rotdeg = cmnd["rotate"]
                self.__activepainter.rotate(rotdeg)
            except KeyError:
                pass
            try:
                mypen = self.__helper.getPenFromCmnd(cmnd["fill"])
                self.__activepainter.setPen(mypen)
            except KeyError:
                pass
            self.__activepainter.drawText(0, 0, mytext)
            self.__drawcount += 1
        finally:
            # return the painter to the default state
            self.__activepainter.restore()
        # Limit the number of drawing commands per picture
        if self.__drawcount >= self.__maxdraws:
            self.updateScene()

    def adjustPoint(self, xypair):
        '''
        Returns appropriate "view" window (logical) coordinates
        corresponding to the coordinate pair given in xypair
        obtained from a command.
        '''
        (xpos, ypos) = xypair
        ypos = self.__userymax - ypos
        return (xpos, ypos)


class PyQtPipedImagerProcess(Process):
    '''
    A Process specifically tailored for creating a PyQtPipedImager.
    '''
    def __init__(self, cmndpipe, rspdpipe):
        '''
        Create a Process that will produce a PyQtPipedImager
        attached to the given Pipes when run.
        '''
        Process.__init__(self)
        self.__cmndpipe = cmndpipe
        self.__rspdpipe = rspdpipe

    def run(self):
        '''
        Create a PyQtPipedImager that is attached
        to the Pipe of this instance.
        '''
        self.__app = QApplication(["PyQtPipedImager"])
        self.__viewer = PyQtPipedImager(self.__cmndpipe, self.__rspdpipe)
        result = self.__app.exec_()
        self.__cmndpipe.close()
        self.__rspdpipe.close()
        SystemExit(result)

#
# The following are for testing this (and the pyqtqcmndhelper) modules
#

class _PyQtCommandSubmitter(QDialog):
    '''
    Testing dialog for controlling the addition of commands to a pipe.
    Used for testing PyQtPipedImager in the same process as the viewer.
    '''
    def __init__(self, parent, cmndpipe, rspdpipe, cmndlist):
        '''
        Create a QDialog with a single QPushButton for controlling
        the submission of commands from cmndlist to cmndpipe.
        '''
        QDialog.__init__(self, parent)
        self.__cmndlist = cmndlist
        self.__cmndpipe = cmndpipe
        self.__rspdpipe = rspdpipe
        self.__nextcmnd = 0
        self.__button = QPushButton("Submit next command", self)
        self.__button.pressed.connect(self.submitNextCommand)
        self.show()

    def submitNextCommand(self):
        '''
        Submit the next command from the command list to the command pipe,
        or shutdown if there are no more commands to submit.
        '''
        try:
            print "Command: %s" % str(self.__cmndlist[self.__nextcmnd])
            self.__cmndpipe.send(self.__cmndlist[self.__nextcmnd])
            self.__nextcmnd += 1
            while self.__rspdpipe.poll():
                print "Response: %s" % str(self.__rspdpipe.recv())
        except IndexError:
            self.__rspdpipe.close()
            self.__cmndpipe.close()
            self.close()


if __name__ == "__main__":
    # vertices of a pentagon (roughly) centered in a 1000 x 1000 square
    pentagonpts = ( (504.5, 100.0), (100.0, 393.9),
                    (254.5, 869.4), (754.5, 869.4),
                    (909.0, 393.9),  )
    # create the list of commands to submit
    drawcmnds = []
    drawcmnds.append( { "action":"setTitle", "title":"Tester" } )
    drawcmnds.append( { "action":"show" } )
    drawcmnds.append( { "action":"clear", "color":0xFFFFFF} )
    drawcmnds.append( { "action":"dpi"} )
    drawcmnds.append( { "action":"resize",
                        "width":5000,
                        "height":5000 } )
    drawcmnds.append( { "action":"beginView",
                        "viewfracs":{"left":0.0, "bottom":0.5,
                                     "right":0.5, "top":1.0},
                        "usercoords":{"left":0, "bottom":0,
                                      "right":1000, "top":1000},
                        "clip":True } )
    drawcmnds.append( { "action":"drawRectangle",
                        "left": 50, "bottom":50,
                        "right":950, "top":950,
                        "fill":{"color":"black", "alpha":64},
                        "outline":{"color":"blue"} } )
    drawcmnds.append( { "action":"drawPolygon",
                        "points":pentagonpts,
                        "fill":{"color":"lightblue"},
                        "outline":{"color":"black",
                                   "width": 50,
                                   "style":"solid",
                                   "capstyle":"round",
                                   "joinstyle":"round" } } )
    drawcmnds.append( { "action":"drawText",
                        "text":"y=100",
                        "font":{"family":"Times", "size":200},
                        "fill":{"color":0x880000},
                        "location":(100,100) } )
    drawcmnds.append( { "action":"drawText",
                        "text":"y=300",
                        "font":{"family":"Times", "size":200},
                        "fill":{"color":0x880000},
                        "location":(100,300) } )
    drawcmnds.append( { "action":"drawText",
                        "text":"y=500",
                        "font":{"family":"Times", "size":200},
                        "fill":{"color":0x880000},
                        "location":(100,500) } )
    drawcmnds.append( { "action":"drawText",
                        "text":"y=700",
                        "font":{"family":"Times", "size":200},
                        "fill":{"color":0x880000},
                        "location":(100,700) } )
    drawcmnds.append( { "action":"endView" } )
    drawcmnds.append( { "action":"show" } )
    drawcmnds.append( { "action":"beginView",
                        "viewfracs":{"left":0.05, "bottom":0.05,
                                     "right":0.95, "top":0.95},
                        "usercoords":{"left":0, "bottom":0,
                                      "right":1000, "top":1000},
                        "clip":True } )
    drawcmnds.append( { "action":"drawMulticolorRectangle",
                        "left": 50, "bottom":50,
                        "right":950, "top":950,
                        "numrows":2, "numcols":3,
                        "colors":( {"color":0xFF0000, "alpha":128},
                                   {"color":0xAA8800, "alpha":128},
                                   {"color":0x00FF00, "alpha":128},
                                   {"color":0x008888, "alpha":128},
                                   {"color":0x0000FF, "alpha":128},
                                   {"color":0x880088, "alpha":128} ) } )
    drawcmnds.append( { "action":"drawText",
                        "text":"R",
                        "font":{"size":200, "bold": True},
                        "fill":{"color":"black"},
                        "rotate":-45,
                        "location":(200,600) } )
    drawcmnds.append( { "action":"drawText",
                        "text":"Y",
                        "font":{"size":200, "bold": True},
                        "fill":{"color":"black"},
                        "rotate":-45,
                        "location":(200,150) } )
    drawcmnds.append( { "action":"drawText",
                        "text":"G",
                        "font":{"size":200, "bold": True},
                        "fill":{"color":"black"},
                        "rotate":-45,
                        "location":(500,600) } )
    drawcmnds.append( { "action":"drawText",
                        "text":"C",
                        "font":{"size":200, "bold": True},
                        "fill":{"color":"black"},
                        "rotate":-45,
                        "location":(500,150) } )
    drawcmnds.append( { "action":"drawText",
                        "text":"B",
                        "font":{"size":200, "bold": True},
                        "fill":{"color":"black"},
                        "rotate":-45,
                        "location":(800,600) } )
    drawcmnds.append( { "action":"drawText",
                        "text":"M",
                        "font":{"size":200, "bold": True},
                        "fill":{"color":"black"},
                        "rotate":-45,
                        "location":(800,150) } )
    drawcmnds.append( { "action":"endView" } )
    drawcmnds.append( { "action":"show" } )
    drawcmnds.append( { "action":"beginView",
                        "viewfracs":{"left":0.0, "bottom":0.0,
                                     "right":1.0, "top":1.0},
                        "usercoords":{"left":0, "bottom":0,
                                      "right":1000, "top":1000},
                        "clip":True } )
    drawcmnds.append( { "action":"drawPoints",
                        "points":( (100, 100),
                                   (100, 300),
                                   (100, 500),
                                   (100, 700),
                                   (100, 900) ),
                        "symbol":".",
                        "size":50,
                        "color":"black" })
    drawcmnds.append( { "action":"drawPoints",
                        "points":( (200, 100),
                                   (200, 300),
                                   (200, 500),
                                   (200, 700),
                                   (200, 900) ),
                        "symbol":"o",
                        "size":50,
                        "color":"black" })
    drawcmnds.append( { "action":"drawPoints",
                        "points":( (300, 100),
                                   (300, 300),
                                   (300, 500),
                                   (300, 700),
                                   (300, 900) ),
                        "symbol":"+",
                        "size":50,
                        "color":"blue" })
    drawcmnds.append( { "action":"drawPoints",
                        "points":( (400, 100),
                                   (400, 300),
                                   (400, 500),
                                   (400, 700),
                                   (400, 900) ),
                        "symbol":"x",
                        "size":50,
                        "color":"black" })
    drawcmnds.append( { "action":"drawPoints",
                        "points":( (500, 100),
                                   (500, 300),
                                   (500, 500),
                                   (500, 700),
                                   (500, 900) ),
                        "symbol":"*",
                        "size":50,
                        "color":"black" })
    drawcmnds.append( { "action":"drawPoints",
                        "points":( (600, 100),
                                   (600, 300),
                                   (600, 500),
                                   (600, 700),
                                   (600, 900) ),
                        "symbol":"^",
                        "size":50,
                        "color":"blue" })
    drawcmnds.append( { "action":"drawPoints",
                        "points":( (700, 100),
                                   (700, 300),
                                   (700, 500),
                                   (700, 700),
                                   (700, 900) ),
                        "symbol":"#",
                        "size":50,
                        "color":"black" })
    drawcmnds.append( { "action":"drawMultiline",
                        "points":( (600, 100),
                                   (300, 300),
                                   (700, 500),
                                   (500, 700),
                                   (300, 500),
                                   (100, 900) ),
                        "pen": {"color":"white",
                                "width":10,
                                "style":"dash",
                                "capstyle":"round",
                                "joinstyle":"round"} } )
    drawcmnds.append( { "action":"endView" } )
    drawcmnds.append( { "action":"show" } )
    drawcmnds.append( { "action":"exit" } )
    # start PyQt
    app = QApplication(["PyQtPipedImager"])
    # create a PyQtPipedViewer in this process
    cmndrecvpipe, cmndsendpipe = Pipe(False)
    rspdrecvpipe, rspdsendpipe = Pipe(False)
    viewer = PyQtPipedImager(cmndrecvpipe, rspdsendpipe)
    # create a command submitter dialog
    tester = _PyQtCommandSubmitter(viewer, cmndsendpipe,
                                   rspdrecvpipe, drawcmnds)
    tester.show()
    # let it all run
    result = app.exec_()
    if result != 0:
        sys.exit(result)

