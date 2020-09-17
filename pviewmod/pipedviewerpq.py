'''
PipedViewerPQ is a graphics viewer application written in PyQt
that receives its drawing and other commands primarily from another
application through a pipe.  A limited number of commands are
provided by the viewer itself to allow saving and some manipulation
of the displayed image.  The controlling application, however, may
be unaware of these modifications made to the image.

PipedViewerPQProcess is used to create and run a PipedViewerPQ.

This package was developed by the Thermal Modeling and Analysis
Project (TMAP) of the National Oceanographic and Atmospheric
Administration's (NOAA) Pacific Marine Environmental Lab (PMEL).
'''

from __future__ import print_function

import sys
import os
import signal
import time
import math

# First try to import PySide2, then try PyQt5 if that fails, and finally try PyQt4 if that fails
try:
    import PySide2
    PYTHONQT_VERSION = 'PySide2'
except ImportError:
    try:
        import PyQt5
        PYTHONQT_VERSION = 'PyQt5'
    except ImportError:
        import PyQt4
        PYTHONQT_VERSION = 'PyQt4'

# Now that the Python Qt version is determined, import the parts
# allowing any import errors to propagate out
if PYTHONQT_VERSION == 'PySide2':
    from PySide2.QtCore    import Qt, QPointF, QRect, QRectF, QSize, QSizeF, QTimer
    from PySide2.QtGui     import QBrush, QColor, QFontMetricsF, QImage, QPainter, \
                                  QPalette, QPen, QPicture, QPixmap, QPolygonF, \
                                  QTextDocument
    from PySide2.QtWidgets import QAction, QApplication, QDialog, QFileDialog, QLabel, \
                                  QMainWindow, QMessageBox, QPushButton, QScrollArea
    from PySide2.QtSvg     import QSvgGenerator
    from PySide2.QtPrintSupport import QPrinter
elif PYTHONQT_VERSION == 'PyQt5':
    from PyQt5.QtCore    import Qt, QPointF, QRect, QRectF, QSize, QSizeF, QTimer
    from PyQt5.QtGui     import QBrush, QColor, QFontMetricsF, QImage, QPainter, \
                                QPalette, QPen, QPicture, QPixmap, QPolygonF, \
                                QTextDocument
    from PyQt5.QtWidgets import QAction, QApplication, QDialog, QFileDialog, QLabel, \
                                QMainWindow, QMessageBox, QPushButton, QScrollArea
    from PyQt5.QtSvg     import QSvgGenerator
    from PyQt5.QtPrintSupport import QPrinter
else:
    from PyQt4.QtCore import Qt, QPointF, QRect, QRectF, QSize, QSizeF, QTimer, QString
    from PyQt4.QtGui  import QAction, QApplication, QBrush, QColor, QDialog, \
                             QFileDialog, QFontMetricsF, QImage, QLabel, \
                             QMainWindow, QMessageBox, QPainter, QPalette, \
                             QPen, QPicture, QPixmap, QPolygonF, QPrinter, \
                             QPushButton, QScrollArea, QTextDocument
    from PyQt4.QtSvg  import QSvgGenerator

from multiprocessing import Pipe, Process

from pipedviewer import WINDOW_CLOSED_MESSAGE
from pipedviewer.cmndhelperpq import CmndHelperPQ
from pipedviewer.scaledialogpq import ScaleDialogPQ


class PipedViewerPQ(QMainWindow):
    '''
    A PyQt graphics viewer that receives generic drawing commands
    through a pipe.  Uses a list of QPictures to record the drawings
    which are then used to display, manipulate, and save the image.

    A drawing command is a dictionary with string keys that will be
    interpreted into the appropriate PyQt command(s).  For example,
        { "action":"drawText",
          "text":"Hello",
          "font":{"family":"Times", "size":100, "italic":True},
          "fill":{"color":0x880000, "style":"cross"},
          "outline":{"color":"black"},
          "location":(250,350) }

    The command { "action":"exit" } will shutdown the viewer.
    '''

    def __init__(self, cmndpipe, rspdpipe):
        '''
        Create a PyQt viewer which reads commands from the Pipe
        cmndpipe and writes responses back to rspdpipe.
        '''
        super(PipedViewerPQ, self).__init__()
        self.__cmndpipe = cmndpipe
        self.__rspdpipe = rspdpipe
        # ignore Ctrl-C
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        # default scene size
        self.__scenewidth = int(10.8 * self.physicalDpiX())
        self.__sceneheight = int(8.8 * self.physicalDpiY())
        # scaling factor for line widths and symbol sizes
        self.__widthfactor = None
        self.setWidthScalingFactor(0.72)
        # by default pay attention to any alpha channel values in colors
        self.__noalpha = False
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
        # scaling factor for creating the displayed scene
        self.__scalefactor = 1.0
        # automatically adjust the scaling factor to fit the window frame?
        self.__autoscale = True
        # values used to decide if the scene needs to be updated
        self.__lastpicdrawn = 0
        self.__createpixmap = True
        self.__clearpixmap = True
        # Calculations of modified rectangular regions in QPictures
        # currently do not account for width and height of QPictures
        # played inside them.  So keep a expansion value.
        self.__maxsymbolwidth = 0.0
        self.__maxsymbolheight = 0.0
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
        # default file name and format for saving the image
        self.__lastfilename = "ferret.png"
        self.__lastformat = "png"
        self.__addedannomargin = 12
        # command helper object
        self.__helper = CmndHelperPQ(self)
        # Create the menubar
        self.__scaleact = QAction(self.tr("&Scale"), self,
                                shortcut=self.tr("Ctrl+S"),
                                statusTip=self.tr("Scale the image (canvas and image change size)"),
                                triggered=self.inquireSceneScale)
        self.__saveact = QAction(self.tr("Save &As..."), self,
                                shortcut=self.tr("Ctrl+A"),
                                statusTip=self.tr("Save the image to file"),
                                triggered=self.inquireSaveFilename)
        self.__redrawact = QAction(self.tr("&Redraw"), self,
                                shortcut=self.tr("Ctrl+R"),
                                statusTip=self.tr("Clear and redraw the image"),
                                triggered=self.redrawScene)
        self.__aboutact = QAction(self.tr("&About"), self,
                                statusTip=self.tr("Show information about this viewer"),
                                triggered=self.aboutMsg)
        self.__aboutqtact = QAction(self.tr("About &Qt"), self,
                                statusTip=self.tr("Show information about the Qt library"),
                                triggered=self.aboutQtMsg)
        self.createMenus()
        # Set the initial size of the viewer
        self.__framedelta = 4
        mwwidth = self.__scenewidth + self.__framedelta
        mwheight = self.__sceneheight + self.__framedelta \
                 + self.menuBar().height() \
                 + self.statusBar().height()
        self.resize(mwwidth, mwheight)
        # check the command queue any time there are no window events to deal with
        self.__timer = QTimer(self)
        self.__timer.timeout.connect(self.checkCommandPipe)
        self.__timer.setInterval(0)
        self.__timer.start()
        # initialize the parameters for watermark image display
        self.__wmarkFilename = None
        self.__xloc = None
        self.__yloc = None
        self.__scalefrac = None
        self.__opacity = None

    def createMenus(self):
        '''
        Create the menu items for the viewer
        using the previously created actions.
        '''
        menuBar = self.menuBar()
        sceneMenu = menuBar.addMenu(menuBar.tr("&Image"))
        sceneMenu.addAction(self.__scaleact)
        sceneMenu.addAction(self.__saveact)
        sceneMenu.addAction(self.__redrawact)
        helpMenu = menuBar.addMenu(menuBar.tr("&Help"))
        helpMenu.addAction(self.__aboutact)
        helpMenu.addAction(self.__aboutqtact)

    def showEvent(self, event):
        '''
        When the viewer is going to be shown, make sure all
        the current pictures are displayed in the scene.
        '''
        # update, ignoring the visibility flags
        self.drawLastPictures(True)
        event.accept()

    def resizeEvent(self, event):
        '''
        Monitor resizing in case auto-scaling of the image is selected.
        '''
        if self.__autoscale:
            if self.autoScaleScene():
                # continue with the window resize
                event.accept()
            else:
                # another resize coming in, so ignore this one
                event.ignore()
        else:
            # continue with the window resize
            event.accept()

    def closeEvent(self, event):
        '''
        Clean up and send the WINDOW_CLOSED_MESSAGE on the response pipe
        before closing the window.
        '''
        self.__timer.stop()
        self.__cmndpipe.close()
        try:
            try:
                self.__rspdpipe.send(WINDOW_CLOSED_MESSAGE)
            finally:
                self.__rspdpipe.close()
        except Exception:
            pass
        event.accept()

    def exitViewer(self):
        '''
        Close and exit the viewer.
        '''
        self.close()

    def aboutMsg(self):
        QMessageBox.about(self, self.tr("About PipedViewerPQ"),
            self.tr("\n" \
            "PipedViewerPQ is a graphics viewer application that receives its " \
            "drawing and other commands primarily from another application " \
            "through a pipe.  A limited number of commands are provided by " \
            "the viewer itself to allow saving and some manipulation of the " \
            "displayed image.  The controlling application, however, may be " \
            "unaware of these modifications made to the image. " \
            "\n\n" \
            "PipedViewerPQ was developed by the Thermal Modeling and Analysis " \
            "Project (TMAP) of the National Oceanographic and Atmospheric " \
            "Administration's (NOAA) Pacific Marine Environmental Lab (PMEL). "))

    def aboutQtMsg(self):
        QMessageBox.aboutQt(self, self.tr("About Qt"))

    def ignoreAlpha(self):
        '''
        Return whether the alpha channel in colors should always be ignored.
        '''
        return self.__noalpha

    def paintScene(self, painter, first, leftx, uppery, scalefactor,
                   statusmsg, returnregion):
        '''
        Draws the pictures self.__viewpics[first:] using the QPainter
        painter.  This QPainter should have been initialized
        appropriately for the QPaintDevice to which it is painting
        (e.g., QImage.fill with the desired background color).

        The point (leftx, uppery) is the offset of the origin after
        scaling using scalefactor.  (All are floating point values.)

        The status bar will be updated with a message derived from
        statusmsg before drawing each picture.  Upon completion, the
        status bar will be cleared.

        If returnregion is True, a list of QRect objects describing
        the modified regions will be computed and returned.  If
        returnregion is False, the modified region will not be computed
        and an empty list will be returned.

        The call to painter.end() will need to be made after calling
        this function.
        '''
        # change the cursor to warn the user this may take some time
        QApplication.setOverrideCursor(Qt.WaitCursor)
        # create the incomplete status message
        if (first + 1) < len(self.__viewpics):
            mymsg = "%s (piece %%s of %s)" % (statusmsg, str(len(self.__viewpics)))
        else:
            mymsg = "%s (piece %%s)" % statusmsg
        # get the origin for drawing the pictures after scaling
        myorigin = QPointF(leftx, uppery)
        # set the scaling factor for the pictures
        painter.scale(scalefactor, scalefactor)
        modrects = [ ]
        # draw the appropriate pictures
        k = first
        for (viewpic, _) in self.__viewpics[first:]:
            k += 1
            # show the progress message
            self.statusBar().showMessage( self.tr(mymsg % str(k)) )
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
        # done - clear the status message
        self.statusBar().clearMessage()
        # restore the cursor back to normal
        QApplication.restoreOverrideCursor()
        # if watermark is specified, display after other plotting occurs
        if self.__wmarkFilename is not None:
            # Initialize watermark objects
            wmkpic = QPixmap(self.__wmarkFilename)
            wmkpt = QPointF()
            wmkpt.setX(self.__xloc)
            wmkpt.setY(self.__yloc)
            # set watermark image display opacity
            painter.setOpacity(self.__opacity / k)
            # set image scale
            painter.scale(self.__scalefrac, self.__scalefrac)
            # paint watermark image at specified location
            painter.setRenderHint(QPainter.Antialiasing)
            painter.drawPixmap(wmkpt, wmkpic)
        return modrects

    def drawLastPictures(self, ignorevis):
        '''
        Update the scene with pictures yet to be drawn.
        If ignorevis is True, the update will be done
        even if the viewer is not visible; otherwise
        drawing to the scene label is only done if the
        viewer is visible.
        '''
        if not ignorevis:
            if self.isMinimized() or not self.isVisible():
                # Not shown, so do not waste time drawing
                return
        if self.__createpixmap:
            # Create and assign a cleared pixmap
            mypixmap = QPixmap(self.__label.size())
            mypixmap.fill(self.__lastclearcolor)
            self.__label.setPixmap(mypixmap)
            self.__createpixmap = False
            self.__clearpixmap = False
            wascleared = True
        elif self.__clearpixmap:
            # Clear the existing pixmap
            self.__label.pixmap().fill(self.__lastclearcolor)
            self.__clearpixmap = False
            wascleared = True
        elif len(self.__viewpics) > self.__lastpicdrawn:
            # New pictures to add to an existing scene
            # wascleared = False
            # Rectangles of modified regins incorrect for drawText
            # so always update the entire scene
            wascleared = True
        else:
            # Nothing changed so just return
            return
        # Only create the QPainter if there are pictures
        # to draw (this is more than just a clear)
        if len(self.__viewpics) > self.__lastpicdrawn:
            painter = QPainter(self.__label.pixmap())
            modrects = self.paintScene(painter, self.__lastpicdrawn, \
                                       0.0, 0.0, self.__scalefactor, \
                                       "Drawing", not wascleared)
            painter.end()
        # Notify the label of changes to the scene
        if wascleared:
            # the entire scene changed
            self.__label.update()
        else:
            # the scene changed only in the modrects areas
            for rect in modrects:
                self.__label.update(rect)
        # Update the record of which pictures have been displayed
        self.__lastpicdrawn = len(self.__viewpics)

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
            self.endView(False)
            restartview = True
        else:
            restartview = False
        # get the color to use for clearing (the background color)
        if bkgcolor:
            if bkgcolor.isValid():
                self.__lastclearcolor = bkgcolor
        # Delete all the pictures from the list and
        # mark that the pixmap needs to be cleared
        self.__viewpics[:] = [ ]
        self.__maxsymbolwidth = 0.0
        self.__maxsymbolheight = 0.0
        self.__clearpixmap = True
        self.__lastpicdrawn = 0
        # Update the scene label if visible
        self.drawLastPictures(False)
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
            self.endView(False)
            hadactiveview = True
        else:
            hadactiveview = False
        if bkgcolor:
            if bkgcolor.isValid():
                self.__lastclearcolor = bkgcolor
        # mark that the pixmap needs to be cleared
        # and all the pictures redrawn
        self.__clearpixmap = True
        self.__lastpicdrawn = 0
        # Update the scene label if visible
        self.drawLastPictures(False)
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
            # Resize the label and set label values
            # so the scrollarea knows of the new size
            labelwidth = int(newwidth * self.__scalefactor + 0.5)
            labelheight = int(newheight * self.__scalefactor + 0.5)
            self.__label.setMinimumSize(labelwidth, labelheight)
            self.__label.resize(labelwidth, labelheight)
            # mark that the pixmap needs to be recreated
            self.__scenewidth = newwidth
            self.__sceneheight = newheight
            self.__createpixmap = True
            # If auto-scaling, set scaling factor to 1.0 and resize the window
            if self.__autoscale:
                self.__scalefactor = 1.0
                barheights = self.menuBar().height() + self.statusBar().height()
                self.resize(newwidth + self.__framedelta,
                            newheight + self.__framedelta + barheights)
                # the resize should redraw the scene
            else:
                # Redraw the scene from the beginning using the scaling factor
                self.redrawScene()


    def inquireSceneScale(self):
        '''
        Prompt the user for the desired scaling factor for the scene.
        '''
        labelwidth = int(self.__scenewidth * self.__scalefactor + 0.5)
        labelheight = int(self.__sceneheight * self.__scalefactor + 0.5)
        scaledlg = ScaleDialogPQ(self.__scalefactor, labelwidth, labelheight,
                       self.__minsize, self.__minsize, self.__autoscale, self)
        if scaledlg.exec_():
            (newscale, autoscale, okay) = scaledlg.getValues()
            if okay:
                if autoscale:
                    self.__autoscale = True
                    self.autoScaleScene()
                else:
                    self.__autoscale = False
                    self.scaleScene(newscale, False)

    def autoScaleScene(self):
        '''
        Selects a scaling factor that maximizes the scene within the window
        frame without requiring scroll bars.  Intended to be called when
        the window size is changed by the user and auto-scaling is turn on.

        Returns:
            True if scaling of this scene is done (no window resize)
            False if the a new resize command was issued
        '''
        barheights = self.menuBar().height() + self.statusBar().height()

        # get the size for the central widget
        cwheight = self.height() - barheights - self.__framedelta
        heightsf = float(cwheight) / float(self.__sceneheight)

        cwwidth = self.width() - self.__framedelta
        widthsf = float(cwwidth) / float(self.__scenewidth)

        if heightsf < widthsf:
            factor = heightsf
        else:
            factor = widthsf

        newcwheight = int(factor * self.__sceneheight + 0.5)
        newcwwidth = int(factor * self.__scenewidth + 0.5)

        # if the window does not have the correct aspect ratio, resize it so
        # it will; this will generate another call to this method.  Otherwise,
        # scale the scene and be done.
        if self.isMaximized() or \
           ( (abs(cwheight - newcwheight) <= self.__framedelta) and \
             (abs(cwwidth - newcwwidth) <= self.__framedelta) ):
            self.scaleScene(factor, False)
            return True
        else:
            self.resize(newcwwidth + self.__framedelta,
                        newcwheight + self.__framedelta + barheights)
            return False

    def scaleScene(self, factor, resizewin):
        '''
        Scales both the horizontal and vertical directions by factor.
        Scaling factors are not accumulative.  So if the scene was
        already scaled, that scaling is "removed" before this scaling
        factor is applied.  If resizewin is True, the main window is
        resized to accommodate this new scaled scene size.

        If factor is zero, just switch to auto-scaling at the current
        window size.  If factor is negative, rescale using the absolute
        value (possibly resizing the window) then switch to auto-scaling.
        '''
        fltfactor = float(factor)
        if fltfactor != 0.0:
            if resizewin:
                # from command - turn off autoscaling for the following
                # then turn back on if appropriate
                self.__autoscale = False
            newfactor = abs(fltfactor)
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
                # mark that the pixmap needs to be recreated
                self.__createpixmap = True
                # Redraw the scene from the beginning
                self.redrawScene()
            if resizewin:
                # resize the main window (if possible)
                barheights = self.menuBar().height() + self.statusBar().height()
                mwheight = newlabheight + barheights + self.__framedelta
                mwwidth = newlabwidth + self.__framedelta
                # Do not exceed the available real estate on the screen.
                # If autoscaling is in effect, the resize will trigger
                # any required adjustments.
                scrnrect = QApplication.desktop().availableGeometry()
                if mwwidth > 0.95 * scrnrect.width():
                    mwwidth = int(0.9 * scrnrect.width())
                if mwheight > 0.95 * scrnrect.height():
                    mwheight = int(0.9 * scrnrect.height())
                self.resize(mwwidth, mwheight)
        if fltfactor <= 0.0:
            # From command - turn on autoscaling
            self.__autoscale = True
            self.autoScaleScene();

    def inquireSaveFilename(self):
        '''
        Prompt the user for the name of the file into which to save the scene.
        The file format will be determined from the filename extension.
        '''
        formattypes = [ ( "png",
                          "PNG - Portable Networks Graphics (*.png)"),
                        ( "jpeg",
                          "JPEG - Joint Photographic Experts Group (*.jpeg *.jpg *.jpe)" ),
                        ( "tiff",
                          "TIFF - Tagged Image File Format (*.tiff *.tif)" ),
                        ( "pdf",
                          "PDF - Portable Document Format (*.pdf)" ),
                        ( "ps",
                          "PS - PostScript (*.ps)" ),
                        ( "svg",
                          "SVG - Scalable Vector Graphics (*.svg)" ),
                        ( "bmp",
                          "BMP - Windows Bitmap (*.bmp)" ),
                        ( "ppm",
                          "PPM - Portable Pixmap (*.ppm)" ),
                        ( "xpm",
                          "XPM - X11 Pixmap (*.xpm)" ),
                        ( "xbm",
                          "XBM - X11 Bitmap (*.xbm)" ), ]
        filters = ";;".join( [ t[1] for t in formattypes ] )
        if PYTHONQT_VERSION == 'PyQt4':
            # tr returns QStrings in PyQt4 (Python2)
            (fileName, fileFilter) = QFileDialog.getSaveFileNameAndFilter(self,
                 self.tr("Save the current image as "), self.tr(self.__lastfilename), self.tr(filters))
        else:
            # tr returns Python unicode strings in PySide2 or PyQt5 (Python3)
            (fileName, fileFilter) = QFileDialog.getSaveFileName(self,
                self.tr("Save the current image as "), self.tr(self.__lastfilename), self.tr(filters))
        if fileName:
            for (fmt, fmtQName) in formattypes:
                if self.tr(fmtQName) == fileFilter:
                    fileFormat = fmt
                    break
            else:
                raise RuntimeError("Unexpected file format name '%s'" % fileFilter)
            self.saveSceneToFile(fileName, fileFormat, None, None, None, None)
            self.__lastfilename = fileName
            self.__lastformat = fileFormat

    def saveSceneToFile(self, filename, imageformat, transparent,
                        vectsize, rastsize, myannotations):
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

        If myannotations is not None, the strings given in the tuple
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
            elif fileext == '.eps':
                # needs a PS QPrinter and never rotate
                myformat = 'eps'
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

        if myannotations:
            annopicture = QPicture()
            annopainter = QPainter(annopicture)
            annotextdoc = QTextDocument()
            # Leave room for the added margins to the width
            annotextdoc.setTextWidth(self.__scenewidth - 2.0 * self.__addedannomargin)
            annotextdoc.setHtml("<p>" + "<br />".join(myannotations) + "</p>")
            annotextdoc.drawContents(annopainter)
            annopainter.end()
            annosize = annotextdoc.documentLayout().documentSize()
        else:
            annopicture = None
            annosize = None

        if (myformat == 'ps') or (myformat == 'eps') or (myformat == 'pdf'):
            # Setup the QPrinter that will be used to create the EPS, PS, or PDF file
            printer = QPrinter(QPrinter.HighResolution)
            printer.setOutputFileName(myfilename)
            # The print format is automatically set from the
            # filename extension; so the following is actually
            # only needed for absent or strange extensions
            if (myformat == 'ps') or (myformat == 'eps'):
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
                             / float(self.physicalDpiX())
                imageheight = self.__sceneheight * self.__scalefactor \
                              / float(self.physicalDpiY())
            # Add in any height needed for the annotations
            if annopicture:
                annoheight = (annosize.height() + 2 * self.__addedannomargin) * \
                             imageheight / self.__sceneheight
                imageheight += annoheight
            # Set the image size
            try:
                # Set custom paper size to just fit around the image
                if (myformat != 'eps') and (imagewidth > imageheight):
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
            if (myformat != 'eps') and (imagewidth > imageheight):
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
            widthscalefactor = imagewidth * self.physicalDpiX() / float(self.__scenewidth)
            # Check if there are annotations to add
            if annopicture:
                # Scale the scene now for the annotations
                painter.scale(widthscalefactor, widthscalefactor)
                # factor that makes it work after scaling (12.5 = 1200 / 96)
                printfactor = printres / self.physicalDpiX()
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
                        1.0, "Saving", False)
            else:
                # No annotations so just do the normal drawing
                self.paintScene(painter, 0, 0.0, 0.0,
                                widthscalefactor, "Saving", False)
            painter.end()
        elif myformat == 'svg':
            generator = QSvgGenerator()
            generator.setFileName(myfilename)
            if vectsize:
                imagewidth = int(vectsize.width() * self.physicalDpiX() + 0.5)
                imageheight = int(vectsize.height() * self.physicalDpiY() + 0.5)
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
                    int(0.5 * (self.physicalDpiX() + self.physicalDpiY()) + 0.5) )
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
                                1.0, "Saving", False)
            else:
                # No annotations so just do the normal drawing
                self.paintScene(painter, 0, 0.0, 0.0,
                                widthscalefactor, "Saving", False)
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
            # Indicate the recommended displayed size of PNG images
            image.setDotsPerMeterX(self.physicalDpiX() / 0.0254)
            image.setDotsPerMeterY(self.physicalDpiY() / 0.0254)
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
                                1.0, "Saving", False)
            else:
                # No annotations so just do the normal drawing
                self.paintScene(painter, 0, 0.0, 0.0,
                                widthscalefactor, "Saving", False)
            painter.end()

            # save the image to file
            if not image.save(myfilename, myformat):
                raise ValueError("Unable to save the plot as " + myfilename)

    def checkCommandPipe(self):
        '''
        Get and perform commands waiting in the pipe.
        Stop when no more commands or if more than 50
        milliseconds have passed.
        '''
        try:
            if (sys.version_info[0] >= 3) and (sys.version_info[1] >= 3):
                starttime = time.process_time()
            else:
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
                if (sys.version_info[0] >= 3) and (sys.version_info[1] >= 3):
                    elapsed = time.process_time() - starttime
                else:
                    elapsed = time.clock() - starttime
                if elapsed > 0.050:
                    break
        except EOFError:
            # Assume PyFerret has shut down
            self.exitViewer()
        except Exception:
            # Some problem, but presumably still functional
            (exctype, excval) = sys.exc_info()[:2]
            try:
                if excval:
                    self.__rspdpipe.send("**ERROR %s: %s" % (str(exctype), str(excval)))
                else:
                    self.__rspdpipe.send("**ERROR %s" % str(exctype))
            except Exception:
                pass

    def processCommand(self, cmnd):
        '''
        Examine the action of cmnd and call the appropriate
        method to deal with this command.  Raises a ValueError
        if the "action" key is missing.
        '''
        try:
            cmndact = cmnd["action"]
        except KeyError:
            raise ValueError("Unknown command '%s'" % str(cmnd))

        if cmndact == "clear":
            try:
                bkgcolor = self.__helper.getColorFromCmnd(cmnd)
            except KeyError:
                bkgcolor = None
            self.clearScene(bkgcolor)
        elif cmndact == "exit":
            self.exitViewer()
        elif cmndact == "hide":
            self.showMinimized()
        elif cmndact == "screenInfo":
            scrnrect = QApplication.desktop().availableGeometry()
            info = ( self.physicalDpiX(), self.physicalDpiY(),
                     scrnrect.width(), scrnrect.height() )
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
            self.scaleScene(newscale, True)
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
                myannotations = cmnd["annotations"]
            except KeyError:
                myannotations = None
            self.saveSceneToFile(filename, fileformat, transparent,
                                 vectsize, rastsize, myannotations)
        elif cmndact == "setWidthFactor":
            newfactor = float(cmnd.get("factor", -1.0))
            if newfactor <= 0.0:
                raise ValueError("Invalid width factor")
            self.setWidthScalingFactor(newfactor)
        elif cmndact == "setTitle":
            self.setWindowTitle(cmnd["title"])
        elif cmndact == "imgname":
            value = cmnd.get("name", None)
            if value:
                self.__lastfilename = value
            value = cmnd.get("format", None)
            if value:
                self.__lastformat = value.lower()
        elif cmndact == "show":
            if not self.isVisible():
                self.show()
        elif cmndact == "noalpha":
            self.__noalpha = True
        elif cmndact == "beginView":
            self.beginView(cmnd)
        elif cmndact == "clipView":
            self.clipView(cmnd)
        elif cmndact == "endView":
            self.endView(True)
        elif cmndact == "beginSegment":
            self.beginSegment(cmnd["segid"])
        elif cmndact == "endSegment":
            self.endSegment(True)
        elif cmndact == "deleteSegment":
            self.deleteSegment(cmnd["segid"])
        elif cmndact == "createSymbol":
            # Define this symbol in self.__symbolpaths
            sympath = self.__helper.getSymbolFromCmnd(cmnd)
            # The name is now all that is needed to use this symbol
            self.__rspdpipe.send(cmnd['name'])
        elif cmndact == "drawMultiline":
            self.drawMultiline(cmnd)
        elif cmndact == "drawPoints":
            self.drawPoints(cmnd)
        elif cmndact == "drawPolygon":
            self.drawPolygon(cmnd)
        elif cmndact == "drawRectangle":
            self.drawRectangle(cmnd)
        elif cmndact == "textSize":
            info = self.getSimpleTextSize(cmnd)
            self.__rspdpipe.send(info)
        elif cmndact == "drawText":
            self.drawSimpleText(cmnd)
        elif cmndact == "setWaterMark":
            # self.setWaterMark(cmnd)
            self.setWaterMark(cmnd['filename'], None, cmnd['xloc'], cmnd['yloc'], cmnd['scalefrac'], cmnd['opacity'])
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
            self.endView(True)
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
            raise ValueError("Invalid left, right view fractions: " \
                             "left in pixels = %s, right in pixels = %s" \
                             % (str(leftpixel), str(rightpixel)) )
        if (0.0 > toppixel) or (toppixel >= bottompixel) or (bottompixel > height):
            raise ValueError("Invalid bottom, top view fractions: " \
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

    def endView(self, update):
        '''
        Ends the current view and appends it to the list of pictures
        drawn in the scene.  If update is True, the displayed scene
        is updated.
        '''
        self.__activepainter.end()
        self.__activepainter = None
        # Only save the active picture if it contains something
        if self.__drawcount > 0:
            self.__viewpics.append((self.__activepicture, self.__segid))
            self.__drawcount = 0
        if update:
            # update the scene
            self.drawLastPictures(False)
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
            self.endView(True)
            self.beginViewFromSides(self.__fracsides, self.__clipit)
        self.__segid = segid

    def endSegment(self, update):
        '''
        Ends the current active view and starts a new view.
        Removes the current segment ID associated with views.
        '''
        if self.__activepainter and (self.__drawcount > 0):
            self.endView(update)
            self.beginViewFromSides(self.__fracsides, self.__clipit)
        if update:
            self.drawLastPictures(False)
        self.__segid = None

    def deleteSegment(self, segid):
        '''
        Removes all pictures associated with the given segment ID
        '''
        # if deleting the current segment, end the current segment
        if segid == self.__segid:
            self.endSegment(False)
        # Go through all the pictures, determining which to save
        newpicts = [ ]
        for (viewpic, vsegid) in self.__viewpics:
            if vsegid != segid:
                newpicts.append((viewpic, vsegid))
            else:
                # picture was deleted, so will need to
                # regenerate the scene from the beginning
                self.__clearpixmap = True
                self.__lastpicdrawn = 0
        self.__viewpics[:] = newpicts
        # Do NOT update since there may be more segments to be deleted
        # Rely on the receiving an update or redraw command at the end

    def updateScene(self):
        '''
        Updates the displayed graphics to include all drawn elements.
        '''
        # If there is an active picture containing something,
        # end the view, thus adding and display this picture,
        # then restart the view.
        if self.__drawcount > 0:
            self.endView(True)
            self.beginViewFromSides(self.__fracsides, self.__clipit)
        self.drawLastPictures(False)

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
            "symbol": symbol to use (see CmndHelperPQ.getSymbolFromCmnd)
            "size": size of the symbol in points (1/72 inches) before
                    scaling by the width scaling factor
            "color": color name or 24-bit RGB integer value (eg, 0xFF0088)
            "alpha": alpha value from 0 (transparent) to 255 (opaque)
            "highlight": dictionary of "color" and "alpha" (as above)
                     for filled symbol outline color; not outlined if omitted

        The coordinates are device coordinates from the upper left corner.

        Raises a KeyError if the "symbol", "points", or "size" key
        is not given.
        '''
        ptcoords = cmnd["points"]
        ptsize = cmnd["size"]
        try:
            highlight = self.__helper.getColorFromCmnd(cmnd["highlight"])
        except KeyError:
            highlight = None
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
            if highlight:
                # highlighted filled plot - pen width is 4% of the width of the symbol
                mybrush = QBrush(highlight, Qt.SolidPattern)
                mypen = QPen(mybrush, 4.0, Qt.SolidLine, Qt.FlatCap, Qt.MiterJoin)
                self.__activepainter.setPen(mypen)
            else:
                # filled plot without highlight - no pen, only brush
                self.__activepainter.setPen(Qt.NoPen)
        else:
            # stroked path - no brush, pen width is 8% of the width of the symbol, highlight is ignored
            self.__activepainter.setBrush(Qt.NoBrush)
            mypen = QPen(mybrush, 8.0, Qt.SolidLine, Qt.FlatCap, Qt.MiterJoin)
            self.__activepainter.setPen(mypen)
        # typical symbols are 100x100 pixels
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
            # mypen = QPen(mybrush, 0.0, Qt.SolidLine, Qt.SquareCap, Qt.BevelJoin)
            mypen = Qt.NoPen
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
            raise ValueError("width of the rectangle (%s) in not positive" % str(width))
        height = sides.bottom() - sides.top()
        if height <= 0.0:
            raise ValueError("height of the rectangle (%s) in not positive" % str(height))
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
            # mypen = QPen(mybrush, 1.0, Qt.SolidLine, Qt.SquareCap, Qt.BevelJoin)
            mypen = Qt.NoPen
        self.__activepainter.setBrush(mybrush)
        self.__activepainter.setPen(mypen)
        self.__activepainter.drawRect(myrect)
        self.__drawcount += 1
        # Limit the number of drawing commands per picture
        if self.__drawcount >= self.__maxdraws:
            self.updateScene()

    def getSimpleTextSize(self, cmnd):
        '''
        Returns the pair (width, height) for given text when drawn.
        Raises a KeyError if the "text" key is not given.

        The width value is the width for the text that can be used
        for positioning the next text item to draw.  The height
        value is the ascent plus decent for the font and does not
        depend of the text.  The bounding rectangle for the actual
        drawn text may exceed this (width, height) if,
        e.g., italic or unusual characters.

        Recognized keys from cmnd:
            "text": string to displayed
            "font": dictionary describing the font to use;  see
                    CmndHelperPQ.getFontFromCmnd.  If not given
                    the default font for this viewer is used.
        '''
        try:
            myfont = self.__helper.getFontFromCmnd(cmnd["font"])
        except KeyError:
            myfont = self.__activepainter.font()
        myfontmetrics = QFontMetricsF(myfont)
        mytext = cmnd["text"]
        if PYTHONQT_VERSION == 'PyQt4':
            mytext = QString.fromUtf8(mytext)
        width = myfontmetrics.width(mytext)
        height = myfontmetrics.height()
        return (width, height)

    def drawSimpleText(self, cmnd):
        '''
        Draws a "simple" text item in the current view.
        Raises a KeyError if the "text" or "location" key is not given.

        Recognized keys from cmnd:
            "text": null-terminated UTF-8 encoded string to be displayed
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

            if PYTHONQT_VERSION == 'PyQt4':
                mytext = QString.fromUtf8(mytext)
            self.__activepainter.drawText(0, 0, mytext)
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
        self.__widthfactor  = (self.physicalDpiX() + self.physicalDpiY()) / 144.0
        self.__widthfactor *= factor

    def widthScalingFactor(self):
        '''
        Return the scaling factor for line widths and symbol sizes
        to convert from points (1/72 inches) to pixels, and to apply
        any additional width scaling specified by setWidthFactor.
        '''
        return self.__widthfactor

    def setWaterMark(self, filename, len_filename, xloc, yloc, scalefrac, opacity):
        '''
        Overlay watermark from contents of filename.

        Recognized keys from cmnd:
            "filename":  water mark image file
            "xloc":      horizontal position of upper left corner of watermark image
            "yloc":      vertical position of upper left corner of watermark image
            "scalefrac": multiple of original image size to display plot as
            "opacity":   image visibility in range [0.0, 1.0] where 0->invisible, 1->opaque
        '''
        self.__wmarkFilename = str(filename)
        self.__xloc = xloc
        self.__yloc = yloc
        self.__scalefrac = scalefrac
        self.__opacity = opacity

class PipedViewerPQProcess(Process):
    '''
    A Process specifically tailored for creating a PipedViewerPQ.
    '''
    def __init__(self, cmndpipe, rspdpipe):
        '''
        Create a Process that will produce a PipedViewerPQ
        attached to the given Pipes when run.
        '''
        super(PipedViewerPQProcess,self).__init__(group=None, target=None, name='PipedViewerPQ')
        self.__cmndpipe = cmndpipe
        self.__rspdpipe = rspdpipe
        self.__app = None
        self.__viewer = None

    def run(self):
        '''
        Create a PipedViewerPQ that is attached
        to the Pipe of this instance.
        '''
        self.__app = QApplication(["PipedViewerPQ"])
        self.__viewer = PipedViewerPQ(self.__cmndpipe, self.__rspdpipe)
        myresult = self.__app.exec_()
        sys.exit(myresult)


#
# The following are for testing this (and the cmndhelperpq) modules
#

class _CommandSubmitterPQ(QDialog):
    '''
    Testing dialog for controlling the addition of commands to a pipe.
    Used for testing PipedViewerPQ in the same process as the viewer.
    '''
    def __init__(self, parent, cmndpipe, rspdpipe, cmndlist):
        '''
        Create a QDialog with a single QPushButton for controlling
        the submission of commands from cmndlist to cmndpipe.
        '''
        super(_CommandSubmitterPQ,self).__init__(parent)
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
            print("Command: %s" % str(self.__cmndlist[self.__nextcmnd]))
            self.__cmndpipe.send(self.__cmndlist[self.__nextcmnd])
            self.__nextcmnd += 1
            while self.__rspdpipe.poll(0.1):
                print("Response: %s" % str(self.__rspdpipe.recv()))
        except IndexError:
            self.__rspdpipe.close()
            self.__cmndpipe.close()
            self.close()


def _test_pipedviewerpq():
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
    drawcmnds.append( { "action":"beginSegment",
                        "segid":"text" } )
    drawcmnds.append( { "action":"drawText",
                        "text":"y=480",
                        "font":{"family":"Times", "size":16},
                        "fill":{"color":"red"},
                        "location":(50,480) } )
    drawcmnds.append( { "action":"drawText",
                        "text":"y=430",
                        "font":{"family":"Times", "size":16},
                        "fill":{"color":"red"},
                        "location":(50,430) } )
    drawcmnds.append( { "action":"drawText",
                        "text":"y=380",
                        "font":{"family":"Times", "size":16},
                        "fill":{"color":"red"},
                        "location":(50,380) } )
    drawcmnds.append( { "action":"drawText",
                        "text":"y=330",
                        "font":{"family":"Times", "size":16},
                        "fill":{"color":"red"},
                        "location":(50,330) } )
    drawcmnds.append( { "action":"textSize",
                        "text":"This is a some line of text",
                        "font":{"family":"Times", "size":16} } )
    drawcmnds.append( { "action":"endSegment" } )
    drawcmnds.append( { "action":"endView" } )
    drawcmnds.append( { "action":"show" } )
    drawcmnds.append( { "action":"createSymbol",
                        "name": "uptrifill",
                        "pts": ( (-40.0, -30.0), (0.0, 40.0), (40.0, -30.0), (-40.0, -30.0), ),
                        "fill": True } )
    drawcmnds.append( { "action":"createSymbol",
                        "name": "bararrow",
                        "pts": ( (-50,50), (-10,10),
                                 (-999, -999),
                                 (50,0), (50,50), (0,50),
                                 (-999, -999),
                                 (0,-10), (20,-30), (10,-30), (10,-50), (-10,-50), (-10,-30), (-20,-30), (0,-10), ),
                        "fill": False } )
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
                        "symbol":"dot",
                        "size":20,
                        "color":"magenta" })
    drawcmnds.append( { "action":"drawPoints",
                        "points":( (150,  50),
                                   (150, 150),
                                   (150, 250),
                                   (150, 350),
                                   (150, 450) ),
                        "symbol":"circle",
                        "size":20,
                        "color":"magenta" })
    drawcmnds.append( { "action":"drawPoints",
                        "points":( (200,  50),
                                   (200, 150),
                                   (200, 250),
                                   (200, 350),
                                   (200, 450) ),
                        "symbol":"dotplus",
                        "size":20,
                        "color":"magenta" })
    drawcmnds.append( { "action":"drawPoints",
                        "points":( (250,  50),
                                   (250, 150),
                                   (250, 250),
                                   (250, 350),
                                   (250, 450) ),
                        "symbol":"circplus",
                        "size":20,
                        "color":"magenta" })
    drawcmnds.append( { "action":"drawPoints",
                        "points":( (300,  50),
                                   (300, 150),
                                   (300, 250),
                                   (300, 350),
                                   (300, 450) ),
                        "symbol":"dotex",
                        "size":20,
                        "color":"magenta" })
    drawcmnds.append( { "action":"drawPoints",
                        "points":( (350,  50),
                                   (350, 150),
                                   (350, 250),
                                   (350, 350),
                                   (350, 450) ),
                        "symbol":"circex",
                        "size":20,
                        "color":"magenta" })
    drawcmnds.append( { "action":"drawPoints",
                        "points":( (400,  50),
                                   (400, 150),
                                   (400, 250),
                                   (400, 350),
                                   (400, 450) ),
                        "symbol":"uptrifill",
                        "size":20,
                        "color":"magenta" })
    drawcmnds.append( { "action":"drawPoints",
                        "points":( (450,  50),
                                   (450, 150),
                                   (450, 250),
                                   (450, 350),
                                   (450, 450) ),
                        "symbol":"bararrow",
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
    drawcmnds.append( { "action":"deleteSegment",
                        "segid":"text" } )
    drawcmnds.append( { "action":"update" } )
    drawcmnds.append( { "action":"show" } )
    annotations = ( "The 1<sup>st</sup> CO<sub>2</sub> annotations line",
                    "Another line with <i>lengthy</i> details that go on and on " + \
                    "and on and should wrap to a 2<sup>nd</sup> annotation line",
                    "<b>Final</b> annotation line" )
    drawcmnds.append( { "action":"save",
                        "filename":"test.pdf",
                        "vectsize":{"width":7.0, "height":7.0},
                        "rastsize":{"width":750, "height":750},
                        "annotations":annotations } )
    drawcmnds.append( { "action":"save",
                        "filename":"test.png",
                        "vectsize":{"width":7.0, "height":7.0},
                        "rastsize":{"width":750, "height":750},
                        "annotations":annotations } )
    drawcmnds.append( { "action":"exit" } )

    # start PyQt
    app = QApplication(["PipedViewerPQ"])
    # create a PipedViewerPQ in this process
    cmndrecvpipe, cmndsendpipe = Pipe(False)
    rspdrecvpipe, rspdsendpipe = Pipe(False)
    viewer = PipedViewerPQ(cmndrecvpipe, rspdsendpipe)
    # create a command submitter dialog
    tester = _CommandSubmitterPQ(viewer, cmndsendpipe,
                                 rspdrecvpipe, drawcmnds)
    tester.show()
    # let it all run
    result = app.exec_()
    if result != 0:
        sys.exit(result)

if __name__ == "__main__":
    _test_pipedviewerpq()
