'''
PipedImagerPQ is a graphics viewer application written in PyQt
that receives its images and commands primarily from another
application through a pipe.  A limited number of commands are
provided by the viewer itself to allow saving and some manipulation
of the displayed image.  The controlling application, however, may
be unaware of these modifications made to the image.

PipedImagerPQProcess is used to create and run a PipedImagerPQ.

This package was developed by the Thermal Modeling and Analysis
Project (TMAP) of the National Oceanographic and Atmospheric
Administration's (NOAA) Pacific Marine Environmental Lab (PMEL).
'''

from __future__ import print_function

import sys
import os
import time
import signal

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
    from PySide2.QtCore    import Qt, QPointF, QRectF, QSize, QTimer
    from PySide2.QtGui     import QBrush, QColor, QImage, QPainter, \
                                  QPalette, QPen, QPixmap, QPolygonF
    from PySide2.QtWidgets import QAction, QApplication, QDialog, \
                                  QFileDialog, QLabel, QMainWindow, \
                                  QMessageBox, QPushButton, QScrollArea
elif PYTHONQT_VERSION == 'PyQt5':
    from PyQt5.QtCore    import Qt, QPointF, QRectF, QSize, QTimer
    from PyQt5.QtGui     import QBrush, QColor, QImage, QPainter, \
                                QPalette, QPen, QPixmap, QPolygonF
    from PyQt5.QtWidgets import QAction, QApplication, QDialog, \
                                QFileDialog, QLabel, QMainWindow, \
                                QMessageBox, QPushButton, QScrollArea
else:
    from PyQt4.QtCore import Qt, QPointF, QRectF, QSize, QTimer
    from PyQt4.QtGui  import QAction, QApplication, QBrush, QColor, QDialog, \
                             QFileDialog, QImage, QLabel, QMainWindow, \
                             QMessageBox, QPainter, QPalette, QPen, QPixmap, \
                             QPolygonF, QPushButton, QScrollArea

import multiprocessing

from pipedviewer import WINDOW_CLOSED_MESSAGE
from pipedviewer.cmndhelperpq import CmndHelperPQ
from pipedviewer.scaledialogpq import ScaleDialogPQ


class PipedImagerPQ(QMainWindow):
    '''
    A PyQt graphics viewer that receives images and commands through
    a pipe.

    A command is a dictionary with string keys.  For example,
        { "action":"save",
          "filename":"ferret.png",
          "fileformat":"png" }

    The command { "action":"exit" } will shutdown the viewer.
    '''

    def __init__(self, cmndpipe, rspdpipe):
        '''
        Create a PyQt viewer which reads commands from the Pipe
        cmndpipe and writes responses back to rspdpipe.
        '''
        super(PipedImagerPQ, self).__init__()
        self.__cmndpipe = cmndpipe
        self.__rspdpipe = rspdpipe
        # ignore Ctrl-C
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        # unmodified image for creating the scene
        self.__sceneimage = None
        # bytearray of data for the above image
        self.__scenedata = None
        # flag set if in the process of reading image data from commands
        self.__loadingimage = False
        # width and height of the unmodified scene image
        # when the image is defined
        # initialize the width and height to values that will create
        # a viewer (mainWindow) of the right size
        self.__scenewidth = int(10.8 * self.physicalDpiX())
        self.__sceneheight = int(8.8 * self.physicalDpiY())
        # by default pay attention to any alpha channel values in colors
        self.__noalpha = False
        # initial default color for the background (opaque white)
        self.__lastclearcolor = QColor(0xFFFFFF)
        self.__lastclearcolor.setAlpha(0xFF)
        # scaling factor for creating the displayed scene
        self.__scalefactor = 1.0
        # automatically adjust the scaling factor to fit the window frame?
        self.__autoscale = True
        # minimum label width and height (for minimum scaling factor)
        # and minimum image width and height (for error checking)
        self.__minsize = 128
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
        # default file name and format for saving the image
        self.__lastfilename = "ferret.png"
        self.__lastformat = "png"
        # command helper object
        self.__helper = CmndHelperPQ(self)
        # create the menubar
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
        # set the initial size of the viewer
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
        QMessageBox.about(self, self.tr("About PipedImagerPQ"),
            self.tr("\n" \
            "PipedImagerPQ is a graphics viewer application that receives its " \
            "displayed image and commands primarily from another application " \
            "through a pipe.  A limited number of commands are provided by the " \
            "viewer itself to allow saving and some manipulation of the " \
            "displayed image.  The controlling application, however, may be " \
            "unaware of these modifications made to the image. " \
            "\n\n" \
            "PipedImagerPQ was developed by the Thermal Modeling and Analysis " \
            "Project (TMAP) of the National Oceanographic and Atmospheric " \
            "Administration's (NOAA) Pacific Marine Environmental Lab (PMEL). "))

    def aboutQtMsg(self):
        QMessageBox.aboutQt(self, self.tr("About Qt"))

    def ignoreAlpha(self):
        '''
        Return whether the alpha channel in colors should always be ignored.
        '''
        return self.__noalpha

    def updateScene(self):
        '''
        Clear the displayed scene using self.__lastclearcolor,
        then draw the scaled current image.
        '''
        # get the scaled scene size
        labelwidth = int(self.__scalefactor * self.__scenewidth + 0.5)
        labelheight = int(self.__scalefactor * self.__sceneheight + 0.5)
        # Create the new pixmap for the label to display
        newpixmap = QPixmap(labelwidth, labelheight)
        newpixmap.fill(self.__lastclearcolor)
        if self.__sceneimage != None:
            # Draw the scaled image to the pixmap
            mypainter = QPainter(newpixmap)
            trgrect = QRectF(0.0, 0.0, float(labelwidth),
                                       float(labelheight))
            srcrect = QRectF(0.0, 0.0, float(self.__scenewidth),
                                       float(self.__sceneheight))
            mypainter.drawImage(trgrect, self.__sceneimage, srcrect, Qt.AutoColor)
            mypainter.end()
        # Assign the new pixmap to the label
        self.__label.setPixmap(newpixmap)
        # set the label size and values
        # so the scrollarea knows of the new size
        self.__label.setMinimumSize(labelwidth, labelheight)
        self.__label.resize(labelwidth, labelheight)
        # update the label from the new pixmap
        self.__label.update()

    def clearScene(self, bkgcolor=None):
        '''
        Deletes the scene image and fills the label with bkgcolor.
        If bkgcolor is None or an invalid color, the color used is
        the one used from the last clearScene or redrawScene call
        with a valid color (or opaque white if a color has never
        been specified).
        '''
        # get the color to use for clearing (the background color)
        if bkgcolor:
            if bkgcolor.isValid():
                self.__lastclearcolor = bkgcolor
        # Remove the image and its bytearray
        self.__sceneimage = None
        self.__scenedata = None
        # Update the scene label using the current clearing color and image
        self.updateScene()

    def redrawScene(self, bkgcolor=None):
        '''
        Clear and redraw the displayed scene.
        '''
        # get the background color
        if bkgcolor:
            if bkgcolor.isValid():
                self.__lastclearcolor = bkgcolor
        # Update the scene label using the current clearing color and image
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.statusBar().showMessage( self.tr("Redrawing image") )
        try:
            self.updateScene()
        finally:
            self.statusBar().clearMessage()
            QApplication.restoreOverrideCursor()

    def resizeScene(self, width, height):
        '''
        Resize the scene to the given width and height in units of pixels.
        If the size changes, this deletes the current image and clear the
        displayed scene.
        '''
        newwidth = int(width + 0.5)
        if newwidth < self.__minsize:
            newwidth = self.__minsize
        newheight = int(height + 0.5)
        if newheight < self.__minsize:
            newheight = self.__minsize
        if (newwidth != self.__scenewidth) or (newheight != self.__sceneheight):
            # set the new size for the empty scene
            self.__scenewidth = newwidth
            self.__sceneheight = newheight
            # If auto-scaling, set scaling factor to 1.0 and resize the window
            if self.__autoscale:
                self.__scalefactor = 1.0
                barheights = self.menuBar().height() + self.statusBar().height()
                self.resize(newwidth + self.__framedelta,
                            newheight + self.__framedelta + barheights)
            # clear the scene with the last clearing color
            self.clearScene(None)

    def loadNewSceneImage(self, imageinfo):
        '''
        Create a new scene image from the information given in this
        and subsequent dictionaries imageinfo.  The image is created
        from multiple calls to this function since there is a limit
        on the size of a single object passed through a pipe.

        The first imageinfo dictionary given when creating an image
        must define the following key and value pairs:
            "width": width of the image in pixels
            "height": height of the image in pixels
            "stride": number of bytes in one line of the image
                      in the bytearray
        The scene image data is initialized to all zero (transparent)
        at this time.

        This initialization call must be followed by (multiple) calls
        to this method with imageinfo dictionaries defining the key
        and value pairs:
            "blocknum": data block number (1, 2, ... numblocks)
            "numblocks": total number of image data blocks
            "startindex": index in the bytearray of image data
                          where this block of image data starts
            "blockdata": this block of data as a bytearray

        On receipt of the last block of data (blocknum == numblocks)
        the scene image will be created and the scene will be updated.

        Raises:
            KeyError - if one of the above keys is not given
            ValueError - if a value for a key is not valid
        '''
        if not self.__loadingimage:
            # prepare for a new image data from subsequent calls
            # get dimensions of the new image
            myimgwidth = int( imageinfo["width"] )
            myimgheight = int( imageinfo["height"] )
            myimgstride = int( imageinfo["stride"] )
            if (myimgwidth < self.__minsize) or (myimgheight < self.__minsize):
                raise ValueError("image width and height cannot be less than %s" % str(self.__minsize))
            # Newer PyQt versions allow separate specification of the stride
            if myimgstride != 4 * myimgwidth:
                raise ValueError("image stride is not four times the image width")
            # create the bytearray to contain the new scene data
            # automatically initialized to zero
            self.__scenedata = bytearray(myimgstride * myimgheight)
            self.__scenewidth = myimgwidth
            self.__sceneheight = myimgheight
            # set the flag for subsequent calls to this method
            self.__loadingimage = True
            # change the cursor to warn the user this may take some time
            QApplication.setOverrideCursor(Qt.WaitCursor)
            # put up an appropriate status message
            self.statusBar().showMessage( self.tr("Loading new image") )
            return
        # loading an image; add the next block of data
        myblocknum = int( imageinfo["blocknum"] )
        mynumblocks = int( imageinfo["numblocks"] )
        mystartindex = int( imageinfo["startindex"] )
        myblockdata = imageinfo["blockdata"]
        if (myblocknum < 1) or (myblocknum > mynumblocks):
            self.statusBar().clearMessage()
            QApplication.restoreOverrideCursor()
            raise ValueError("invalid image data block number or number of blocks")
        if (mystartindex < 0) or (mystartindex >= len(self.__scenedata)):
            self.statusBar().clearMessage()
            QApplication.restoreOverrideCursor()
            raise ValueError("invalid start index for an image data block")
        myblocksize = len(myblockdata)
        myendindex = mystartindex + myblocksize
        if (myblocksize < 1) or (myendindex > len(self.__scenedata)):
            self.statusBar().clearMessage()
            QApplication.restoreOverrideCursor()
            raise ValueError("invalid length of an image data block")
        # update the status message to show progress
        self.statusBar().showMessage( self.tr("Loading new image (block %s of %s)" % \
                                              (str(myblocknum),str(mynumblocks))) )
        # assign the data
        self.__scenedata[mystartindex:myendindex] = myblockdata
        # if this is the last block of data, create and display the scene image
        if myblocknum == mynumblocks:
            self.__loadingimage = False
            self.statusBar().showMessage( self.tr("Creating new image") )
            try:
                self.__sceneimage = QImage(self.__scenedata,
                                           self.__scenewidth,
                                           self.__sceneheight,
                                           QImage.Format_ARGB32_Premultiplied)
                self.statusBar().showMessage( self.tr("Drawing new image") )
                # update the displayed scene in the label
                self.updateScene()
            finally:
                # clear the status message
                self.statusBar().clearMessage()
                # restore the cursor back to normal
                QApplication.restoreOverrideCursor()

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
            False if the a window resize command was issued
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
                # Update the scene label using the current clearing color and image
                QApplication.setOverrideCursor(Qt.WaitCursor)
                self.statusBar().showMessage( self.tr("Scaling image") )
                try:
                    self.updateScene()
                finally:
                    self.statusBar().clearMessage()
                    QApplication.restoreOverrideCursor()
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
                    mwwidth = int(0.9 * scrnrect.width() + 0.5)
                if mwheight > 0.95 * scrnrect.height():
                    mwheight = int(0.9 * scrnrect.height() + 0.5)
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
                          "PNG - Portable Networks Graphics (*.png)" ),
                        ( "jpeg",
                          "JPEG - Joint Photographic Experts Group (*.jpeg *.jpg *.jpe)" ),
                        ( "tiff",
                          "TIFF - Tagged Image File Format (*.tiff *.tif)" ),
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
            # getSaveFileNameAndFilter; tr returns QStrings in PyQt4 (Python2)
            (fileName, fileFilter) = QFileDialog.getSaveFileNameAndFilter(self,
                 self.tr("Save the current image as "), self.tr(self.__lastfilename), self.tr(filters))
        else:
            # getSaveFileName; tr returns Python unicode strings in PySide2 and PyQt5 (Python3)
            (fileName, fileFilter) = QFileDialog.getSaveFileName(self,
                self.tr("Save the current image as "), self.tr(self.__lastfilename), self.tr(filters))
        if fileName:
            for (fmt, fmtQName) in formattypes:
                if self.tr(fmtQName) == fileFilter:
                    fileFormat = fmt
                    break
            else:
                raise RuntimeError("Unexpected file format name '%s'" % fileFilter)
            self.saveSceneToFile(fileName, fileFormat, None, None)
            self.__lastfilename = fileName
            self.__lastformat = fileFormat

    def saveSceneToFile(self, filename, imageformat, transparent, rastsize):
        '''
        Save the current scene to the named file.

        If imageformat is empty or None, the format is guessed from
        the filename extension.

        If transparent is False, the entire scene is initialized
        to the last clearing color.

        If given, rastsize is the pixels size of the saved image.
        If rastsize is not given, the saved image will be saved
        at the current scaled image size.
        '''
        # This could be called when there is no image present.
        # If this is the case, ignore the call.
        if ( self.__sceneimage == None ):
            return
        if not imageformat:
            # Guess the image format from the filename extension
            # This is only done to silently change gif to png
            fileext = ( os.path.splitext(filename)[1] ).lower()
            if fileext == '.gif':
                myformat = 'gif'
            else:
                # let QImage figure out the format
                myformat = None
        else:
            myformat = imageformat.lower()

        if myformat == 'gif':
            # Silently convert gif filename and format to png
            myformat = 'png'
            myfilename = os.path.splitext(filename)[0] + ".png"
        else:
            myfilename = filename
        # set the cursor and status message to indicate a save is happending
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.statusBar().showMessage( self.tr("Saving image") )
        try:
            if rastsize:
                imagewidth = int(rastsize.width() + 0.5)
                imageheight = int(rastsize.height() + 0.5)
            else:
                imagewidth = int(self.__scenewidth * self.__scalefactor + 0.5)
                imageheight = int(self.__sceneheight * self.__scalefactor + 0.5)
            myimage = QImage( QSize(imagewidth, imageheight),
                            QImage.Format_ARGB32_Premultiplied )
            # Initialize the image
            if not transparent:
                # Clear the image with self.__lastclearcolor
                fillint = self.__helper.computeARGB32PreMultInt(self.__lastclearcolor)
            else:
                fillint = 0
            myimage.fill(fillint)
            # draw the scaled scene to this QImage
            mypainter = QPainter(myimage)
            trgrect = QRectF(0.0, 0.0, float(imagewidth),
                                       float(imageheight))
            srcrect = QRectF(0.0, 0.0, float(self.__scenewidth),
                                       float(self.__sceneheight))
            mypainter.drawImage(trgrect, self.__sceneimage, srcrect, Qt.AutoColor)
            mypainter.end()
            # save the image to file
            if not myimage.save(myfilename, myformat):
                raise ValueError("Unable to save the plot as " + myfilename)
        finally:
            self.statusBar().clearMessage()
            QApplication.restoreOverrideCursor()

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
        method to deal with this command.  Raises a KeyError
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
        elif cmndact == "redraw":
            try:
                bkgcolor = self.__helper.getColorFromCmnd(cmnd)
            except KeyError:
                bkgcolor = None
            self.redrawScene(bkgcolor)
        elif cmndact == "rescale":
            self.scaleScene(float(cmnd["factor"]), True)
        elif cmndact == "resize":
            mysize = self.__helper.getSizeFromCmnd(cmnd)
            self.resizeScene(mysize.width(), mysize.height())
        elif cmndact == "newImage":
            self.loadNewSceneImage(cmnd)
        elif cmndact == "save":
            filename = cmnd["filename"]
            fileformat = cmnd.get("fileformat", None)
            try:
                bkgcolor = self.__helper.getColorFromCmnd(cmnd)
            except KeyError:
                bkgcolor = None
            rastsize = self.__helper.getSizeFromCmnd(cmnd["rastsize"])
            self.saveSceneToFile(filename, fileformat, bkgcolor, rastsize)
        elif cmndact == "setTitle":
            self.setWindowTitle(cmnd["title"])
        elif cmndact == "imgname":
            myvalue = cmnd.get("name", None)
            if myvalue:
                self.__lastfilename = myvalue
            myvalue = cmnd.get("format", None)
            if myvalue:
                self.__lastformat = myvalue.lower()
        elif cmndact == "show":
            if not self.isVisible():
                self.show()
        elif cmndact == "noalpha":
            # ignore any alpha channel values in colors
            self.__noalpha = True
        else:
            raise ValueError("Unknown command action %s" % str(cmndact))


class PipedImagerPQProcess(multiprocessing.Process):
    '''
    A Process specifically tailored for creating a PipedImagerPQ.
    '''
    def __init__(self, cmndpipe, rspdpipe):
        '''
        Create a Process that will produce a PipedImagerPQ
        attached to the given Pipes when run.
        '''
        super(PipedImagerPQProcess,self).__init__(group=None, target=None, name='PipedImagerPQ')
        self.__cmndpipe = cmndpipe
        self.__rspdpipe = rspdpipe
        self.__app = None
        self.__viewer = None

    def run(self):
        '''
        Create a PipedImagerPQ that is attached
        to the Pipe of this instance.
        '''
        self.__app = QApplication(["PipedImagerPQ"])
        self.__viewer = PipedImagerPQ(self.__cmndpipe, self.__rspdpipe)
        myresult = self.__app.exec_()
        sys.exit(myresult)


#
# The following are for testing this module
#

class _CommandSubmitterPQ(QDialog):
    '''
    Testing dialog for controlling the addition of commands to a pipe.
    Used for testing PipedImagerPQ in the same process as the viewer.
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
            cmndstr = str(self.__cmndlist[self.__nextcmnd])
            if len(cmndstr) > 188:
                cmndstr = cmndstr[:188] + '...'
            print("Command: %s" % cmndstr)
            self.__cmndpipe.send(self.__cmndlist[self.__nextcmnd])
            self.__nextcmnd += 1
            while self.__rspdpipe.poll(0.1):
                print("Response: %s" % str(self.__rspdpipe.recv()))
        except IndexError:
            self.__rspdpipe.close()
            self.__cmndpipe.close()
            self.close()


def _test_pipedimagerpq():
    # vertices of a pentagon (roughly) centered in a 1000 x 1000 square
    pentagonpts = ( (504.5, 100.0), (100.0, 393.9),
                    (254.5, 869.4), (754.5, 869.4),
                    (909.0, 393.9),  )
    linepts = ( (350,  50),
                (200, 150),
                (400, 250),
                (300, 350),
                (150, 250),
                (100, 450) )
    # start PyQt
    testapp = QApplication(["PipedImagerPQ"])
    # create the list of commands to submit
    drawcmnds = []
    drawcmnds.append( { "action":"setTitle", "title":"Tester" } )
    drawcmnds.append( { "action":"show" } )
    drawcmnds.append( { "action":"clear", "color":"black"} )
    drawcmnds.append( { "action":"screenInfo"} )
    # create the image to be displayed
    testimage = QImage(500, 500, QImage.Format_ARGB32_Premultiplied)
    # initialize a black background
    testimage.fill(0xFF000000)
    # draw some things in the image
    testpainter = QPainter(testimage)
    testpainter.setBrush( QBrush(QColor(0, 255, 0, 128), Qt.SolidPattern) )
    testpainter.setPen( QPen(QBrush(QColor(255, 0, 0, 255), Qt.SolidPattern),
                         5.0, Qt.SolidLine, Qt.SquareCap, Qt.MiterJoin) )
    testpainter.drawRect( QRectF(5.0, 255.0, 240.0, 240.0) )
    testpainter.setBrush( QBrush(QColor(0, 0, 255, 255), Qt.SolidPattern) )
    testpainter.setPen( QPen(QBrush(QColor(0, 0, 0, 255), Qt.SolidPattern),
                         5.0, Qt.DashLine, Qt.RoundCap, Qt.RoundJoin) )
    testpainter.drawPolygon( QPolygonF(
            [ QPointF(.25 * ptx, .25 * pty + 250) for (ptx, pty) in pentagonpts ] ) )
    testpainter.setBrush( Qt.NoBrush )
    testpainter.setPen( QPen(QBrush(QColor(255, 255, 255, 255), Qt.SolidPattern),
                         3.0, Qt.DashLine, Qt.RoundCap, Qt.RoundJoin) )
    testpainter.drawPolyline( QPolygonF(
            [ QPointF(pts, pty) for (pts, pty) in linepts ] ) )
    testpainter.end()
    # add the image command
    testimgwidth = testimage.width()
    testimgheight = testimage.height()
    testimgstride = testimage.bytesPerLine()
    # not a good way to get the pixel data
    testimgdata = bytearray(testimgheight * testimgstride)
    k = 0
    for pty in range(testimgheight):
        for ptx in range(testimgwidth):
            pixval = testimage.pixel(ptx, pty)
            (aval, rgbval) = divmod(pixval, 256 * 256 * 256)
            (rval, gbval) = divmod(rgbval, 256 * 256)
            (gval, bval) = divmod(gbval, 256)
            testimgdata[k] = bval
            k += 1
            testimgdata[k] = gval
            k += 1
            testimgdata[k] = rval
            k += 1
            testimgdata[k] = aval
            k += 1
    testblocksize = 4000
    testnumblocks = (testimgheight * testimgstride + testblocksize - 1) // testblocksize
    drawcmnds.append( { "action":"newImage",
                        "width":testimgwidth,
                        "height":testimgheight,
                        "stride":testimgstride } )
    for k in range(testnumblocks):
        if k < (testnumblocks - 1):
            blkdata = testimgdata[k*testblocksize:(k+1)*testblocksize]
        else:
            blkdata = testimgdata[k*testblocksize:]
        drawcmnds.append( { "action":"newImage",
                            "blocknum":k+1,
                            "numblocks":testnumblocks,
                            "startindex":k*testblocksize,
                            "blockdata":blkdata } )
    # finish the command list
    drawcmnds.append( { "action":"show" } )
    drawcmnds.append( { "action":"exit" } )
    # create a PipedImagerPQ in this process
    (cmndrecvpipe, cmndsendpipe) = multiprocessing.Pipe(False)
    (rspdrecvpipe, rspdsendpipe) = multiprocessing.Pipe(False)
    testviewer = PipedImagerPQ(cmndrecvpipe, rspdsendpipe)
    # create a command submitter dialog
    tester = _CommandSubmitterPQ(testviewer, cmndsendpipe,
                                   rspdrecvpipe, drawcmnds)
    tester.show()
    # let it all run
    testresult = testapp.exec_()
    if testresult != 0:
        sys.exit(testresult)

if __name__ == "__main__":
    _test_pipedimagerpq()
