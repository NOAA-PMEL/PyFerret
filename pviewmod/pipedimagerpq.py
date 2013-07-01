'''
PipedImagerPQ is a graphics viewer application written in PyQt4
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

from cmndhelperpq import CmndHelperPQ
from scaledialogpq import ScaleDialogPQ
from multiprocessing import Pipe, Process
import sys
import time
import os


class PipedImagerPQ(QMainWindow):
    '''
    A PyQt graphics viewer that receives images and commands through
    a pipe.

    A command is a dictionary with string keys.  For example,
        { "action":"save",
          "filename":"ferret.png",
          "fileformat":"png" }

    The command { "action":"exit" } will shutdown the viewer and is
    the only way the viewer can be closed.  GUI actions can only hide
    the viewer.
    '''

    def __init__(self, cmndpipe, rspdpipe):
        '''
        Create a PyQt viewer which reads commands from the Pipe
        cmndpipe and writes responses back to rspdpipe.
        '''
        super(PipedImagerPQ, self).__init__()
        self.__cmndpipe = cmndpipe
        self.__rspdpipe = rspdpipe
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
        self.__scenewidth = 816
        self.__sceneheight = 692
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
        # control whether the window will be destroyed or hidden
        self.__shuttingdown = False
        # command helper object
        self.__helper = CmndHelperPQ(self)
        # create the menubar
        self.createActions()
        self.createMenus()
        # set the initial size of the viewer
        mwwidth = self.__scenewidth + 4
        mwheight = self.__sceneheight + 4 \
                 + self.menuBar().height() \
                 + self.statusBar().height()
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
        # self.__hideact = QAction(self.tr("&Hide"), self,
        #                         shortcut=self.tr("Ctrl+H"),
        #                         statusTip=self.tr("Hide the viewer"),
        #                         triggered=self.hide)
        self.__aboutact = QAction(self.tr("&About"), self,
                                statusTip=self.tr("Show information about this viewer"),
                                triggered=self.aboutMsg)
        self.__aboutqtact = QAction(self.tr("About &Qt"), self,
                                statusTip=self.tr("Show information about the Qt library"),
                                triggered=self.aboutQtMsg)
        self.__exitact = QAction(self.tr("&Exit"), self,
                                statusTip=self.tr("Shut down the viewer"),
                                triggered=self.exitViewer)

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
        # sceneMenu.addSeparator()
        # sceneMenu.addAction(self.__hideact)
        helpMenu = menuBar.addMenu(menuBar.tr("&Help"))
        helpMenu.addAction(self.__aboutact)
        helpMenu.addAction(self.__aboutqtact)
        helpMenu.addSeparator()
        helpMenu.addAction(self.__exitact)

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
        QMessageBox.about(self, self.tr("About PipedImagerPQ"),
            self.tr("\n" \
            "PipedImagerPQ is a graphics viewer application that " \
            "receives its displayed image and commands primarily from " \
            "another application through a pipe.  A limited number " \
            "of commands are provided by the viewer itself to allow " \
            "saving and some manipulation of the displayed image.  " \
            "The controlling application, however, may be unaware " \
            "of these modifications made to the image. " \
            "\n\n" \
            "Normally, the controlling program will exit the viewer " \
            "when it is no longer needed.  The Help -> Exit menu item " \
            "should not normally be used.  It is provided when problems " \
            "occur and the controlling program cannot shut down the " \
            "viewer properly. " \
            "\n\n" \
            "PipedImagerPQ was developed by the Thermal Modeling and Analysis " \
            "Project (TMAP) of the National Oceanographic and Atmospheric " \
            "Administration's (NOAA) Pacific Marine Environmental Lab (PMEL). "))

    def aboutQtMsg(self):
        QMessageBox.aboutQt(self, self.tr("About Qt"))

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
            painter = QPainter(newpixmap)
            trgrect = QRectF(0.0, 0.0, float(labelwidth),
                                       float(labelheight))
            srcrect = QRectF(0.0, 0.0, float(self.__scenewidth),
                                       float(self.__sceneheight))
            painter.drawImage(trgrect, self.__sceneimage, srcrect, Qt.AutoColor)
            painter.end()
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
                self.resize(newwidth+4, newheight+4+barheights)
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
            imgwidth = int( imageinfo["width"] )
            imgheight = int( imageinfo["height"] )
            imgstride = int( imageinfo["stride"] )
            if (imgwidth < self.__minsize) or (imgheight < self.__minsize):
                raise ValueError( self.tr("image width and height cannot be less than %1") \
                                      .arg(str(self.__minsize)) )
            # Newer PyQt versions allow separate specification of the stride
            if imgstride != 4 * imgwidth:
                raise ValueError( self.tr("image stride is not four times the image width") )
            # create the bytearray to contain the new scene data
            # automatically initialized to zero
            self.__scenedata = bytearray(imgstride * imgheight)
            self.__scenewidth = imgwidth
            self.__sceneheight = imgheight
            # set the flag for subsequent calls to this method
            self.__loadingimage = True
            # change the cursor to warn the user this may take some time
            QApplication.setOverrideCursor(Qt.WaitCursor)
            # put up an appropriate status message
            self.statusBar().showMessage( self.tr("Loading new image") )
            return
        # loading an image; add the next block of data
        blocknum = int( imageinfo["blocknum"] )
        numblocks = int( imageinfo["numblocks"] )
        startindex = int( imageinfo["startindex"] )
        blockdata = imageinfo["blockdata"]
        if (blocknum < 1) or (blocknum > numblocks):
            self.statusBar().clearMessage()
            QApplication.restoreOverrideCursor()
            raise ValueError( self.tr("invalid image data block number or number of blocks") )
        if (startindex < 0) or (startindex >= len(self.__scenedata)):
            self.statusBar().clearMessage()
            QApplication.restoreOverrideCursor()
            raise ValueError( self.tr("invalid start index for an image data block") )
        blocksize = len(blockdata)
        endindex = startindex + blocksize
        if (blocksize < 1) or (endindex > len(self.__scenedata)):
            self.statusBar().clearMessage()
            QApplication.restoreOverrideCursor()
            raise ValueError( self.tr("invalid length of an image data block") )
        # update the status message to show progress
        self.statusBar().showMessage( self.tr("Loading new image (block %1 of %2)") \
                                          .arg(str(blocknum)).arg(str(numblocks)) )
        # assign the data
        self.__scenedata[startindex:endindex] = blockdata
        # if this is the last block of data, create and display the scene image
        if blocknum == numblocks:
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
            True if the scene was resized
            False if the a new resize command was issued
        '''
        barheights = self.menuBar().height() + self.statusBar().height()

        # get the size for the central widget
        cwheight = self.height() - barheights - 4
        heightsf = float(cwheight) / float(self.__sceneheight)

        cwwidth = self.width() - 4
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
        if (abs(cwheight - newcwheight) <= 4) and (abs(cwwidth - newcwwidth) <= 4):
            self.scaleScene(factor, False)
            return True
        else:
            self.resize(newcwwidth+4, newcwheight+4+barheights)
            return False

    def scaleScene(self, factor, resizewin):
        '''
        Scales both the horizontal and vertical directions by factor.
        Scaling factors are not accumulative.  So if the scene was
        already scaled, that scaling is "removed" before this scaling
        factor is applied.  If resizewin is True, the main window is 
        resize to accommodate this new scaled scene size.
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
            # Update the scene label using the current clearing color and image
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.statusBar().showMessage( self.tr("Scaling image") )
            try:
                self.updateScene()
            finally:
                self.statusBar().clearMessage()
                QApplication.restoreOverrideCursor()
        if resizewin:
            # resize the main window 
            barheights = self.menuBar().height() + self.statusBar().height()
            mwheight = newlabheight + barheights + 4
            mwwidth = newlabwidth + 4
            # Do not exceed 7/8 of the available real estate on the screen.
            # If autoscaling is in effect, the resize will trigger 
            # any required adjustments.
            scrnrect = QApplication.desktop().availableGeometry()
            if mwwidth > 0.875 * scrnrect.width():
                mwwidth = int(0.875 * scrnrect.width() + 0.5)
            if mwheight > 0.875 * scrnrect.height():
                mwheight = int(0.875 * scrnrect.height() + 0.5)
            self.resize(mwwidth, mwheight)

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
        # getSaveFileNameAndFilter does not want to accept a default filter
        # for (fmt, fmtQName) in formattypes:
        #     if self.__lastformat == fmt:
        #         dfltfilter = fmtQName
        #         break
        # else:
        #     dfltfilter = formattypes[0][1]
        # getSaveFileNameAndFilter is a PyQt (but not Qt?) method
        (fileName, fileFilter) = QFileDialog.getSaveFileNameAndFilter(self,
                                      self.tr("Save the current image as "),
                                      self.__lastfilename, filters)
        if fileName:
            for (fmt, fmtQName) in formattypes:
                if fmtQName.compare(fileFilter) == 0:
                    fileFormat = fmt
                    break
            else:
                raise RuntimeError( self.tr("Unexpected file format name '%1'") \
                                        .arg(fileFilter) )
            self.saveSceneToFile(fileName, fileFormat, None)
            self.__lastfilename = fileName
            self.__lastformat = fileFormat

    def saveSceneToFile(self, filename, imageformat, bkgcolor):
        '''
        Save the current scene to the named file.
        
        If imageformat is empty or None, the format is guessed from
        the filename extension.

        If bkgcolor is given, the entire scene is initialized
        to this color, using a filled rectangle for vector images.
        If bkgcolor is not given, the last clearing color is used.
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
            imagewidth = int(self.__scenewidth * self.__scalefactor + 0.5)
            imageheight = int(self.__sceneheight * self.__scalefactor + 0.5)
            image = QImage( QSize(imagewidth, imageheight),
                            QImage.Format_ARGB32_Premultiplied )
            # Initialize the image
            if bkgcolor:
                fillint = self.__helper.computeARGB32PreMultInt(bkgcolor)
            else:
                # Clear the image with self.__lastclearcolor
                fillint = self.__helper.computeARGB32PreMultInt(self.__lastclearcolor)
            image.fill(fillint)
            # draw the scaled scene to this QImage
            painter = QPainter(image)
            trgrect = QRectF(0.0, 0.0, float(imagewidth),
                                       float(imageheight))
            srcrect = QRectF(0.0, 0.0, float(self.__scenewidth),
                                       float(self.__sceneheight))
            painter.drawImage(trgrect, self.__sceneimage, srcrect, Qt.AutoColor)
            painter.end()
            # save the image to file
            image.save(myfilename, myformat)
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
            raise ValueError( self.tr("Unknown command %1").arg(str(cmnd)) )

        if cmndact == "clear":
            try:
                bkgcolor = self.__helper.getColorFromCmnd(cmnd)
            except KeyError:
                bkgcolor = None
            self.clearScene(bkgcolor)
        elif cmndact == "exit":
            self.exitViewer()
        elif cmndact == "hide":
            self.hide()
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
            newscale = float(cmnd["factor"])
            if newscale <= 0.0:
                raise ValueError("invalid scaling factor")
            self.scaleScene(newscale, True)
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
            self.saveSceneToFile(filename, fileformat, bkgcolor)
        elif cmndact == "setTitle":
            self.setWindowTitle(cmnd["title"])
        elif cmndact == "imgname":
            value = cmnd.get("name", None)
            if value:
                self.__lastfilename = value;
            value = cmnd.get("format", None)
            if value:
                self.__lastformat = value.lower();
        elif cmndact == "show":
            if self.isHidden():
                self.showNormal()
        else:
            raise ValueError( self.tr("Unknown command action %1") \
                                  .arg(str(cmndact)) )


class PipedImagerPQProcess(Process):
    '''
    A Process specifically tailored for creating a PipedImagerPQ.
    '''
    def __init__(self, cmndpipe, rspdpipe):
        '''
        Create a Process that will produce a PipedImagerPQ
        attached to the given Pipes when run.
        '''
        Process.__init__(self)
        self.__cmndpipe = cmndpipe
        self.__rspdpipe = rspdpipe

    def run(self):
        '''
        Create a PipedImagerPQ that is attached
        to the Pipe of this instance.
        '''
        self.__app = QApplication(["PipedImagerPQ"])
        self.__viewer = PipedImagerPQ(self.__cmndpipe, self.__rspdpipe)
        result = self.__app.exec_()
        self.__cmndpipe.close()
        self.__rspdpipe.close()
        SystemExit(result)


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
            cmndstr = str(self.__cmndlist[self.__nextcmnd])
            if len(cmndstr) > 188:
                cmndstr = cmndstr[:188] + '...'
            print "Command: %s" % cmndstr
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
    linepts = ( (350,  50),
                (200, 150),
                (400, 250),
                (300, 350),
                (150, 250),
                (100, 450) )
    # start PyQt
    app = QApplication(["PipedImagerPQ"])
    # create the list of commands to submit
    drawcmnds = []
    drawcmnds.append( { "action":"setTitle", "title":"Tester" } )
    drawcmnds.append( { "action":"show" } )
    drawcmnds.append( { "action":"clear", "color":"black"} )
    drawcmnds.append( { "action":"screenInfo"} )
    # create the image to be displayed
    image = QImage(500, 500, QImage.Format_ARGB32_Premultiplied)
    # initialize a black background
    image.fill(0xFF000000)
    # draw some things in the image
    painter = QPainter(image)
    painter.setBrush( QBrush(QColor(0, 255, 0, 128), Qt.SolidPattern) )
    painter.setPen( QPen(QBrush(QColor(255, 0, 0, 255), Qt.SolidPattern),
                         5.0, Qt.SolidLine, Qt.SquareCap, Qt.MiterJoin) )
    painter.drawRect( QRectF(5.0, 255.0, 240.0, 240.0) )
    painter.setBrush( QBrush(QColor(0, 0, 255, 255), Qt.SolidPattern) )
    painter.setPen( QPen(QBrush(QColor(0, 0, 0, 255), Qt.SolidPattern),
                         5.0, Qt.DashLine, Qt.RoundCap, Qt.RoundJoin) )
    painter.drawPolygon( QPolygonF(
            [ QPointF(.25 * ptx, .25 * pty + 250) for (ptx, pty) in pentagonpts ] ) )
    painter.setBrush( Qt.NoBrush )
    painter.setPen( QPen(QBrush(QColor(255, 255, 255, 255), Qt.SolidPattern),
                         3.0, Qt.DashLine, Qt.RoundCap, Qt.RoundJoin) )
    painter.drawPolyline( QPolygonF(
            [ QPointF(pts, pty) for (pts, pty) in linepts ] ) )
    painter.end()
    # add the image command
    imgwidth = image.width()
    imgheight = image.height()
    imgstride = image.bytesPerLine()
    # not a good way to get the pixel data
    imgdata = bytearray(imgheight * imgstride)
    k = 0
    for pty in xrange(imgheight):
        for ptx in xrange(imgwidth):
            pixval = image.pixel(ptx, pty)
            (aval, rgbval) = divmod(pixval, 256 * 256 * 256)
            (rval, gbval) = divmod(rgbval, 256 * 256)
            (gval, bval) = divmod(gbval, 256)
            imgdata[k] = bval
            k += 1
            imgdata[k] = gval
            k += 1
            imgdata[k] = rval
            k += 1
            imgdata[k] = aval
            k += 1
    blocksize = 2000
    numblocks = (imgheight * imgstride + blocksize - 1) // blocksize
    drawcmnds.append( { "action":"newImage",
                        "width":imgwidth,
                        "height":imgheight,
                        "stride":imgstride } )
    for k in xrange(numblocks):
        if k < (numblocks - 1):
            blkdata = imgdata[k*blocksize:(k+1)*blocksize]
        else:
            blkdata = imgdata[k*blocksize:]
        drawcmnds.append( { "action":"newImage",
                            "blocknum":k+1,
                            "numblocks":numblocks,
                            "startindex":k*blocksize,
                            "blockdata":blkdata } )
    # finish the command list
    drawcmnds.append( { "action":"show" } )
    drawcmnds.append( { "action":"exit" } )
    # create a PipedImagerPQ in this process
    cmndrecvpipe, cmndsendpipe = Pipe(False)
    rspdrecvpipe, rspdsendpipe = Pipe(False)
    viewer = PipedImagerPQ(cmndrecvpipe, rspdsendpipe)
    # create a command submitter dialog
    tester = _CommandSubmitterPQ(viewer, cmndsendpipe,
                                   rspdrecvpipe, drawcmnds)
    tester.show()
    # let it all run
    result = app.exec_()
    if result != 0:
        sys.exit(result)

