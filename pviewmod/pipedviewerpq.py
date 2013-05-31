'''
PipedViewerPQ is a graphics viewer application written in PyQt4
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

import sip
try:
    sip.setapi('QVariant', 2)
except AttributeError:
    pass

from PyQt4.QtCore import Qt, QPointF, QRect, QRectF, QSize, QString, QTimer
from PyQt4.QtGui  import QAction, QApplication, QBrush, QColor, QDialog, \
                         QFileDialog, QImage, QLabel, QMainWindow, \
                         QMessageBox, QPainter, QPalette, QPen, QPicture, \
                         QPixmap, QPolygonF, QPrintDialog, QPrinter, \
                         QPushButton, QScrollArea

try:
    from PyQt4.QtSvg  import QSvgGenerator
    HAS_QSvgGenerator = True
except ImportError:
    HAS_QSvgGenerator = False

from cmndhelperpq import CmndHelperPQ
from scaledialogpq import ScaleDialogPQ
from multiprocessing import Pipe, Process
import sys
import time
import os
import math


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

    The command { "action":"exit" } will shutdown the viewer and is
    the only way the viewer can be closed.  GUI actions can only hide
    the viewer.
    '''

    def __init__(self, cmndpipe, rspdpipe):
        '''
        Create a PyQt viewer which reads commands from the Pipe
        cmndpipe and writes responses back to rspdpipe.
        '''
        super(PipedViewerPQ, self).__init__()
        self.__cmndpipe = cmndpipe
        self.__rspdpipe = rspdpipe
        # default scene size
        self.__scenewidth = 840
        self.__sceneheight = 720
        # initial default color for the background (opaque white)
        self.__lastclearcolor = QColor(0xFFFFFF)
        self.__lastclearcolor.setAlpha(0xFF)
        # List of QPictures creating the current scene
        self.__viewpics = [ ]
        # QPicture/QPainter pair for the current view
        self.__activepicture = None
        self.__activepainter = None
        # Antialias when drawing?
        self.__antialias = False
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
        # Control whether the window will be destroyed or hidden
        self.__shuttingdown = False
        # command helper object
        self.__helper = CmndHelperPQ(self)
        # Create the menubar
        self.createActions()
        self.createMenus()
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

    def showEvent(self, event):
        '''
        When the viewer is going to be shown, make sure all
        the current pictures are displayed in the scene.
        '''
        # update, ignoring the visibility flags
        self.drawLastPictures(True)
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
        QMessageBox.about(self, self.tr("About PipedViewerPQ"),
            self.tr("\n" \
            "PipedViewerPQ is a graphics viewer application that " \
            "receives its drawing and other commands primarily from " \
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
            "PipedViewerPQ was developed by the Thermal Modeling and Analysis " \
            "Project (TMAP) of the National Oceanographic and Atmospheric " \
            "Administration's (NOAA) Pacific Marine Environmental Lab (PMEL). "))

    def aboutQtMsg(self):
        QMessageBox.aboutQt(self, self.tr("About Qt"))

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
            mymsg = self.tr("%s (piece %%1 of %%2)" % statusmsg)
            endstr = str(len(self.__viewpics))
        else:
            mymsg = self.tr("%s (piece %%1)" % statusmsg)
            endstr = None
        # get the origin for drawing the pictures after scaling
        myorigin = QPointF(leftx, uppery)
        # set the scaling factor for the pictures
        painter.scale(scalefactor, scalefactor)
        modrects = [ ]
        # draw the appropriate pictures
        k = first
        for viewpic in self.__viewpics[first:]:
            k += 1
            # show the progress message
            if endstr != None:
                self.statusBar().showMessage( mymsg.arg(str(k)).arg(endstr) )
            else:
                self.statusBar().showMessage( mymsg.arg(str(k)) )
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
            if self.isHidden() or self.isMinimized():
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
            wascleared = False
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
            # Redraw the scene from the beginning
            self.redrawScene()

    def inquireSceneScale(self):
        '''
        Prompt the user for the desired scaling factor for the scene.
        '''
        labelwidth = int(self.__scenewidth * self.__scalefactor + 0.5)
        labelheight = int(self.__sceneheight * self.__scalefactor + 0.5)
        scaledlg = ScaleDialogPQ(self.tr("Image Size Scaling"),
                       self.tr("Scaling factor (both horiz. and vert.) for the image"),
                       self.__scalefactor, labelwidth, labelheight,
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
            # mark that the pixmap needs to be recreated
            self.__createpixmap = True
            # Redraw the scene from the beginning
            self.redrawScene()

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
                        ( "pdf",
                          self.tr("PDF - Portable Document Format (*.pdf)") ),
                        ( "ps",
                          self.tr("PS - PostScript (*.ps)") ),
                        ( "bmp",
                          self.tr("BMP - Windows Bitmap (*.bmp)") ),
                        ( "ppm",
                          self.tr("PPM - Portable Pixmap (*.ppm)") ),
                        ( "xpm",
                          self.tr("XPM - X11 Pixmap (*.xpm)") ),
                        ( "xbm",
                          self.tr("XBM - X11 Bitmap (*.xbm)") ), ]
        if HAS_QSvgGenerator:
            formattypes.insert(5, ( "svg",
                          self.tr("SVG - Scalable Vector Graphics (*.svg)") ) )
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
            self.saveSceneToFile(fileName, fileFormat, None, True)
            self.__lastfilename = fileName
            self.__lastformat = fileFormat

    def saveSceneToFile(self, filename, imageformat, bkgcolor, showPrintDialog):
        '''
        Save the current scene to the named file.  If imageformat
        is empty or None, the format is guessed from the filename
        extension.

        If bkgcolor is given, the entire scene is initialized
        to this color, using a filled rectangle for vector images.
        If bkgcolor is not given, the last clearing color is used.

        If showPrintDialog is True, the standard printer options
        dialog will be shown for PostScript and PDF formats,
        allowing customizations to the file to be created.
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
            raise ValueError( self.tr("Your version of Qt does not " \
                                      "support generation of SVG files") )

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
            # Default paper size (letter)
            try:
                printer.setPaperSize(QPrinter.Letter)
            except AttributeError:
                # setPaperSize introduced in 4.4 and made setPageSize obsolete
                # but RHEL5 Qt4 is 4.2
                printer.setPageSize(QPrinter.Letter)
            # Default orientation
            if ( self.__scenewidth > self.__sceneheight ):
                printer.setOrientation(QPrinter.Landscape)
            else:
                printer.setOrientation(QPrinter.Portrait)
            # Since printing to file (and not a printer), use the full page
            # Also, ferret already has incorporated a margin in the drawing
            printer.setFullPage(True)
            # Interactive modifications?
            if showPrintDialog:
                # bring up a dialog to allow the user to tweak the default settings
                printdialog = QPrintDialog(printer, self)
                printdialog.setWindowTitle(
                            self.tr("Save Image PS/PDF Options (Margins Ignored)"))
                if printdialog.exec_() != QDialog.Accepted:
                    return
            # Determine the scaling factor and offsets for centering and filling the page
            pagerect = printer.pageRect()
            (printleftx, printuppery, printfactor) = \
                self.computeScaleAndOffset(pagerect.width(), pagerect.height(),
                                           printer.resolution())
            # Set up to send the drawing commands to the QPrinter
            painter = QPainter(printer)
            if bkgcolor:
                # Draw a rectangle filling the entire scene
                # with the given background color.
                # Only draw if not completely transparent
                if (bkgcolor.getRgb())[3] > 0:
                    painter.fillRect(QRectF(pagerect), bkgcolor)
            else:
                # Draw a rectangle filling the entire scene
                # with the last clearing color.
                # Only draw if not completely transparent
                if (self.__lastclearcolor.getRgb())[3] > 0:
                    painter.fillRect(QRectF(pagerect), self.__lastclearcolor)
            # Draw the scene to the printer
            self.paintScene(painter, 0, printleftx, printuppery, printfactor,
                            "Saving", False)
            painter.end()
        elif myformat == 'svg':
            # if HAS_QSvgGenerator is False, it should never get here
            generator = QSvgGenerator()
            generator.setFileName(myfilename)
            imagewidth = int(self.__scenewidth * self.__scalefactor + 0.5)
            imageheight = int(self.__sceneheight * self.__scalefactor + 0.5)
            generator.setSize( QSize(imagewidth, imageheight) )
            generator.setViewBox( QRect(0, 0, imagewidth, imageheight) )
            # paint the scene to this QSvgGenerator
            painter = QPainter(generator)
            if bkgcolor:
                # Draw a rectangle filling the entire scene
                # with the given background color.
                # Only draw if not completely transparent
                if (bkgcolor.getRgb())[3] > 0:
                    painter.fillRect( QRectF(0, 0, imagewidth, imageheight),
                                      bkgcolor )
            else:
                # Draw a rectangle filling the entire scene
                # with the last clearing color.
                # Only draw if not completely transparent
                if (self.__lastclearcolor.getRgb())[3] > 0:
                    painter.fillRect( QRectF(0, 0, imagewidth, imageheight),
                                      self.__lastclearcolor )
            self.paintScene(painter, 0, 0.0, 0.0, self.__scalefactor,
                            "Saving", False)
            painter.end()
        else:
            # ARGB32_Premultiplied is reported significantly faster than ARGB32
            imagewidth = int(self.__scenewidth * self.__scalefactor + 0.5)
            imageheight = int(self.__sceneheight * self.__scalefactor + 0.5)
            image = QImage( QSize(imagewidth, imageheight),
                            QImage.Format_ARGB32_Premultiplied )
            # Initialize the image
            # Note that completely transparent gives black for formats not supporting 
            # the alpha channel (JPEG) whereas ARGB32 with 0x00FFFFFF gives white
            if bkgcolor:
                fillint = self.__helper.computeARGB32PreMultInt(bkgcolor)
            else:
                # Clear the image with self.__lastclearcolor
                fillint = self.__helper.computeARGB32PreMultInt(self.__lastclearcolor)
            image.fill(fillint)
            # paint the scene to this QImage
            painter = QPainter(image)
            self.paintScene(painter, 0, 0.0, 0.0, self.__scalefactor,
                            "Saving", False)
            painter.end()
            # save the image to file
            image.save(myfilename, myformat)

    def computeScaleAndOffset(self, printwidth, printheight, printresolution):
        '''
        Computes the scaling factor and upper left coordinates required so
        the current scene will be centered and fill the page on described
        by printwidth, printheight, and printresolution.

        Arguments:
            printwidth: width of the print page in pixels
            printheight: height of the print page in pixels
            printresolution: resolution of the print page in DPI

        Returns:
            (leftx, uppery, scalefactor) giving the required
            left offset, top offset, and scaling factor for
            the paintScene method.
        '''
        # get the widths and heights of the printer page and label in inches
        fltprintresolution = float(printresolution)
        fltprintwidth = float(printwidth) / fltprintresolution
        fltprintheight = float(printheight) / fltprintresolution
        fltscenewidth = float(self.__scenewidth) / float(self.physicalDpiX())
        fltsceneheight = float(self.__sceneheight) / float(self.physicalDpiY())
        # Determine the scaling factor for filling the page
        scalefactor = min(fltprintwidth / fltscenewidth, 
                          fltprintheight / fltsceneheight)
        # Determine the offset to center the picture
        leftx  = 0.5 * fltprintresolution * \
                (fltprintwidth - scalefactor * fltscenewidth)
        uppery = 0.5 * fltprintresolution * \
                (fltprintheight - scalefactor * fltsceneheight)
        # Account for the scaling factor in the offsets
        leftx /= scalefactor
        uppery /= scalefactor
        return (leftx, uppery, scalefactor)

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
        elif cmndact == "dpi":
            windowdpi = ( self.physicalDpiX(), self.physicalDpiY() )
            self.__rspdpipe.send(windowdpi)
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
        elif cmndact == "resize":
            mysize = self.__helper.getSizeFromCmnd(cmnd)
            self.resizeScene(mysize.width(), mysize.height())
        elif cmndact == "save":
            filename = cmnd["filename"]
            fileformat = cmnd.get("fileformat", None)
            try:
                bkgcolor = self.__helper.getColorFromCmnd(cmnd)
            except KeyError:
                bkgcolor = None
            self.saveSceneToFile(filename, fileformat, bkgcolor, False)
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
            raise ValueError( self.tr("Invalid left, right view fractions: " \
                                      "left in pixels = %1, right in pixels = %2") \
                                  .arg(str(leftpixel)).arg(str(rightpixel)) )
        if (0.0 > toppixel) or (toppixel >= bottompixel) or (bottompixel > height):
            raise ValueError( self.tr("Invalid bottom, top view fractions: " \
                                      "top in pixels = %1, bottom in pixels = %2") \
                                  .arg(str(toppixel)).arg(str(bottompixel)) )
        # Create the view rectangle in device coordinates
        vrectf = QRectF(leftpixel, toppixel,
                       rightpixel - leftpixel, bottompixel - toppixel)
        # Assign the view factor for line widths, symbol sizes, and font sizes
        self.__viewfactor = math.hypot(vrectf.width() / 1000.0,
                                       vrectf.height() / 1000.0) / 1.414213562
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
            self.__viewpics.append(self.__activepicture)
            self.__drawcount = 0
            if update:
                # update the scene
                self.drawLastPictures(False)
        self.__activepicture = None

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
            "size": size of the symbol (scales with view size)
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
            mypen = QPen(mybrush, 15.0, Qt.SolidLine,
                         Qt.SquareCap, Qt.BevelJoin)
            self.__activepainter.setPen(mypen)
        # Unmodified symbols are 100x100 pixels 
        scalefactor = ptsize * self.viewScalingFactor() / 100.0
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
            mypen = QPen(mybrush, 0.0, Qt.SolidLine, Qt.SquareCap, Qt.BevelJoin)
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
            mypen = QPen(mybrush, 0.0, Qt.SolidLine, Qt.SquareCap, Qt.BevelJoin)
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
            self.__activepainter.drawText(0, 0, mytext)
            self.__drawcount += 1
        finally:
            # return the painter to the default state
            self.__activepainter.restore()
        # Limit the number of drawing commands per picture
        if self.__drawcount >= self.__maxdraws:
            self.updateScene()

    def viewScalingFactor(self):
        '''
        Return the scaling factor for line widths, point sizes, and
        font sizes for the current view.  If the view is 1000 x 1000
        pixels, 1.0 is returned.  The value changes linearly with the
        length of the diagonal of the scene.
        '''
        # the value is computed in the beginViewFromSides method
        return self.__viewfactor 


class PipedViewerPQProcess(Process):
    '''
    A Process specifically tailored for creating a PipedViewerPQ.
    '''
    def __init__(self, cmndpipe, rspdpipe):
        '''
        Create a Process that will produce a PipedViewerPQ
        attached to the given Pipes when run.
        '''
        Process.__init__(self)
        self.__cmndpipe = cmndpipe
        self.__rspdpipe = rspdpipe

    def run(self):
        '''
        Create a PipedViewerPQ that is attached
        to the Pipe of this instance.
        '''
        self.__app = QApplication(["PipedViewerPQ"])
        self.__viewer = PipedViewerPQ(self.__cmndpipe, self.__rspdpipe)
        result = self.__app.exec_()
        self.__cmndpipe.close()
        self.__rspdpipe.close()
        SystemExit(result)


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
    drawcmnds.append( { "action":"clear", "color":"black"} )
    drawcmnds.append( { "action":"dpi"} )
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
    drawcmnds.append( { "action":"drawText",
                        "text":"y=480",
                        "font":{"family":"Times", "size":50},
                        "fill":{"color":"red"},
                        "location":(50,480) } )
    drawcmnds.append( { "action":"drawText",
                        "text":"y=430",
                        "font":{"family":"Times", "size":50},
                        "fill":{"color":"red"},
                        "location":(50,430) } )
    drawcmnds.append( { "action":"drawText",
                        "text":"y=380",
                        "font":{"family":"Times", "size":50},
                        "fill":{"color":"red"},
                        "location":(50,380) } )
    drawcmnds.append( { "action":"drawText",
                        "text":"y=330",
                        "font":{"family":"Times", "size":50},
                        "fill":{"color":"red"},
                        "location":(50,330) } )
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

