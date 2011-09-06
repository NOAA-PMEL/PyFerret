import sip
sip.setapi('QVariant', 2)

from PyQt4.QtCore import QPointF, QTimer
from PyQt4.QtGui import QAction, QApplication, QBrush, QDialog, QFileDialog, \
                        QFont, QGraphicsScene, QGraphicsView, QImage, \
                        QMainWindow, QPen, QPainter, QPolygonF, QPushButton
from pyqtqueuecmndhelper import PyQtQueueCmndHelper
from multiprocessing import Process
from Queue import Empty
import math
import sys


class PyQtQueuedViewer(QMainWindow):
    '''
    A PyQt graphics viewer that receives generic drawing commands through a queue.

    A drawing command is a dictionary with string keys that will be interpreted
    into the appropriate PyQt command(s).  For example,
      { "type":"text", "text":"Hello", 
        "font":{"family":"Times", "size":36, "italic":True},
        "fill":{color:"darkred"}, "outline":{color:"black"},
        "location":(25,35) }

    The command { "type":"exit" } will shutdown the viewer and is the only way
    the viewer can be closed.  GUI actions can only hide the viewer.    
    '''

    def __init__(self, cmndQueue):
        '''
        Create a PyQt viewer with with given command queue.
        '''
        QMainWindow.__init__(self)
        self.__queue = cmndQueue
        self.__scene = QGraphicsScene(self)
        self.__view = QGraphicsView(self.__scene, self)
        self.setCentralWidget(self.__view)
        self.createActions()
        self.createMenus()
        self.__lastfilename = ""
        self.__shuttingdown = False
        self.__timer = QTimer(self)
        self.__timer.timeout.connect(self.checkCommandQueue)
        self.__timer.setInterval(0)
        self.__timer.start()

    def createActions(self):
        '''
        Create the actions used by the menus in this viewer.  Ownership
        of the actions are not transferred in addAction, thus the need
        to maintain references here.
        '''
        self.__saveAct = QAction("&Save", self, shortcut="Ctrl+S",
                                 statusTip="Save the current scene",
                                 triggered=self.inquireSaveFilename)
        self.__refreshAct = QAction("Re&fresh", self, shortcut="Ctrl+F",
                                    statusTip="Refresh the current scene",
                                    triggered=self.refreshScene)
        self.__resizeAct = QAction("&Resize Scene", self, shortcut="Ctrl+R",
                                   statusTip="Resize the underlying scene",
                                   triggered=self.inquireResizeScene)
        self.__hideAct = QAction("&Hide", self, shortcut="Ctrl+H",
                                 statusTip="Hide the viewer",
                                 triggered=self.hide)

    def createMenus(self):
        '''
        Create the menu items for the viewer 
        using the previously created actions.
        '''
        sceneMenu = self.menuBar().addMenu("&Scene")
        sceneMenu.addAction(self.__saveAct)
        sceneMenu.addAction(self.__refreshAct)
        sceneMenu.addAction(self.__resizeAct)
        sceneMenu.addAction(self.__hideAct)

    def closeEvent(self, event):
        '''
        Override so the viewer cannot be closed from user GUI actions;
        instead only hide the window.  The viewer can only be closed
        by sending the {"type":"exit"} command to the queue.
        '''
        if self.__shuttingdown:
            event.accept()
        else:
            event.ignore()
            self.hide()

    def exitViewer(self):
        '''
        Close and exit the viewer
        '''
        self.__timer.stop()
        self.__shuttingdown = True
        self.close()

    def refreshScene(self):
        '''
        Refresh the current scene
        '''
        self.__scene.update(self.__scene.sceneRect())

    def inquireResizeScene(self):
        '''
        Prompt the user for the desired origina and size 
        of the underlying scene.
        
        Short-circuited to call resizeScene(0,0,400,400)
        '''
        self.resizeScene(0, 0, 400, 400)

    def resizeScene(self, x, y, width, height):
        '''
        Resize the underlying graphics scene to the given width and 
        height, with the upper left corner located at the coordinate
        (x, y).  If width or height is less than one, the upper-left
        corner will be set to (0, 0) and the scene size will be set
        to the size of the bounding rectangle of all the currently
        drawn items. 
        '''
        if (width < 1.0) or (height < 1.0):
            self.__scene.setSceneRect(self.__scene.itemsBoundingRect())
        else:
            self.__scene.setSceneRect(x, y, width, height)

    def resizeViewer(self, width, height):
        '''
        Resize the viewer (the QMainWindow) to the given width and
        height.  If width or height is less than one, the viewer is
        resized to slightly larger than the required width and height
        to show all the underlying graphics scene.
        '''
        if (width < 1.0) or (height < 1.0):
            scenerect = self.__scene.sceneRect()
            menuheight = self.menuBar().size().height()
            vwidth = int(scenerect.width() + 0.5 * menuheight)
            vheight = int(scenerect.height() + 1.5 * menuheight)
        else:
            vwidth = int(math.ceil(width) + 0.25)
            vheight = int(math.ceil(height) + 0.25)
        self.resize(vwidth, vheight)

    def inquireSaveFilename(self):
        '''
        Prompt the user for the name of the file into which to save the scene.
        The file format will be determined from the filename extension.
        '''
        formatTypes = "Portable Networks Graphics (*.png);;" \
                      "Joint Photographic Experts Group (*.jpeg *.jpg *.jpe);;" \
                      "Tagged Image File Format (*.tiff *.tif);;" \
                      "Portable Pixmap (*.ppm);;" \
                      "X11 Pixmap (*.xpm);;" \
                      "X11 Bitmap (*.xbm);;" \
                      "Windows Bitmap (*.bmp)"
        fileName = QFileDialog.getSaveFileName(self, "Save the current scene as ", 
                               self.__lastfilename, formatTypes)
        # TODO: Linux dialogs do not append/modify the extension.  Need to get the format from the dialog.
        if fileName:
            self.saveSceneToFile(fileName)
            self.__lastfilename = fileName

    def saveSceneToFile(self, fileName):
        '''
        Save the current scene to the named file.  The format is guessed 
        from the filename extension.
        '''
        sceneRect = self.__scene.sceneRect()
        image = QImage(sceneRect.width(), sceneRect.height(), QImage.Format_ARGB32)
        # Initialize the image by filling it with transparent white
        image.fill(0x00FFFFFF)
        painter = QPainter(image)
        self.__scene.render(painter)
        painter.end()
        image.save(fileName)

    def checkCommandQueue(self):
        '''
        Check the queue for a command.  If there are any, get and 
        perform one command only, then return.
        '''
        try:
            cmnd = self.__queue.get_nowait()
            self.processCommand(cmnd)
            self.__queue.task_done()
        except Empty:
            pass

    def processCommand(self, cmnd):
        '''
        Examine the command type of cmnd and call the appropriate 
        method to deal with this command.
        '''
        cmndtype = cmnd["type"]
        if cmndtype == "exit":
            self.exitViewer()
        elif cmndtype == "hide":
            self.hide()
        elif cmndtype == "polygon":
            self.addPolygon(cmnd)
        elif cmndtype == "refresh":
            self.refreshScene()
        elif cmndtype == "resizeScene":
            myrect = PyQtQueueCmndHelper.getRect(cmnd)
            self.resizeScene(myrect.x(), myrect.y(),
                             myrect.width(), myrect.height())
        elif cmndtype == "resizeViewer":
            myrect = PyQtQueueCmndHelper.getRect(cmnd)
            self.resizeViewer(myrect.width(), myrect.height())
        elif cmndtype == "save":
            self.saveScene()
        elif cmndtype == "text":
            self.addSimpleText(cmnd)
        elif cmndtype == "show":
            self.show()
        else:
            raise ValueError("Unknown command cmndtype %s" % str(cmndtype))

    def saveScene(self, cmnd):
        '''
        Save the current scene to file.  Raises a KeyError if the
        "filename" key is not given.  The file format is guessed
        from the filename extension.
        '''
        fileName = cmnd["filename"]
        self.saveSceneToFile(fileName)
        
    def addSimpleText(self, cmnd):
        '''
        Add a "simple" text item to the viewer.  Raises a KeyError if the
        "text" key is not given.

        Recognized keys from cmnd:
            "text": string to displayed
            "font": dictionary describing the font to use; 
                    see PyQtQueueCmndHelper.getFontFromCommand
            "fill": dictionary describing the brush used to draw the text; 
                    see PyQtQueueCmndHelper.getBrushFromCommand
            "outline": dictionary describing the pen used to outline the text; 
                       see PyQtQueueCmndHelper.getPenFromCommand
            "location": (x,y) location for the start of text in pixels 
                        from the upper left corner of the scene
        '''
        try:
            myfont = PyQtQueueCmndHelper.getFontFromCommand(cmnd, "font")
        except KeyError:
            myfont = QFont()
        mytext = self.__scene.addSimpleText(cmnd["text"], myfont)
        try:
            mybrush = PyQtQueueCmndHelper.getBrushFromCommand(cmnd, "fill")
            mytext.setBrush(mybrush)
        except KeyError:
            pass
        try:
            mypen = PyQtQueueCmndHelper.getPenFromCommand(cmnd, "outline")
            mytext.setPen(mypen)
        except KeyError:
            pass
        try:
            (x, y) = cmnd["location"] 
            mytext.translate(x, y)
        except KeyError:
            pass

    def addPolygon(self, cmnd):
        '''
        Adds a polygon item to the viewer.  Raises a KeyError if the "points"
        key is not given.

        Recognized keys from cmnd:
            "points": the vertices of the polygon as a list of (x,y) points
                      of pixel values from the upper left corner of the scene
            "fill": dictionary describing the brush used to fill the polygon; 
                    see PyQtQueueCmndHelper.getBrushFromCommand
            "outline": dictionary describing the pen used to outline the polygon; 
                       see PyQtQueueCmndHelper.getPenFromCommand
            "offset": (x,y) offset, in pixels from the upper left corner of 
                      the scene, for the polygon
        '''
        mypoints = cmnd["points"]
        mypolygon = QPolygonF([ QPointF(x,y) for (x,y) in mypoints ])
        try:
            (x, y) = cmnd["offset"]
            mypolygon.translate(x, y)
        except KeyError:
            pass
        try:
            mypen = PyQtQueueCmndHelper.getPenFromCommand(cmnd, "outline")
        except KeyError:
            mypen = QPen()
        try:
            mybrush = PyQtQueueCmndHelper.getBrushFromCommand(cmnd, "fill")
        except KeyError:
            mybrush = QBrush()
        self.__scene.addPolygon(mypolygon, mypen, mybrush)


class PyQtQueuedViewerProcess(Process):
    def __init__(self, joinablequeue):
        Process.__init__(self)
        self.__queue = joinablequeue

    def run(self):
        self.__app = QApplication(sys.argv)
        self.__viewer = PyQtQueuedViewer(self.__queue)
        result = self.__app.exec_()
        self.__queue.close()
        self.__queue.join()
        SystemExit(result)


class _PyQtCommandQueuer(QDialog):
    '''
    Testing dialog for controlling the addition of commands to a queue.
    Used for testing PyQtQueuedViewer in the same process as the viewer. 
    '''
    def __init__(self, parent, queue, cmndlist):
        QDialog.__init__(self, parent)
        self.__cmndlist = cmndlist
        self.__queue = queue
        self.__nextcmnd = 0
        self.__button = QPushButton("Queue next command", self)
        self.__button.pressed.connect(self.queueNextCommand)
        self.show()

    def queueNextCommand(self):
        try:
            self.__queue.put(self.__cmndlist[self.__nextcmnd])
            self.__nextcmnd += 1
        except IndexError:
            self.__queue.close()
            self.__queue.join()
            self.close()


if __name__ == "__main__":
    from Queue import Queue
    
    app = QApplication(sys.argv)
    cmndqueue = Queue()
    viewer = PyQtQueuedViewer(cmndqueue)
    drawcmnds = []
    squarepts = ( (0, 0), (0, 200), (200, 200), (200, 0), (0, 0) )
    pentagonpts = ( ( 80.90,   0.00), (  0.00,  58.78), 
                   ( 30.90, 153.88), (130.90, 153.88), 
                   (161.80,  58.78), ( 80.90,   0.00) )
    drawcmnds.append( { "type":"show" } )
    drawcmnds.append( { "type":"resizeScene",
                        "x":-10,
                        "y":-10,
                        "width":300,
                        "height":300} )
    drawcmnds.append( { "type":"resizeViewer",
                        "width":400,
                        "height":400} )
    drawcmnds.append( { "type":"polygon", "points":squarepts,
                        "fill":{"color":"black", "alpha":32},
                        "outline":{"color":"black", "alpha":32} } )
    drawcmnds.append( { "type":"polygon", "points":pentagonpts, 
                        "fill":{"color":"lightblue"},
                        "outline":{"color":"black", "width": 5, "style":"dash"},
                        "offset":(24,24) } )
    drawcmnds.append( { "type":"polygon", "points":squarepts,
                        "fill":{"color":"pink", "alpha":32},
                        "outline":{"color":"black", "alpha":32} } )
    drawcmnds.append( { "type":"text", "text":"Bye", 
                        "font":{"family":"Times", "size":42, "italic": True},
                        "fill":{"color":0x880000}, 
                        "location":(55,60) } )
    drawcmnds.append( { "type":"resizeScene" } )
    drawcmnds.append( { "type":"resizeViewer"})
    drawcmnds.append( {"type":"exit"} )
    tester = _PyQtCommandQueuer(viewer, cmndqueue, drawcmnds)
    tester.show()
    result = app.exec_()
    if result != 0:
        sys.exit(result)
