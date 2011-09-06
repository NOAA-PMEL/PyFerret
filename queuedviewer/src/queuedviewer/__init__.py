
from multiprocessing import JoinableQueue
from pyqtqueuedviewer import PyQtQueuedViewerProcess
import sys

class QueuedViewer(object):
    '''
    Creates and starts a QueuedViewer of one of the supported viewer types.
    Provides methods for interacting with the QueuedViewer.
    '''
    def __init__(self, viewertype):
        '''
        Create and start a QueuedViewer of one of the supported viewer types.
        The viewer will probably not be displayed until the { "type":"show" }
        command is submitted to the viewer queue using queueCommand.

        Currently supported viewer types are:
            "PyQt": PyQtQueuedViewer using PyQt4
        '''
        self.__jqueue = JoinableQueue()
        if viewertype == "PyQt":
            self.__vprocess = PyQtQueuedViewerProcess(self.__jqueue)
        else:
            raise TypeError("Unknown viewer type %s" % str(viewertype))
        self.__vprocess.start()
        self.__shutdown = False

    def queueCommand(self, cmnd):
        '''
        Submit the command cmnd to the queue to the viewer.
        '''
        self.__jqueue.put(cmnd)

    def waitForViewerExit(self):
        '''
        Wait for all the queued commands to be consumed and the the viewer
        to return.  The command { "type":"exit" } should have been the last
        command submitted to the viewer queue before calling this method.
        '''
        # close the queue to reject any calls to queueCommand
        self.__jqueue.close()
        self.__jqueue.join()
        self.__vprocess.join()

    def getViewerExitCode(self):
        return self.__vprocess.exitcode


if __name__ == "__main__":
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
    drawcmnds.append( { "type":"show" } )
    drawcmnds.append( { "type":"polygon", "points":squarepts,
                        "fill":{"color":"pink", "alpha":32},
                        "outline":{"color":"black", "alpha":32} } )
    drawcmnds.append( { "type":"text", "text":"Bye", 
                        "font":{"family":"Times", "size":42, "italic": True},
                        "fill":{"color":0x880000}, 
                        "location":(55,60) } )
    drawcmnds.append( { "type":"show" } )
    drawcmnds.append( { "type":"resizeScene" } )
    drawcmnds.append( { "type":"resizeViewer"})
    drawcmnds.append( {"type":"exit"} )

    qviewer = QueuedViewer("PyQt")
    for cmd in drawcmnds:
        raw_input("Press Enter to submit next command (type: %s)" % cmd["type"])
        qviewer.queueCommand(cmd)
    qviewer.waitForViewerExit()
    result = qviewer.getViewerExitCode()
    if result != 0:
        sys.exit(result)
    else:
        print "Done"
