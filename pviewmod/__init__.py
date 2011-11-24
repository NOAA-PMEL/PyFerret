'''
Class to create, send commands, and shutdown PipedViewers.
Currently, the only known viewer type is "PyQtPipedViewer".
'''

from multiprocessing import Pipe
import sys

class PipedViewer(object):
    '''
    Creates and starts a PipedViewer of one of the supported
    viewer types.  Provides methods for interacting with the
    PipedViewer.
    '''
    def __init__(self, viewertype):
        '''
        Create and start a PipedViewer of one of the supported
        viewer types.  The viewer will probably not be displayed
        until the { "action":"show" } command is submitted to
        the viewer cmndpipe using submitCommand.

        Currently supported viewer types are:
            "PyQtPipedViewer": PyQtPipedViewer using PyQt4
        '''
        (self.__cmndrecvpipe, self.__cmndsendpipe) = Pipe(False)
        (self.__rspdrecvpipe, self.__rspdsendpipe) = Pipe(False)
        if viewertype == "PyQtPipedViewer":
            try:
                from pyqtpipedviewer import PyQtPipedViewerProcess
            except ImportError:
                raise TypeError("The PyQt viewer requires PyQt4")
            self.__vprocess = PyQtPipedViewerProcess(self.__cmndrecvpipe,
                                                     self.__rspdsendpipe)
        else:
            raise TypeError("Unknown viewer type %s" % str(viewertype))
        self.__vprocess.start()
        self.__shutdown = False

    def submitCommand(self, cmnd):
        '''
        Submit the command cmnd to the command pipe for the viewer.
        '''
        self.__cmndsendpipe.send(cmnd)

    def checkForResponse(self, timeout = 0.0):
        '''
        Check for a reponse from the viewer.  The argument timeout
        (a number) is the maximum time in seconds to block (default:
        0.0; returns immediately).  If timeout is None, it will block
        until something is read.  Returns the response from the viewer,
        or None if there was no response in the allotted time.
        '''
        if self.__rspdrecvpipe.poll(timeout):
            response = self.__rspdrecvpipe.recv()
        else:
            response = None
        return response

    def waitForViewerExit(self):
        '''
        Wait for all the submitted commands to be consumed and the
        viewer  to return.  The command { "action":"exit" } should
        have been the last command submitted to the command pipe
        before calling this method.
        '''
        self.__cmndsendpipe.close()
        self.__rspdrecvpipe.close()
        self.__vprocess.join()

    def getViewerExitCode(self):
        return self.__vprocess.exitcode


if __name__ == "__main__":
    # vertices of a pentagon (roughly) centered in a 1000 x 1000 square
    pentagonpts = ( (504.5, 100.0), (100.0, 393.9),
                    (254.5, 869.4), (754.5, 869.4),
                    (909.0, 393.9),  )
    # create the list of commands to submit
    drawcmnds = []
    drawcmnds.append( { "action":"setTitle", "title":"Tester" } )
    drawcmnds.append( { "action":"show" } )
    drawcmnds.append( { "action":"clear", "color":"white", "alpha": 255})
    drawcmnds.append( { "action":"resize",
                        "width":5000,
                        "height":5000 } )
    drawcmnds.append( { "action":"beginView",
                        "viewfracs": {"left":0.0, "bottom":0.5,
                                      "right":0.5, "top":1.0 },
                        "usercoords": {"left":0, "bottom":0,
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
                        "viewfracs": {"left":0.05, "bottom":0.05,
                                      "right":0.95, "top":0.95 },
                        "usercoords": {"left":0, "bottom":0,
                                       "right":1000, "top":1000},
                        "clip":True } )
    drawcmnds.append( { "action":"drawMulticolorRectangle",
                        "left": 50, "bottom":50,
                        "right":950, "top":950,
                        "numrows":2, "numcols":3,
                        "colors":( {"color":0xFF0000, "alpha":128},
                                   {"color":0x888800, "alpha":128},
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
                        "viewfracs": {"left":0.0, "bottom":0.0,
                                      "right":1.0, "top":1.0 },
                        "usercoords": {"left":0, "bottom":0,
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
                                "width":8,
                                "style":"dash",
                                "capstyle":"round",
                                "joinstyle":"round"} } )
    drawcmnds.append( { "action":"endView" } )
    drawcmnds.append( { "action":"show" } )
    drawcmnds.append( { "action":"exit" } )

    # Test each known viewer.  Currently only PyQtPipedViewer is complete.
    for viewername in ( "PyQtPipedViewer", ):
        # create the viewer
        pviewer = PipedViewer(viewername)
        # submit the commands, pausing after each "show" command
        for cmd in drawcmnds:
            pviewer.submitCommand(cmd)
            response = pviewer.checkForResponse()
            while response:
                print "Response: %s" % str(response)
                response = pviewer.checkForResponse()
            if cmd["action"] == "show":
                raw_input("Press Enter to continue")
        # end of the commands - shut down and check return value
        pviewer.waitForViewerExit()
        result = pviewer.getViewerExitCode()
        if result != 0:
            sys.exit(result)
        else:
            print "Done with %s" % viewername

