'''
The PipedViewer class is used to create, send commands, and shutdown viewers 
in this module.  Currently, the only known viewer types are "PipedViewerPQ", 
and "PipedImagerPQ".

This package was developed by the Thermal Modeling and Analysis Project 
(TMAP) of the National Oceanographic and Atmospheric Administration's (NOAA) 
Pacific Marine Environmental Lab (PMEL).
'''

from __future__ import print_function

import sys

from multiprocessing import Pipe

class PipedViewer(object):
    '''
    Creates and starts a PipedViewer of one of the supported viewer 
    types.  Provides methods for interacting with the PipedViewer.
    '''
    def __init__(self, viewertype):
        '''
        Create and start a PipedViewer of one of the supported viewer 
        types.  The viewer will probably not be displayed until the 
        { "action":"show" } command is submitted to the viewer cmndpipe 
        using submitCommand.

        Currently supported viewer types are:
            "PipedViewerPQ": PipedViewerPQ using PyQt5 or PyQt4
            "PipedImagerPQ": PipedImagerPQ using PyQt5 or PyQt4
        '''
        super(PipedViewer, self).__init__()
        (self.__cmndrecvpipe, self.__cmndsendpipe) = Pipe(False)
        (self.__rspdrecvpipe, self.__rspdsendpipe) = Pipe(False)
        if viewertype == "PipedViewerPQ":
            try:
                from pipedviewer.pipedviewerpq import PipedViewerPQProcess
            except ImportError:
                raise TypeError("The PQ viewers requires PyQt5 or PyQt4")
            self.__vprocess = PipedViewerPQProcess(self.__cmndrecvpipe,
                                                   self.__rspdsendpipe)
        elif viewertype == "PipedImagerPQ":
            try:
                from pipedviewer.pipedimagerpq import PipedImagerPQProcess
            except ImportError:
                raise TypeError("The PQ viewers requires PyQt5 or PyQt4")
            self.__vprocess = PipedImagerPQProcess(self.__cmndrecvpipe,
                                                   self.__rspdsendpipe)
        else:
            raise TypeError("Unknown viewer type %s" % str(viewertype))
        self.__vprocess.start()

    def submitCommand(self, cmnd):
        '''
        Submit the command cmnd to the command pipe for the viewer.
        '''
        self.__cmndsendpipe.send(cmnd)

    def checkForResponse(self, timeout = 0.0):
        '''
        Check for a response from the viewer.  The argument timeout
        (a number) is the maximum time in seconds to block (default:
        0.0; returns immediately).  If timeout is None, it will block
        until something is read.  Returns the response from the viewer,
        or None if there was no response in the allotted time.
        '''
        if self.__rspdrecvpipe.poll(timeout):
            myresponse = self.__rspdrecvpipe.recv()
        else:
            myresponse = None
        return myresponse

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


def _testviewers():
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
    testannotations = ( "The 1<sup>st</sup> CO<sub>2</sub> annotations line",
                        "Another line with <i>lengthy</i> details that should " + \
                        "wrap to a 2<sup>nd</sup> annotation line",
                        "<b>Final</b> annotation line" )

    # Test each known viewer.
    for viewername in ( "PipedViewerPQ", "PipedImagerPQ" ):
        print("Testing Viewer %s" % viewername)
        # create the viewer
        pviewer = PipedViewer(viewername)
        mydrawcmnds = drawcmnds[:]
        mydrawcmnds.append( { "action":"save",
                              "filename":viewername + "_test.pdf",
                              "vectsize":{"width":7.0, "height":7.0},
                              "rastsize":{"width":750, "height":750},
                              "annotations":testannotations } )
        mydrawcmnds.append( { "action":"save",
                              "filename":viewername + "_test.png",
                              "vectsize":{"width":7.0, "height":7.0},
                              "rastsize":{"width":750, "height":750},
                              "annotations":testannotations } )
        mydrawcmnds.append( { "action":"exit" } )
        
        # submit the commands, pausing after each "show" command
        for cmd in mydrawcmnds:
            print("Command: %s" % str(cmd))
            pviewer.submitCommand(cmd)
            response = pviewer.checkForResponse()
            while response:
                print("Response: %s" % str(response))
                response = pviewer.checkForResponse()
            if cmd["action"] == "show":
                if sys.version_info[0] > 2:
                    input("Press Enter to continue")
                else:
                    raw_input("Press Enter to continue")
        # end of the commands - shut down and check return value
        pviewer.waitForViewerExit()
        result = pviewer.getViewerExitCode()
        if result != 0:
            sys.exit(result)
        else:
            print("Done with %s" % viewername)

if __name__ == "__main__":
    _testviewers()
