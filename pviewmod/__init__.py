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

import multiprocessing
import threading

WINDOW_CLOSED_MESSAGE = 'Window was closed'

class ErrorMonitor(threading.Thread):
    '''
    Thread to monitor unexpected responses (error messages) sent by 
    a PipedViewer process.  When a response from the PipedViewer is 
    detected, a lock paired with this response pipe is acquired, the 
    response (if still present) is read, and the lock is released. 
    Any error message read is appropriately dealt with and monitoring 
    of the response pipe resumes.
    '''

    def __init__(self, responsepipe, readlock, vprocess):
        '''
        Monitor unexpected responses (error message) from the given 
        response pipe for the PipedViewer.  Prior to reading a response 
        message, the given lock is acquired.  After reading the message, 
        if any was still present, the lock is released, and any error 
        message is handled appropriately.  Exits when the PipedViewer
        process is no longer alive.

        Arguments:
            responsepipe: (multiprocessing.Pipe) provides responses sent by the PipedViewer
            readlock: (threading.RLock) lock associated with the above response pipe
            vprocess: (multiprocessing.Process) PipedViewer process
        '''
        super(ErrorMonitor, self).__init__(group=None, target=None, name='ErrorMonitor')
        self.__responsepipe = responsepipe
        self.__readlock = readlock
        self.__vprocess = vprocess

    def run(self):
        '''
        Monitor unexpected responses (error message) from the PipedViewer 
        response pipe and read lock associated with this instance.  Prior 
        to reading a response message, the lock is acquired.  After reading 
        the message, if any was still present, the lock is released, and 
        any error message is handled appropriately.  Exits when the reponse 
        pipe is closed (when multiprocessing.Pipe.poll raises an Exception).
        '''
        while True:
            try:
                # wait for something to appear or the PipedViewer to exit
                while not self.__responsepipe.poll(0.1):
                    if not self.__vprocess.is_alive():
                        sys.exit(0)
            except Exception:
                # response pipe was closed - quit
                break
            # read the message, if still there, only after acquiring the read lock
            fullerrmsg = ''
            self.__readlock.acquire()
            try:
                try:
                    while self.__responsepipe.poll():
                        errmsg = self.__responsepipe.recv()
                        if fullerrmsg:
                            fullerrmsg += '\n'
                        fullerrmsg += str(errmsg)
                except Exception:
                    # response pipe was closed - deal with anything already read;
                    # next loop through will break and quit
                    pass
            finally:
                self.__readlock.release()
            # deal with the error message, if any
            if fullerrmsg == WINDOW_CLOSED_MESSAGE:
                # TODO: tell the ferret engine that the window was closed 
                #       and maybe don't print anything
                print('\n', file=sys.stderr)
                print(fullerrmsg, file=sys.stderr)
            elif fullerrmsg:
                print('\n', file=sys.stderr)
                print(fullerrmsg, file=sys.stderr)
        sys.exit(0)


class PipedViewer(object):
    '''
    Creates and starts a PipedViewer of one of the supported viewer 
    types.  Provides methods for interacting with the PipedViewer.
    '''
    def __init__(self, viewertype):
        '''
        Create and start a PipedViewer of one of the supported viewer 
        types.  The viewer will probably not be displayed until the 
        {"action":"show"} command is submitted to the viewer.

        Currently supported viewer types are:
            "PipedViewerPQ": PipedViewerPQ using PyQt5 or PyQt4
            "PipedImagerPQ": PipedImagerPQ using PyQt5 or PyQt4
        '''
        super(PipedViewer, self).__init__()
        (self.__cmndrecvpipe, self.__cmndsendpipe) = multiprocessing.Pipe(False)
        (self.__rspdrecvpipe, self.__rspdsendpipe) = multiprocessing.Pipe(False)
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
        self.__rspdrecvlock = threading.RLock()
        self.__errmonitor = ErrorMonitor(self.__rspdrecvpipe, self.__rspdrecvlock, self.__vprocess)
        self.__errmonitor.start()

    def blockErrMonitor(self):
        '''
        Block monitoring of error messages sent from the PipedViewer. 

        Call this method prior to submitting a command which is expected 
        to respond with desired information so this response will not be 
        mistaken as an error message.  After reading the response, call 
        the "resumeErrMonitor" method. 
        '''
        self.__rspdrecvlock.acquire()

    def resumeErrMonitor(self):
        '''
        Resume monitoring of error messages sent from the PipedViewer. 

        Only call this method after calling "blockErrMonitor", which 
        presumably was called prior to submitting a command expected 
        to respond with desired information and reading the response.
        '''
        self.__rspdrecvlock.release()

    def submitCommand(self, cmnd):
        '''
        Submit the command cmnd to the command pipe for the viewer.

        If a response is expected from the command, the "blockErrMonitor" 
        method should be called prior to submitting the command, and 
        the "resumeErrMonitor" method should be called after reading 
        the response.  This prevent the response from being mistaken 
        as an error message.
        '''
        self.__cmndsendpipe.send(cmnd)

    def checkForResponse(self, timeout = 0.0):
        '''
        Check for a response from the viewer.  

        Arguments: 
            timeout: (number) maximum time in seconds to wait for a 
                     response to appear.  Zero (default) does not wait;
                     None waits indefinitely.
        Returns: 
            the response from the viewer, or None if there was no 
            response in the allotted time.
        Raises: 
            IOError: if the response pipe is closed while waiting 
                     for a response to appear.
        '''
        if self.__rspdrecvpipe.poll(timeout):
            myresponse = self.__rspdrecvpipe.recv()
        else:
            myresponse = None
        return myresponse

    def shutdownViewer(self):
        '''
        If the PipedViewer is still alive, submits the command {"action":"exit"}
        and reads any responses.  Waits for the viewer and the error montior to exit.  

        Returns: 
            any responses from the viewer, or 
            an empty string if there was no response (viewer was not alive).
        '''
        closingremarks = ''
        if self.__vprocess.is_alive():
            # block the error monitor
            self.__rspdrecvlock.acquire()
            self.__cmndsendpipe.send({'action':'exit'})
            self.__cmndsendpipe.close()
            try:
                # Read everything from the response pipe until it is closed on the viewer side
                if closingremarks:
                    closingremarks += '\n'
                closingremarks += self.__rspdrecvpipe.recv()
            except Exception:
                pass
            self.__rspdrecvpipe.close()
            self.__rspdrecvlock.release()
        self.__vprocess.join()
        self.__errmonitor.join()
        return closingremarks

    def getViewerExitCode(self):
        '''
        Returns: 
            the viewer process exit code.
        '''
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
                        "symbol":"bararrow",
                        "size":20,
                        "color":"magenta" })
    drawcmnds.append( { "action":"drawPoints",
                        "points":( (400,  50),
                                   (400, 150),
                                   (400, 250),
                                   (400, 350),
                                   (400, 450) ),
                        "symbol":"^",
                        "size":20,
                        "color":"magenta" })
    drawcmnds.append( { "action":"drawPoints",
                        "points":( (450,  50),
                                   (450, 150),
                                   (450, 250),
                                   (450, 350),
                                   (450, 450) ),
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
                        "Another line with <i>lengthy</i> details that go on and on " + \
                        "and on and should wrap to a 2<sup>nd</sup> annotation line",
                        "<b>Final</b> annotation line" )

    # Test each known viewer.
    for viewername in ( "PipedViewerPQ", ):
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

        # submit the commands, pausing after each "show" command
        for cmd in mydrawcmnds:
            print("Command: %s" % str(cmd))
            pviewer.submitCommand(cmd)
            if cmd["action"] == "show":
                if sys.version_info[0] > 2:
                    input("Press Enter to continue")
                else:
                    raw_input("Press Enter to continue")
        # end of the commands - shut down and check return value
        response = pviewer.shutdownViewer()
        if response:
            print("Closing remarks: %s" % str(response))
        result = pviewer.getViewerExitCode()
        if result != 0:
            print("Exit code %i from %s" % (result, viewername))
        else:
            print("Done with %s" % viewername)

if __name__ == "__main__":
    _testviewers()
