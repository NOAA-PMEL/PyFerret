'''
'''
import pygtk
pygtk.require('2.0')
import gtk
import gtk.gdk
import gobject
import sys
import time
import math

from pygtkcmndhelper import PyGtkCmndHelper, RectF, SimpleTransform
from pygtkscaledialog import PyGtkScaleDialog
from multiprocessing import Pipe, Process

class PyGtkPipedImager(gtk.Window):
    '''
    '''

    def __init__(self, cmndpipe, rspdpipe):
        '''
        Create a PyGtk viewer which reads commands from the Pipe
        cmndpipe and writes responses back to rspdpipe.
        '''
        super(PyGtkPipedImager, self).__init__(gtk.WINDOW_TOPLEVEL)
        self.__cmndpipe = cmndpipe
        self.__rspdpipe = rspdpipe
        # root window for creating the pixmaps
        self.__rootwindow = gtk.gdk.get_default_root_window()
        # colormap for drawings
        self.__drawcolormap = gtk.gdk.colormap_get_system()
        # allocated color for clearing the scene
        self.__lastclearcolor = self.__drawcolormap.alloc_color(
                                     gtk.gdk.Color(65535, 65535, 65535) )
        # default scene size = sqrt(0.75) * (1280, 1024)
        self.__scenewidth = 1110
        self.__sceneheight = 890
        # minimum size of the scene and the scaled scene
        self.__minsize = 128
        # Pixmap containing the scene
        self.__scenepixmap = None
        # scaling (zoom) factor for creating the displayed scene
        self.__scalefactor = 1.0
        # scaled pixmap size
        self.__scaledwidth = self.__scenewidth
        self.__scaledheight = self.__sceneheight
        # Pixmap containing the scaled scene for exposure events.
        # This is set to None when new content is added to the scene Pixmap
        self.__scaledpixmap = None
        # simple transformation object for the current view
        self.__activetrans = None
        # clipping rectangle for the current view
        self.__cliprect = None
        # data for recreating the current view
        self.__fracsides = None
        self.__usersides = None
        self.__clipit = True
        # maximum user Y coordinate - used by adjustPoint
        self.__userymax = 1.0
        # vertical box that will be the content of this top-level window
        vertbox = gtk.VBox(False, 2)
        self.add(vertbox)
        # menubar as the top widget
        menubar = self.createMenuBar()
        vertbox.pack_start(menubar, False, False, 0)
        # DrawingArea in a scrolled window as the central widget
        # TODO: comment out the next line - just there for pydev in eclipse
        self.__scenedrawarea = None
        (self.__scenedrawarea, scrolledwindow) = self.createScrolledDrawArea()
        vertbox.pack_start(scrolledwindow, True, True, 0)
        # statusbar as the bottom widget
        self.__statusbar = self.createStatusBar()
        vertbox.pack_start(self.__statusbar, False, False, 0)
        # context id for messages in the status bar
        self.__infostatus = self.__statusbar.get_context_id("INFO")
        # vertical box now complete - show it and everything in it
        vertbox.show_all()
        # command helper object
        self.__helper = PyGtkCmndHelper(self, self.__drawcolormap)
        # last save filename
        self.__lastfilename = ""
        # flag for allowing a shutdown
        self.__shuttingdown = False
        # Set the initial size of the viewer
        menuheight = menubar.size_request()[1]
        statusheight = self.__statusbar.size_request()[1]
        mwwidth = self.__scenewidth + 8
        mwheight = self.__sceneheight + menuheight + statusheight + 12
        self.set_default_size(mwwidth, mwheight)
        # initialize the scene pixmap, scaled values, drawing area size
        self.initializeScene()
        # update the draw area from the scene pixmap on exposures
        self.__scenedrawarea.connect("expose_event", self.updateDrawArea, None)
        # return button press locations
        self.__scenedrawarea.connect("button_press_event", self.buttonPressMonitor, None)
        # mark the events to be tracked by the drawing area
        self.__scenedrawarea.set_events(gtk.gdk.EXPOSURE_MASK | gtk.gdk.BUTTON_PRESS_MASK)
        # setup the delete event handler (request window be destroyed)
        self.connect("delete_event", self.deleteEventMonitor, None)
        # setup the destroy callback (window is being destroyed)
        self.connect("destroy", self.exitViewer, None)
        # check the command pipe for new commands when no other events
        self.__idleid = gobject.idle_add(self.checkCommandPipe)

    def createMenuBar(self):
        '''
        Creates and returns a menubar for this viewer.
        '''
        # Scene menu - not shown until needed
        scenemenu = gtk.Menu()
        # Scene -> Save
        saveitem = gtk.MenuItem("_Save")
        saveitem.connect("activate", self.inquireSaveFile, None)
        scenemenu.append(saveitem)
        saveitem.show()
        # Scene -> Scale
        scaleitem = gtk.MenuItem("Sc_ale")
        scaleitem.connect("activate", self.inquireScale, None)
        scenemenu.append(scaleitem)
        scaleitem.show()
        # Scene -> Refresh
        refreshitem = gtk.MenuItem("_Refresh")
        refreshitem.connect("activate", self.refreshDrawArea, None)
        scenemenu.append(refreshitem)
        refreshitem.show()
        # Scene separator
        separator = gtk.SeparatorMenuItem()
        scenemenu.append(separator)
        separator.show()
        # Scene -> Hide
        hideitem = gtk.MenuItem("_Hide")
        hideitem.connect("activate", self.hideViewer, None)
        scenemenu.append(hideitem)
        hideitem.show()
        # Help menu - not shown until needed
        helpmenu = gtk.Menu()
        # Help -> About
        aboutitem = gtk.MenuItem("About")
        aboutitem.connect("activate", self.showAbout, None)
        helpmenu.append(aboutitem)
        aboutitem.show()
        # Help separator
        separator = gtk.SeparatorMenuItem()
        helpmenu.append(separator)
        separator.show()
        # Help -> Exit
        exititem = gtk.MenuItem("Exit")
        exititem.connect("activate", self.exitViewer, None)
        helpmenu.append(exititem)
        exititem.show()
        # Menubar
        menubar = gtk.MenuBar()
        # Scene item on the menubar
        sceneitem = gtk.MenuItem("_Scene")
        sceneitem.set_submenu(scenemenu)
        menubar.append(sceneitem)
        # Help item on the menubar
        helpitem = gtk.MenuItem("Help")
        helpitem.set_submenu(helpmenu)
        menubar.append(helpitem)
        # Return the completed, but not shown, menubar
        return menubar

    def createScrolledDrawArea(self):
        '''
        Creates a drawing area inside a scrolled window.
        Returns (drawingarea, scrolledwindow) widget tuple.
        '''
        # drawing area
        drawingarea = gtk.DrawingArea()
        # scrolled window containing the drawing area
        scrolledwindow = gtk.ScrolledWindow()
        scrolledwindow.set_policy(gtk.POLICY_AUTOMATIC, gtk.POLICY_AUTOMATIC)
        scrolledwindow.add_with_viewport(drawingarea)
        drawingarea.show()
        # Return the completed, but not shown scrolled window and drawing area
        return (drawingarea, scrolledwindow)

    def createStatusBar(self):
        '''
        Creates and return a status bar for the viewer.
        '''
        statusbar = gtk.Statusbar()
        statusbar.set_has_resize_grip(True)
        return statusbar

    def initializeScene(self):
        '''
        Create and clear the scene pixmap
        '''
        # Create the scene pixmap from the recorded width and height.
        self.__scenepixmap = gtk.gdk.Pixmap(self.__rootwindow,
                                     self.__scenewidth, self.__sceneheight, -1)
        # get rid of any scaled pixmap
        self.__scaledpixmap = None
        # but assign the size of the scaled pixmap when created
        self.__scaledwidth = int(self.__scenewidth * self.__scalefactor + 0.5)
        self.__scaledheight = int(self.__sceneheight * self.__scalefactor + 0.5)
        # set the size of the (scaled scene) drawing area
        self.__scenedrawarea.set_size_request(self.__scaledwidth,
                                              self.__scaledheight)
        # clear the scene and queue a redraw of the entire drawing area
        self.clearScene(None)
        # restart any active view with the new scene size
        if self.__activetrans != None:
            self.beginViewFromSides(self.__fracsides, self.__usersides, self.__clipit)

    def runViewer(self):
        '''
        Runs the viewer.  Does not return until the viewer
        has ended (exited).
        
        This just calls the gtk.main function to enter into
        the gtk processing loop until gtk.main_quit is called.
        '''
        gtk.main()

    def refreshDrawArea(self, widget, data):
        '''
        Callback for refreshing the scene draw area
        '''
        self.__scenedrawarea.queue_draw_area(0, 0, self.__scaledwidth, self.__scaledheight)

    def hideViewer(self, widget, data):
        '''
        Callback for hiding the viewer
        '''
        self.hide()

    def deleteEventMonitor(self, widget, event, data):
        '''
        Handler for delete events; notably, close events
        from pressing the 'X' button on window frame.
        '''
        if not self.__shuttingdown:
            # Not shutting down, so just hide the window
            self.hide()
            # done with this event
            return True
        # accept shutdown, so not done with this event
        # continue on the chain (delete, destroy)
        return False

    def exitViewer(self, widget, data):
        '''
        Callback for exiting the viewer.
        
        This calls the gtk.main_quit function to signal the
        gtk.main function to return.
        '''
        self.__shuttingdown = True
        gobject.source_remove(self.__idleid)
        self.destroy()
        gtk.main_quit()

    def showAbout(self, widget, data):
        '''
        Callback for displaying a message dialog about this program.
        '''
        # TODO:
        pass

    def updateDrawArea(self, widget, event, data):
        '''
        Update the exposed area of the displayed scene from the scene pixmap
        of scaled pixmap.  If the scene is scaled and the scaled pixmap is None,
        the scaled pixmap is created from the scene pixmap. 
        '''
        if self.__scalefactor == 1.0:
            # use the scene pixmap directly
            pixmap = self.__scenepixmap
        elif self.__scaledpixmap != None:
            # resuse the existing scaled pixmap
            pixmap = self.__scaledpixmap
        else:
            # create, then use, the scaled scene pixmap
            #
            # get a pixbuf of the scene
            pixbuf = gtk.gdk.Pixbuf(gtk.gdk.COLORSPACE_RGB, False, 8,
                                    self.__scenewidth, self.__sceneheight)
            pixbuf = pixbuf.get_from_drawable(self.__scenepixmap, self.__drawcolormap,
                            0, 0, 0, 0, self.__scenewidth, self.__sceneheight)
            # scale the pixbuf to the appropriate size
            pixbuf = pixbuf.scale_simple(self.__scaledwidth, self.__scaledheight,
                                         gtk.gdk.INTERP_BILINEAR)
            # create the scaled pixmap 
            pixmap = gtk.gdk.Pixmap(self.__scenedrawarea.window,
                                    self.__scaledwidth, self.__scaledheight, -1)
            # assign the scaled pixmap from the scaled pixbuf
            pixmap.draw_pixbuf(None, pixbuf, 0, 0, 0, 0, self.__scaledwidth,
                            self.__scaledheight, gtk.gdk.RGB_DITHER_NORMAL, 0, 0)
            # saved the scaled pixmap for reuse
            self.__scaledpixmap = pixmap
        # get the rectangle to copy from the event
        (exx, exy, exwidth, exheight) = event.area
        # get the gc for drawing a dark color in unassigned areas
        darkgc = self.__scenedrawarea.get_style().dark_gc[gtk.STATE_NORMAL]
        if (exx < self.__scaledwidth) and (exy < self.__scaledheight):
            # need some part of the pixmap
            pixwidth = min(exwidth, self.__scaledwidth - exx)
            pixheight = min(exheight, self.__scaledheight - exy)
            # copy from the (scaled) pixmap to the drawing area
            self.__scenedrawarea.window.draw_drawable(pixmap.new_gc(), pixmap,
                                    exx, exy, exx, exy, pixwidth, pixheight)
            if pixwidth < exwidth:
                # also need some background to the right of the scene
                self.__scenedrawarea.window.draw_rectangle(darkgc, True,
                                            self.__scaledwidth, exy,
                                            exwidth - pixwidth, exheight)
            if pixheight < exheight:
                # also need some background below the scene
                self.__scenedrawarea.window.draw_rectangle(darkgc, True,
                                            exx, self.__scaledheight,
                                            exwidth, exheight - pixheight)
        else:
            # everything is outside the pixmap
            self.__scenedrawarea.window.draw_rectangle(darkgc, True,
                                        exx, exy, exwidth, exheight)
        # not done with this event - continue on the chain to display
        return False

    def buttonPressMonitor(self, widget, event, data):
        '''
        Monitors button presses (in the drawing area) for reporting back locations. 
        '''
        # TODO:
        return False

    def inquireScale(self, widget, data):
        '''
        Prompt the user for the desired scaling factor for the scene.
        '''
        scaledialog = PyGtkScaleDialog("Scene Size Scaling",
                        "Scaling factor (both horiz. and vert.) for the scene",
                        self.__scalefactor, 
                        self.__scenewidth * self.__scalefactor,
                        self.__sceneheight * self.__scalefactor,
                        self.__minsize, self.__minsize, 
                        self.__helper.getDefaultScreenDpis(), self)
        while True:
            retval = scaledialog.run()
            if retval != gtk.RESPONSE_ACCEPT:
                okay = False
                break
            (newscale, okay) = scaledialog.getValues()
            if okay:
                break
            errdialog = gtk.MessageDialog(scaledialog, gtk.DIALOG_MODAL,
                                          gtk.MESSAGE_ERROR,
                                          (gtk.STOCK_OK, gtk.RESPONSE_ACCEPT),
                                          "Invalid scaling factor")
            errdialog.run()
            errdialog.destroy()
        scaledialog.destroy()
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
        # compute the new scaled width and height
        newscaledwidth = int(newfactor * self.__scenewidth + 0.5)
        newscaledheight = int(newfactor * self.__sceneheight + 0.5)
        # check they are larger to the minimum size
        if (newscaledwidth < self.__minsize) or (newscaledheight < self.__minsize):
            # Set to the minimum factor meeting the minimum size
            if self.__scenewidth <= self.__sceneheight:
                newfactor = float(self.__minsize) / float(self.__scenewidth)
            else:
                newfactor = float(self.__minsize) / float(self.__sceneheight)
            newscaledwidth = int(newfactor * self.__scenewidth + 0.5)
            newscaledheight = int(newfactor * self.__sceneheight + 0.5)
        # Check if the scaled scene size has actually changed
        if (newscaledwidth != self.__scaledwidth) or \
           (newscaledheight != self.__scaledheight):
            # Set the new scaling factor - the scene pixmap does not change
            self.__scalefactor = newfactor
            # get rid of any scaled pixmap
            self.__scaledpixmap = None
            # but assign the size of the scaled pixmap when created
            self.__scaledwidth = newscaledwidth
            self.__scaledheight = newscaledheight
            # set the size of the (scaled scene) drawing area
            self.__scenedrawarea.set_size_request(self.__scaledwidth,
                                                  self.__scaledheight)
            # queue a redraw of the entire scene (ignored if not visible)
            self.__scenedrawarea.queue_draw_area(0, 0, self.__scaledwidth,
                                                       self.__scaledheight)

    def inquireSaveFile(self, widget, data):
        '''
        Prompt the user for the name of the file into which to save the scene.
        '''
        # TODO:
        pass

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
        # TODO:
        pass

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
            # nothing to do (gtk immediately calling this method
            # again only for this method to immediately return).
            while self.__cmndpipe.poll(0.002):
                cmnd = self.__cmndpipe.recv()
                self.processCommand(cmnd)
                # Continue to try to process commands until
                # more than 50 milliseconds have passed.
                # This reduces overhead when there are lots
                # of commands waiting in the queue.
                if (time.clock() - starttime) > 0.050:
                    break
            # return True so this will be called again
            return True
        except Exception:
            # EOFError should never arise from recv since
            # the call is after poll returns True
            (exctype, excval) = sys.exc_info()[:2]
            if excval:
                self.__rspdpipe.send("**ERROR %s: %s" % (str(exctype), str(excval)))
            else:
                self.__rspdpipe.send("**ERROR %s" % str(exctype))
            self.exitViewer(None, None)
            # if it even gets here, return False to stop calling this method
            return False

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
            self.exitViewer(None, None)
        elif cmndact == "hide":
            self.hideViewer(None, None)
        elif cmndact == "dpi":
            dpis = self.__helper.getDefaultScreenDpis()
            self.__rspdpipe.send(dpis)
        elif cmndact == "update":
            self.__scenedrawarea.queue_draw_area(0, 0, self.__scaledwidth, self.__scaledheight)
        elif cmndact == "resize":
            mysize = self.__helper.getSizeFromCmnd(cmnd)
            self.resizeScene(mysize.width(), mysize.height())
        elif cmndact == "save":
            filename = cmnd["filename"]
            fileformat = cmnd.get("fileformat", None)
            transparentbkg = cmnd.get("transparentbkg", False)
            self.saveSceneToFile(filename, fileformat, transparentbkg)
        elif cmndact == "setTitle":
            self.set_title(cmnd["title"])
        elif cmndact == "show":
            self.show()
        elif cmndact == "beginView":
            self.beginView(cmnd)
        elif cmndact == "clipView":
            self.__clipit = bool( cmnd["clip"] )
        elif cmndact == "endView":
            self.__activetrans = None
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
            raise ValueError("Unknown command action %s" % str(cmndact) )

    def clearScene(self, colorinfo):
        '''
        Clear the contents of the scene pixmap with the color described
        in the dictionary colorinfo (see pygtkcmndhelper.getCoorFromCmnd),
        of the the last clearing color if colorinfo is None or the color 
        is not valid.

        If the scaled pixmap exists, it is also cleared with this same
        color (instead of having to recreated it by scaling the cleared
        scene).
        '''
        # get the color to use for clearing (the background color)
        if colorinfo:
            try :
                mycolor = self.__helper.getColorFromCmnd(colorinfo)
                self.__lastclearcolor = mycolor
            except KeyError:
                # No color given
                pass
            except ValueError:
                # Invalid color given
                pass
            except RuntimeError:
                # Unable to allocate the color
                pass
        # clear the scene by drawing a rectangle filled with the last
        # clearing color over the entire pixmap
        cleargc = self.__scenepixmap.new_gc(foreground=self.__lastclearcolor)
        self.__scenepixmap.draw_rectangle(cleargc, True, 0, 0,
                           self.__scenewidth, self.__sceneheight)
        # do the same with the scaled pixmap if it exists
        if self.__scaledpixmap != None:
            self.__scaledpixmap.draw_rectangle(cleargc, True, 0, 0,
                                self.__scaledwidth, self.__scaledheight)
        # queue a redraw of the entire scene (ignored if not visible)
        self.__scenedrawarea.queue_draw_area(0, 0, self.__scaledwidth,
                                                   self.__scaledheight)

    def resizeScene(self, width, height):
        '''
        Clear and resize the scene to the given width and height
        in units of 0.001 inches.
        '''
        dpis = self.__helper.getDefaultScreenDpis()
        newwidth = int(width * 0.001 * dpis[0] + 0.5)
        if newwidth < self.__minsize:
            newwidth = self.__minsize
        newheight = int(height * 0.001 * dpis[1] + 0.5)
        if newheight < self.__minsize:
            newheight = self.__minsize
        if (newwidth != self.__scenewidth) or (newheight != self.__sceneheight):
            # Set the new size of the scene
            self.__scenewidth = newwidth
            self.__sceneheight = newheight
            # Reset and clear the scene pixmap, scaled values, drawing area
            self.initializeScene()

    def beginView(self, cmnd):
        '''
        Setup a new viewport and window for drawing on a portion
        (possibly all) of the scene.  Recognized keys from cmnd
        are:
            "viewfracs": a dictionary of sides positions (see
                    PyGtkCmndHelper.getSidesFromCmnd) giving the
                    fractions [0.0, 1.0] of the way through the
                    scene for the sides of the new View.
            "usercoords": a dictionary of sides positions (see
                    PyGtkCmndHelper.getSidesFromCmnd) giving the
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
        # Get the location for the new view in terms of pixels.
        width = float(self.__scenewidth)
        height = float(self.__sceneheight)
        leftpixel = int( math.floor(fracsides.left() * width) )
        rightpixel = int( math.ceil(fracsides.right() * width) )
        bottompixel = int( math.floor(fracsides.bottom() * height) )
        toppixel = int( math.ceil(fracsides.top() * height) )
        # perform the checks after turning into units of pixels
        # to make sure the values are significantly different
        if (0 > leftpixel) or (leftpixel >= rightpixel) or (rightpixel > self.__scenewidth):
            raise ValueError( "Invalid left, right view fractions: " \
                              "left in pixels = %d, right in pixels = %d" % \
                              (leftpixel, rightpixel) )
        if (0 > bottompixel) or (bottompixel >= toppixel) or (toppixel > self.__sceneheight):
            raise ValueError( "Invalid bottom, top view fractions: " \
                              "bottom in pixels = %d, top in pixels = %d" % \
                              (bottompixel, toppixel) )
        # Create the view rectangle in device coordinates (as floats)
        vrectf = RectF(leftpixel, self.__sceneheight - toppixel,
                       rightpixel - leftpixel, toppixel - bottompixel)
        # Save the device coordinate rectangle (as integers) for clipping
        self.__cliprect = gtk.gdk.Rectangle(leftpixel, self.__sceneheight - toppixel,
                       rightpixel - leftpixel, toppixel - bottompixel)
        # Save whether clipping should be performed
        self.__clipit = clipit
        # Get the user coordinates for this view rectangle
        leftcoord = usersides.left()
        rightcoord = usersides.right()
        bottomcoord = usersides.bottom()
        topcoord = usersides.top()
        if leftcoord >= rightcoord:
            raise ValueError( "Invalid left, right user coordinates: " \
                              "left = %f, right = %f" % \
                              (leftcoord, rightcoord) )
        if bottomcoord >= topcoord:
            raise ValueError( "Invalid bottom, top user coordinates: " \
                              "bottom = %1, top = %2" % \
                              (bottomcoord, topcoord) )
        # Create the view rectangle in user (world) coordinates
        # adjustPoint will correct for the flipped, zero-based Y coordinate
        wrectf = RectF(leftcoord, 0.0, rightcoord - leftcoord, topcoord - bottomcoord)
        # Compute the entries in the transformation matrix
        sx = vrectf.width() / wrectf.width()
        sy = vrectf.height() / wrectf.height()
        dx = vrectf.left() - (sx * wrectf.left())
        dy = vrectf.top() - (sy * wrectf.top())
        self.__activetrans = SimpleTransform(sx, sy, dx, dy)
        # Save the current view sides and clipit setting for recreating the view.
        # Just save the original objects (assume calling functions do not keep them)
        self.__fracsides = fracsides
        self.__usersides = usersides
        self.__clipit = clipit
        # Pull out the top coordinate since this is used a lot (via adjustPoint)
        self.__userymax = usersides.top()

    def drawMultiline(self, cmnd):
        '''
        Draws a collection of connected line segments.

        Recognized keys from cmnd:
            "points": consecutive endpoints of the connected line
                    segments as a list of (x, y) coordinates
            "pen": dictionary describing the line used to draw the
                    segments (see PyGtkCmndHelper.setLineFromCmnd)

        The coordinates are user coordinates from the bottom left corner.

        Raises:
            KeyError if the "points" or "pen" key is not given
            ValueError if there are fewer than two endpoints given
        '''
        # Get the points in device coordinates
        ptcoords = cmnd["points"]
        if len(ptcoords) < 2:
            raise ValueError("fewer that two endpoints given")
        adjpts = [ self.adjustPoint(xypair, round) for xypair in ptcoords ]
        # Create the GC for this drawing
        gc = self.__scenepixmap.new_gc(background=self.__lastclearcolor)
        if self.__clipit:
            gc.set_clip_rectangle(self.__cliprect)
        # Set values for the line
        self.__helper.setLineFromCmnd(gc, cmnd["pen"])
        # Draw the lines
        self.__scenepixmap.draw_lines(gc, adjpts)
        # Check if the draw area is visible to see if the
        # draw requests are needed
        if self.__scenedrawarea.get_property("visible"):
            # queue redraws of the scene if appropriate
            self.queueSceneRedrawFromPoints(adjpts, gc.line_width, False)

    def drawPoints(self, cmnd):
        '''
        Draws a collection of discrete points using a single symbol
        for each point.

        Recognized keys from cmnd:
            "points": point centers as a list of (x,y) coordinates
            "symbol": name of the symbol to use
                    (see PyGtkCmndHelper.getSymbolFromCmnd)
            "size": size of the symbol (scales with view size)
            "color": color name (eg, "white", "#FF0088") or 
                     a 24-bit RGB integer value (eg, 0xFF0088)
 
        The coordinates are user coordinates from the bottom left corner.

        Raises a KeyError if the "symbol", "points", or "size" key
        is not given.
        '''
        # TODO:
        pass

    def drawPolygon(self, cmnd):
        '''
        Draws a polygon item to the viewer.

        Recognized keys from cmnd:
            "points": the vertices of the polygon as a list of (x,y)
                    coordinates
            "fill": dictionary describing the brush used to fill the
                    polygon; see PyGtkCmndHelper.setFillFromCmnd
                    If not given, the polygon will not be filled.
            "outline": dictionary describing the line used to outline
                    the polygon; see PyGtkCmndHelper.setLineFromCmnd

        The coordinates are user coordinates from the bottom left corner.

        Raises a KeyError if the "points" key is not given.
        '''
        # Get the points in device coordinates
        ptcoords = cmnd["points"]
        adjpts = [ self.adjustPoint(xypair, round) for xypair in ptcoords ]
        # Create the GC for this drawing
        gc = self.__scenepixmap.new_gc(background=self.__lastclearcolor)
        if self.__clipit:
            gc.set_clip_rectangle(self.__cliprect)
        # Set values for the line first so the fill color,
        # if given, is the foreground color.
        try:
            self.__helper.setLineFromCmnd(gc, cmnd["outline"])
        except KeyError:
            # add a single-pixel line around the polygon to
            # mask the white-lines-in-a-fill bug in Ferret
            gc.set_line_attributes(1, gtk.gdk.LINE_SOLID, gtk.gdk.CAP_ROUND,
                                   gtk.gdk.JOIN_ROUND)
        # Set values for the fill
        try:
            self.__helper.setFillFromCmnd(gc, cmnd["fill"])
            filled = True
        except KeyError:
            filled = False
        # Draw the polygon
        self.__scenepixmap.draw_polygon(gc, filled, adjpts)
        # explicitly close the polygon for an unfilled redraw
        if (not filled) and (adjpts[0] != adjpts[-1]):
            adjpts.append(adjpts[0])
        # Check if the draw area is visible to see if the
        # draw requests are needed
        if self.__scenedrawarea.get_property("visible"):
            # queue redraws of the scene if appropriate
            self.queueSceneRedrawFromPoints(adjpts, gc.line_width, filled)

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
                    rectangle; see PyGtkCmndHelper.setFillFromCmnd
                    If not given, the rectangle will not be filled.
            "outline": dictionary describing the line used to outline
                    the rectangle; see PyGtkCmndHelper.setLineFromCmnd

        The coordinates are user coordinates from the bottom left corner.

        Raises a ValueError if the width or height of the rectangle
        is not positive.
        '''
        # get the left, bottom, right, and top values
        # any keys not given get a zero value
        sides = self.__helper.getSidesFromCmnd(cmnd)
        # adjust to actual view coordinates from the top left
        lefttop = self.adjustPoint( (sides.left(), sides.top()), math.floor )
        rightbottom = self.adjustPoint( (sides.right(), sides.bottom()), math.ceil )
        width = rightbottom[0] - lefttop[0]
        if width <= 0:
            raise ValueError("width of the rectangle in not positive")
        height = rightbottom[1] - lefttop[1]
        if height <= 0:
            raise ValueError("height of the rectangle in not positive")
        # Create the GC for this drawing
        gc = self.__scenepixmap.new_gc(background=self.__lastclearcolor)
        if self.__clipit:
            gc.set_clip_rectangle(self.__cliprect)
        # Set values for the line first so the fill color,
        # if given, is the foreground color.
        try:
            self.__helper.setLineFromCmnd(gc, cmnd["outline"])
        except KeyError:
            pass
        # Set values for the fill
        try:
            self.__helper.setFillFromCmnd(gc, cmnd["fill"])
            filled = True
        except KeyError:
            filled = False
        # Draw the rectangle
        self.__scenepixmap.draw_rectangle(gc, filled, lefttop[0], lefttop[1],
                                          width, height)
        # Check if the draw area is visible to see if the
        # draw requests are needed
        if self.__scenedrawarea.get_property("visible"):
            # queue redraw of the scene
            adjpts = [ lefttop,
                      (lefttop[0] + width, lefttop[1]),
                      (lefttop[0] + width, lefttop[1] + height),
                      (lefttop[0], lefttop[1] + height),
                       lefttop ]
            self.queueSceneRedrawFromPoints(adjpts, gc.line_width, filled)

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
                    (see PyGtkCmndHelper.getcolorFromCmnd) which are
                    used to solidly fill each of the cells.  The first
                    row is at the top; the first column is on the left.

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
        # convert to device coordinates from the top left
        lefttop = self.adjustPoint( (sides.left(), sides.top()), math.floor  )
        rightbottom = self.adjustPoint( (sides.right(), sides.bottom()), math.ceil )
        fullwidth = rightbottom[0] - lefttop[0]
        if fullwidth <= 0:
            raise ValueError("width of the rectangle in not positive")
        fullheight = rightbottom[1] - lefttop[1]
        if fullheight <= 0:
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
        # Determine the device coordinates for all the rectangles
        fltwidth = float(fullwidth) / float(numcols)
        lefts = [ lefttop[0] + int( fltwidth * float(j) + 0.5 ) for j in xrange(numcols) ]
        widths = [ lefts[j] - lefts[j-1] for j in xrange(1, numcols) ]
        widths.append(rightbottom[0] - lefts[-1])
        fltheight = float(fullheight) / float(numrows)
        tops = [ lefttop[1] + int( fltheight * float(k) + 0.5 ) for k in xrange(numrows) ]
        heights = [ tops[k] - tops[k-1] for k in xrange(1, numrows) ]
        heights.append(rightbottom[1] - tops[-1])
        # Create the GC for this drawing
        gc = self.__scenepixmap.new_gc(background=self.__lastclearcolor)
        if self.__clipit:
            gc.set_clip_rectangle(self.__cliprect)
        gc.set_line_attributes(0, gtk.gdk.LINE_SOLID, gtk.gdk.CAP_ROUND, gtk.gdk.JOIN_ROUND)
        gc.set_fill(gtk.gdk.SOLID)
        colorindex = 0
        for j in xrange(numcols):
            for k in xrange(numrows):
                if (widths[j] > 0) and (heights[k] > 0):
                    # Assign the fill color
                    gc.set_foreground(colors[colorindex])
                    # Draw the cell of the rectangle
                    self.__scenepixmap.draw_rectangle(gc, True, lefts[j], tops[k],
                                                      widths[j], heights[k])
                colorindex += 1
        if self.__scenedrawarea.get_property("visible"):
            # queue redraw of the scene
            adjpts = [ lefttop,
                      (rightbottom[0], lefttop[1]),
                      rightbottom,
                      (lefttop[0], rightbottom[1]),
                       lefttop ]
            self.queueSceneRedrawFromPoints(adjpts, gc.line_width, True)

    def drawSimpleText(self, cmnd):
        '''
        Draws a "simple" text item in the current view.
        Raises a KeyError if the "text" key is not given.

        Recognized keys from cmnd:
            "text": string to displayed
            "font": dictionary describing the font to use;  see
                    PyGtkCmndHelper.setFontFromCmnd.  If not given
                    the default font for this viewer is used.
            "color": color name (eg, "white", "#FF0088") or 
                     a 24-bit RGB integer value (eg, 0xFF0088)
                     giving the color of the text.
            "rotate": clockwise rotation of the text in degrees
            "location": (x,y) location (user coordinates) in the
                    current view window for the baseline of the
                    start of text.
        '''
        # TODO:
        pass

    def queueSceneRedrawFromPoints(self, adjpts, linewidth, filled):
        '''
        Queues scene drawing area redraws of rectangle(s) covering
        the lines, polygon, or rectangle described by the arguments:
            adjpts: device coordinates of the points used in
                    drawing the lines, polygon, or rectangle
            linewidth: width of the line used in the drawing
            filled: whether the (polygon) was filled or not
        '''
        if linewidth < 1:
            linewidth = 1
        if not filled:
            # Queue a redraw of the rectangle affected by each line segment
            # Gtk will combine rectangles prior to the redraw.
            prevpair = None
            for xypair in adjpts:
                # Need pairs of points, skip the first point
                if prevpair != None:
                    # Get the enclosing rectangle in scene pixmap coordinates
                    if prevpair[0] <= xypair[0]:
                        xmin = prevpair[0] - linewidth
                        xmax = xypair[0] + linewidth
                    else:
                        xmin = xypair[0] - linewidth
                        xmax = prevpair[0] + linewidth
                    if prevpair[1] <= xypair[1]:
                        ymin = prevpair[1] - linewidth
                        ymax = xypair[1] + linewidth
                    else:
                        ymin = xypair[1] - linewidth
                        ymax = prevpair[1] + linewidth
                    # Scale the scene pixmap coordinates
                    xmin = int( math.floor(xmin * self.__scalefactor) )
                    xmin = max(xmin, 0)
                    xmax = int( math.ceil(xmax * self.__scalefactor) )
                    xmax = min(xmax, self.__scaledwidth)
                    ymin = int( math.floor(ymin * self.__scalefactor) )
                    ymin = max(ymin, 0)
                    ymax = int( math.ceil(ymax * self.__scalefactor) )
                    ymax = min(ymax, self.__scaledheight)
                    # Queue the redraw of this rectangle in the scene drawing area
                    self.__scenedrawarea.queue_draw_area(xmin, ymin,
                                                         xmax - xmin, ymax - ymin)
                # Record this point for the next iteration
                prevpair = xypair
        else:
            # Get the enclosing rectangle in scene pixmap coordinates
            xmin = self.__scenewidth
            xmax = 0
            ymin = self.__sceneheight
            ymax = 0
            for xypair in adjpts:
                xmin = min(xmin, xypair[0] - linewidth)
                xmax = max(xmax, xypair[0] + linewidth)
                ymin = min(ymin, xypair[1] - linewidth)
                ymax = max(ymax, xypair[1] + linewidth)
            # Scale the scene pixmap coordinates
            xmin = int( math.floor(xmin * self.__scalefactor) )
            xmin = max(xmin, 0)
            xmax = int( math.ceil(xmax * self.__scalefactor) )
            xmax = min(xmax, self.__scaledwidth)
            ymin = int( math.floor(ymin * self.__scalefactor) )
            ymin = max(ymin, 0)
            ymax = int( math.ceil(ymax * self.__scalefactor) )
            ymax = min(ymax, self.__scaledheight)
            # Queue the redraw of this rectangle in the scene drawing area
            self.__scenedrawarea.queue_draw_area(xmin, ymin,
                                                 xmax - xmin, ymax - ymin)

    def adjustPoint(self, xypair, intfunc):
        '''
        Returns appropriate device coordinates as a pair of integer
        corresponding to the user coordinates pair given in xypair.

        This adjusts for the flipped y coordinate as well as 
        transforming the point.
        
        If intfunc is not None, this function is applied to the floating
        point values of the transformed point prior to casting them
        as integers to be return.
        '''
        (userx, usery) = xypair
        usery = self.__userymax - float(usery)
        (devx, devy) = self.__activetrans.transform(float(userx), usery)
        if intfunc != None:
            devx = intfunc(devx)
            devy = intfunc(devy)
        devx = int(devx)
        devy = int(devy)
        return (devx, devy)

    def getScenePixmapSize(self):
        '''
        Returns the size of the current scene pixmap as an int tuple
        (width, height).
        '''
        return (self.__scenewidth, self.__sceneheight)


class PyGtkPipedImagerProcess(Process):
    '''
    A Process specifically tailored for creating a PyGtkPipedImager.
    '''
    def __init__(self, cmndpipe=sys.stdin, rspdpipe=sys.stdout):
        '''
        Create a Process that will produce a PyGtkPipedImager
        attached to the given Pipes when run.
        '''
        Process.__init__(self)
        self.__cmndpipe = cmndpipe
        self.__rspdpipe = rspdpipe

    def run(self):
        '''
        Create and run a PyGtkPipedImager that is attached to the Pipes
        of this instance.  This function does not return.  When the viewer
        exits, this function exits the Process using SystemExit.
        '''
        # Create the viewer, but let it show itself if or when requested
        self.__viewer = PyGtkPipedImager(self.__cmndpipe, self.__rspdpipe)
        # Run the viewer until it exits
        self.__viewer.runViewer()
        # Close pipes (if not standard sys connections) prior to exiting
        # this process
        if self.__cmndpipe != sys.stdin:
            self.__cmndpipe.close()
        if self.__rspdpipe != sys.stdout:
            self.__rspdpipe.close()
        # Exit this process
        SystemExit(0)

#
# The following are for testing this (and the pyqtqcmndhelper) modules
#

class _PyGtkCommandSubmitter(gtk.Dialog):
    '''
    Testing dialog for controlling the addition of commands to a pipe.
    Used for testing PyGtkPipedImager in the same process as the viewer.
    '''
    def __init__(self, parent, cmndpipe, rspdpipe, cmndlist):
        '''
        Create a Dialog with a single PushButton for controlling
        the submission of commands from cmndlist to cmndpipe.
        '''
        super(_PyGtkCommandSubmitter, self).__init__("Command Submitter", parent,
                                                     gtk.DIALOG_NO_SEPARATOR, None)
        self.__cmndlist = cmndlist
        self.__cmndpipe = cmndpipe
        self.__rspdpipe = rspdpipe
        self.__nextcmnd = 0
        button = gtk.Button("Submit next command", )
        button.connect("clicked", self.submitNextCommand)
        self.vbox.pack_start(button, True, True, 0)
        button.show()

    def submitNextCommand(self, widget, data=None):
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
            self.destroy()


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
    # pipes to communicate between the dialog and the viewer in this process
    cmndrecvpipe, cmndsendpipe = Pipe(False)
    rspdrecvpipe, rspdsendpipe = Pipe(False)
    # create a PyGtkPipedViewer in this process
    viewer = PyGtkPipedImager(cmndrecvpipe, rspdsendpipe)
    # create a command submitter dialog in this process
    tester = _PyGtkCommandSubmitter(viewer, cmndsendpipe,
                                    rspdrecvpipe, drawcmnds)
    tester.show()
    # let it all run
    viewer.runViewer()
