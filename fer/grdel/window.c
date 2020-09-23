/*
 * "Window" refers to the full canvas; however, no drawing, except
 * possibly clearing the window, is done on the full window.  Instead,
 * a "View" of the Window is "begun", and drawing is done in that View
 * of the Window.
 *
 * "View" refers to a rectangular subsection of the Window (possibly
 * the complete canvas of the Window).
 *
 * In order to draw in a Window, a View must first have been specified
 * using grdelViewBegin.  When drawing in a View is complete, grdelViewEnd
 * is called, at which point the Window should be updated.
 *
 * Only one View can be active at any time.  So a switch between views
 * requires ending one view and beginning a another view.
 *
 * A segment is an collection of drawing commands with an ID.  Drawing
 * commands in a segment can be deleted and the image recreated from
 * the remaining drawing commands.
 */
#include <Python.h> /* make sure Python.h is first */
#include <string.h>
#include "grdel.h"
#include "cferbind.h"
#include "pyferret.h"
#include "FerMem.h"

#ifdef GRDELDEBUG
#include <stdio.h>
#include <stdlib.h>

FILE *debuglogfile;

static void closelogfile(void)
{
    if ( debuglogfile != NULL ) {
        fclose(debuglogfile);
        debuglogfile = NULL;
    }
}

static void openlogfile(void)
{
    debuglogfile = fopen("grdeldebug.log", "w");
    if ( debuglogfile == NULL ) {
        fputs("Open of grdeldebug.log failed", stderr);
        fflush(stderr);
        exit(1);
    }
    atexit(closelogfile);
}
#endif /* GRDELDEBUG */


static const char *grdelwindowid = "GRDEL_WINDOW";

/* Instantiate the global error message string */
char grdelerrmsg[2048];

typedef struct GDWindow_ {
    const char *id;
    BindObj   bindings;
    grdelBool hasview;
    grdelBool hasseg;
} GDWindow;

/*
 * Copies of the error message in grdelerrmsg to errmsg, but will not null
 * terminated.  The argument errmsg should be at least 2048 characters in
 * length and initialized to all blank prior to calling this function for
 * return of a proper Fortran string.  The argument errmsglen is assigned
 * the actual length of the message returned in errmsg.
 */
void FORTRAN(fgderrmsg)(char *errmsg, int *errmsglen)
{
    *errmsglen = strlen(grdelerrmsg);
    strncpy(errmsg, grdelerrmsg, *errmsglen);
}

/*
 * Creates and returns a Window object.
 * Initializes the graphics engine if needed.
 *
 * Arguments:
 *     engine: name of the graphics engine for creating the Window
 *     enginelen: actual length of the graphics engine name
 *     title: display title for the Window
 *     titlelen: actual length of the title
 *     visible: display Window on start-up?
 *     noalpha: do not use the alpha channel (opacity) in colors
 *     rasteronly: only create raster images
 *
 * Returns a pointer to the window object created.
 * If an error occurs, NULL is returned and
 * grdelerrmsg contains an explanatory message.
 */
grdelType grdelWindowCreate(const char *engine, int enginelen,
                            const char *title, int titlelen,
                            grdelBool visible, grdelBool noalpha,
                            grdelBool rasteronly)
{
    GDWindow *window;
    PyObject *visiblebool;
    PyObject *noalphabool;
    PyObject *rasteronlybool;

    /* Allocate memory for this GDWindow */
    window = (GDWindow *) FerMem_Malloc(sizeof(GDWindow), __FILE__, __LINE__);
    if ( window == NULL ) {
        strcpy(grdelerrmsg, "grdelWindowCreate: out of memory for a new Window");
        return NULL;
    }
    window->id = grdelwindowid;
    window->bindings.cferbind = NULL;
    window->bindings.pyobject = NULL;
    window->hasview = 0;
    window->hasseg = 0;

    /*
     * First try to create the C window bindings for this engine.
     * This will fail if it is a Python-based engine.
     */
    window->bindings.cferbind = cferbind_createWindow(engine, enginelen,
                           title, titlelen, visible, noalpha, rasteronly);
    if ( window->bindings.cferbind != NULL ) {
        /* Success - engine found; done */
#ifdef GRDELDEBUG
        /*
         * Since a window has to be created before anything else can happen
         * the initialization check of debuglogfile only needs to happen here.
         */
        if ( debuglogfile == NULL )
            openlogfile();
        fprintf(debuglogfile, "grdelWindow created with C bindings: "
                              "window = %p\n", window);
        fflush(debuglogfile);
#endif
        grdelerrmsg[0] = '\0';
        return window;
    }

    /* C bindings failed, try Python bindings */

    /*
     * Call pyferret.graphbind.createWindow, which, if successful,
     * will create and return an instance of the bindings for this
     * graphics engine.
     */
    if ( visible )
        visiblebool = Py_True;
    else
        visiblebool = Py_False;
    if ( noalpha )
        noalphabool = Py_True;
    else
        noalphabool = Py_False;
    if ( rasteronly )
        rasteronlybool = Py_True;
    else
        rasteronlybool = Py_False;
    window->bindings.pyobject =
            PyObject_CallMethod(pyferret_graphbind_module_pyobject,
                                "createWindow", "s#s#OOO",
                                engine, enginelen,
                                title, titlelen,
                                visiblebool, noalphabool, rasteronlybool);
    if ( window->bindings.pyobject == NULL ) {
        sprintf(grdelerrmsg, "grdelWindowCreate: error when calling createWindow "
                             "in pyferret.graphbind: %s", pyefcn_get_error());
        FerMem_Free(window, __FILE__, __LINE__);
        return NULL;
    }

#ifdef GRDELDEBUG
    /*
     * Since a window has to be created before anything else can happen
     * the initialization check of debuglogfile only needs to happen here.
     */
    if ( debuglogfile == NULL )
        openlogfile();
    fprintf(debuglogfile, "grdelWindow created with Python bindings: "
                          "window = %p\n", window);
    fflush(debuglogfile);
#endif

    /* return the pointer to the GDWindow */
    return window;
}


/*
 * Verifies the argument is a grdel Window.
 * Returns the graphic engine's window bindings
 * object if it is.  Returns NULL if it is not.
 */
const BindObj *grdelWindowVerify(grdelType window)
{
    GDWindow *mywindow;

    if ( window == NULL )
        return NULL;
    mywindow = (GDWindow *) window;
    if ( mywindow->id != grdelwindowid )
        return NULL;
    return &(mywindow->bindings);
}


/*
 * Deletes (closes and destroys) a Window created by grdelWindowCreate.
 *
 * Arguments:
 *     window: Window to be closed
 *
 * Returns success or failure.  If failure, grdelerrmsg contains
 * an explanatory message.
 */
grdelBool grdelWindowDelete(grdelType window)
{
    GDWindow *mywindow;
    grdelBool success;
    PyObject *result;

#ifdef GRDELDEBUG
    fprintf(debuglogfile, "grdelWindowDelete called: "
            "window = %p\n", window);
    fflush(debuglogfile);
#endif

    if ( grdelWindowVerify(window) == NULL ) {
        strcpy(grdelerrmsg, "grdelWindowDelete: window argument is not "
                            "a grdel Window");
        return 0;
    }
    mywindow = (GDWindow *) window;

    /* if a view is still active, end it */
    if ( mywindow->hasview ) {
        if ( ! grdelWindowViewEnd(window) ) {
            /* grdelerrmsg already assigned */
            return 0;
        }
    }

    if ( mywindow->bindings.cferbind != NULL ) {
        /* Call the deleteWindow method to delete the bindings instance. */
        success = mywindow->bindings.cferbind->
                                    deleteWindow(mywindow->bindings.cferbind);
        if ( ! success ) {
            /* grdelerrmsg already assigned */
            return 0;
        }
    }
    else if ( mywindow->bindings.pyobject != NULL ) {
        /*
         * Call the deleteWindow method of the bindings instance.
         * If True is returned, decrement the reference to this
         * bindings instance.
         */
        result = PyObject_CallMethod(mywindow->bindings.pyobject,
                                     "deleteWindow", NULL);
        if ( result == NULL ) {
            sprintf(grdelerrmsg, "grdelWindowDelete: error when calling the "
                    "Python binding's deleteWindow method: %s", pyefcn_get_error());
            return 0;
        }
        Py_DECREF(result);
        /* Py_True and Py_False are singleton objects */
        if ( result != Py_True ) {
            strcpy(grdelerrmsg, "grdelWindowDelete: deleteWindow method "
                                "returned False");
            return 0;
        }
        Py_DECREF(mywindow->bindings.pyobject);
    }
    else {
        strcpy(grdelerrmsg, "grdelWindowDelete: unexpected error, "
                            "no bindings associated with this Window");
        return 0;
    }

    /* Free the memory for the GDWindow */
    mywindow->id = NULL;
    mywindow->hasview = 0;
    mywindow->hasseg = 0;
    mywindow->bindings.cferbind = NULL;
    mywindow->bindings.pyobject = NULL;
    FerMem_Free(window, __FILE__, __LINE__);

    return 1;
}

/*
 * Frees objects and memory associated with a Window created by
 * grdelWindowCreate.  Assumes the actual window has already been
 * closed; e.g., using the window frame 'X' button.  Assigns
 * grdelerrmsg, normally with the 'window was closed' message.
 *
 * Arguments:
 *     window: Window to be freed
 */
void grdelWindowFree(grdelType window)
{
    GDWindow *mywindow;

#ifdef GRDELDEBUG
    fprintf(debuglogfile, "grdelWindowFree called: "
            "window = %p\n", window);
    fflush(debuglogfile);
#endif

    if ( grdelWindowVerify(window) == NULL ) {
        strcpy(grdelerrmsg, "grdelWindowFree: window argument is not "
                            "a grdel Window");
        return;
    }
    mywindow = (GDWindow *) window;

    /* Free objects associated with this window and remove this window from Ferret's list. */
    FORTRAN(window_killed)(window);

    /* Free the bindings instance */
    if ( mywindow->bindings.cferbind != NULL ) {
        /* Just call the deleteWindow method to delete the bindings instance. */
        mywindow->bindings.cferbind->deleteWindow(mywindow->bindings.cferbind);
    }
    else if ( mywindow->bindings.pyobject != NULL ) {
        /* Just decrement the reference to the Python bindings instance. */
        Py_DECREF(mywindow->bindings.pyobject);
    }
    else {
        strcpy(grdelerrmsg, "grdelWindowFree: unexpected error, "
                            "no bindings associated with this Window");
        return;
    }

    /* Free the memory for the GDWindow */
    mywindow->id = NULL;
    mywindow->hasview = 0;
    mywindow->hasseg = 0;
    mywindow->bindings.cferbind = NULL;
    mywindow->bindings.pyobject = NULL;
    FerMem_Free(window, __FILE__, __LINE__);
    strcpy(grdelerrmsg, "window was closed");
}


/*
 * Assigns the image filename and format.  This may just be a default
 * filename when saving a window (for interactive graphics window), or
 * it may be the filename which is written as the drawing proceeds (for
 * non-interactive "batch mode" graphics without a display window).
 *
 * Arguments:
 *     window:     Window to use
 *     imagename:  filename for saving the image
 *     imgnamelen: actual length of imagename
 *     formatname: name of the format (case insensitive);
 *                 eg, "PNG", "PDF", "PS".  May be NULL.
 *     fmtnamelen: actual lenght of formatname (zero if NULL)
 *
 * If formatname is empty or NULL, the format is quessed for the
 * filename extension of imagename.
 *
 * Returns success (nonzero) or failure (zero).
 * If failure, grdelerrmsg contains an explanatory message.
 */
grdelBool grdelWindowSetImageName(grdelType window, const char *imagename,
                     int imgnamelen, const char *formatname, int fmtnamelen)
{
    GDWindow *mywindow;
    grdelBool success;
    PyObject *result;

#ifdef GRDELDEBUG
    fprintf(debuglogfile, "grdelWindowSetImageName called: "
            "window = %p\n", window);
    fflush(debuglogfile);
#endif

    if ( grdelWindowVerify(window) == NULL ) {
        strcpy(grdelerrmsg, "grdelWindowSetImageName: window argument is not "
                            "a grdel Window");
        return 0;
    }
    mywindow = (GDWindow *) window;

    if ( mywindow->bindings.cferbind != NULL ) {
        success = mywindow->bindings.cferbind->
                            setImageName(mywindow->bindings.cferbind, imagename,
                                         imgnamelen, formatname, fmtnamelen);
        if ( ! success ) {
            /* grdelerrmsg already assigned */
            return 0;
        }
    }
    else if ( mywindow->bindings.pyobject != NULL ) {
        result = PyObject_CallMethod(mywindow->bindings.pyobject, "setImageName",
                                     "s#s#", imagename, imgnamelen,
                                     formatname, fmtnamelen);
        if ( result == NULL ) {
            sprintf(grdelerrmsg, "grdelWindowSetImageName: Error when calling "
                    "the Python binding's setImageName method: %s", pyefcn_get_error());
            return 0;
        }
        Py_DECREF(result);
    }
    else {
        strcpy(grdelerrmsg, "grdelWindowClear: unexpected error, "
                            "no bindings associated with this Window");
        return 0;
    }

    return 1;
}

/*
 * Clears the window of all drawings.  The window is filled
 * (initialized) with bkgcolor; i.e., the background color.
 *
 * Arguments:
 *     window: Window to be cleared
 *     bkgcolor: Color to fill (initialize) the scene
 *
 * Returns success (nonzero) or failure (zero).
 * If failure, grdelerrmsg contains an explanatory message.
 */
grdelBool grdelWindowClear(grdelType window, grdelType bkgcolor)
{
    GDWindow *mywindow;
    grdelType colorobj;
    grdelBool success;
    PyObject *result;

#ifdef GRDELDEBUG
    fprintf(debuglogfile, "grdelWindowClear called: "
            "window = %p, bkgcolor = %p\n", window, bkgcolor);
    fflush(debuglogfile);
#endif

    if ( grdelWindowVerify(window) == NULL ) {
        strcpy(grdelerrmsg, "grdelWindowClear: window argument is not "
                            "a grdel Window");
        return 0;
    }
    mywindow = (GDWindow *) window;
    colorobj = grdelColorVerify(bkgcolor, window);
    if ( colorobj == NULL ) {
        strcpy(grdelerrmsg, "grdelWindowClear: bkgcolor argument is not "
                            "a valid grdel Color for the window");
        return 0;
    }

    if ( mywindow->bindings.cferbind != NULL ) {
        success = mywindow->bindings.cferbind->
                            clearWindow(mywindow->bindings.cferbind, colorobj);
        if ( ! success ) {
            /* grdelerrmsg already assigned */
            return 0;
        }
    }
    else if ( mywindow->bindings.pyobject != NULL ) {
        result = PyObject_CallMethod(mywindow->bindings.pyobject, "clearWindow",
                                     "O", (PyObject *) colorobj);
        if ( result == NULL ) {
            sprintf(grdelerrmsg, "grdelWindowClear: Error when calling "
                    "the Python binding's clearWindow method: %s", pyefcn_get_error());
            return 0;
        }
        Py_DECREF(result);
    }
    else {
        strcpy(grdelerrmsg, "grdelWindowClear: unexpected error, "
                            "no bindings associated with this Window");
        return 0;
    }

    return 1;
}

/*
 * Redraws the current drawing with bkgcolor as the background Color.
 *
 * Arguments:
 *     window: Window to be cleared
 *     bkgcolor: Color to fill (initialize) the scene
 *
 * Returns success (nonzero) or failure (zero).
 * If failure, grdelerrmsg contains an explanatory message.
 */
grdelBool grdelWindowRedraw(grdelType window, grdelType bkgcolor)
{
    GDWindow *mywindow;
    grdelType colorobj;
    grdelBool success;
    PyObject *result;

#ifdef GRDELDEBUG
    fprintf(debuglogfile, "grdelWindowRedraw called: "
            "window = %p, bkgcolor = %p\n", window, bkgcolor);
    fflush(debuglogfile);
#endif

    if ( grdelWindowVerify(window) == NULL ) {
        strcpy(grdelerrmsg, "grdelWindowRedraw: window argument is not "
                            "a grdel Window");
        return 0;
    }
    mywindow = (GDWindow *) window;
    colorobj = grdelColorVerify(bkgcolor, window);
    if ( colorobj == NULL ) {
        strcpy(grdelerrmsg, "grdelWindowRedraw: bkgcolor argument is not "
                            "a valid grdel Color for the window");
        return 0;
    }

    if ( mywindow->bindings.cferbind != NULL ) {
        success = mywindow->bindings.cferbind->
                            redrawWindow(mywindow->bindings.cferbind, colorobj);
        if ( ! success ) {
            /* grdelerrmsg already assigned */
            return 0;
        }
    }
    else if ( mywindow->bindings.pyobject != NULL ) {
        result = PyObject_CallMethod(mywindow->bindings.pyobject, "redrawWindow",
                                     "O", (PyObject *) colorobj);
        if ( result == NULL ) {
            sprintf(grdelerrmsg, "grdelWindowRedraw: Error when calling "
                    "the Python binding's redrawWindow method: %s", pyefcn_get_error());
            return 0;
        }
        Py_DECREF(result);
    }
    else {
        strcpy(grdelerrmsg, "grdelWindowRedraw: unexpected error, "
                            "no bindings associated with this Window");
        return 0;
    }

    return 1;
}

/*
 * Updates a Window, adding any new drawing elements and redrawing.
 *
 * Arguments:
 *     window: Window to be updated
 *
 * Returns success or failure.  If failure, grdelerrmsg contains
 * an explanatory message.
 */
grdelBool grdelWindowUpdate(grdelType window)
{
    GDWindow *mywindow;
    grdelBool success;
    PyObject *result;

#ifdef GRDELDEBUG
    fprintf(debuglogfile, "grdelWindowUpdate called: "
            "window = %p\n", window);
    fflush(debuglogfile);
#endif

    if ( grdelWindowVerify(window) == NULL ) {
        strcpy(grdelerrmsg, "grdelWindowUpdate: window argument is not "
                            "a grdel Window");
        return 0;
    }
    mywindow = (GDWindow *) window;

    if ( mywindow->bindings.cferbind != NULL ) {
        success = mywindow->bindings.cferbind->
                            updateWindow(mywindow->bindings.cferbind);
        if ( ! success ) {
            /* grdelerrmsg already assigned */
            return 0;
        }
    }
    else if ( mywindow->bindings.pyobject != NULL ) {
        /* Call the updateWindow method of the bindings instance. */
        result = PyObject_CallMethod(mywindow->bindings.pyobject, "updateWindow",
                                     NULL);
        if ( result == NULL ) {
            sprintf(grdelerrmsg, "grdelWindowUpdate: error when calling the "
                    "Python binding's updateWindow method: %s", pyefcn_get_error());
            return 0;
        }
        Py_DECREF(result);
    }
    else {
        strcpy(grdelerrmsg, "grdelWindowUpdate: unexpected error, "
                            "no bindings associated with this Window");
        return 0;
    }

    return 1;
}

/*
 * Turns on or off anti-aliasing on items drawn after this call.
 *
 * Arguments:
 *     window: Window to be updated
 *     antialias: if zero, turn off antialiasing;
 *                otherwise, turn on antialiasing
 *
 * Returns one if successful.   If an error occurs, grdelerrmsg
 * is assigned an appropriate error message and zero is returned.
 */
grdelBool grdelWindowSetAntialias(grdelType window, int antialias)
{
    GDWindow *mywindow;
    grdelBool success;
    PyObject *aaobj;
    PyObject *result;

#ifdef GRDELDEBUG
    fprintf(debuglogfile, "grdelWindowSetAntialias called: "
            "window = %p, antialias = %d\n", window, antialias);
    fflush(debuglogfile);
#endif

    if ( grdelWindowVerify(window) == NULL ) {
        strcpy(grdelerrmsg, "grdelWindowSetAntialias: window argument is not "
                            "a grdel Window");
        return 0;
    }
    mywindow = (GDWindow *) window;

    if ( mywindow->bindings.cferbind != NULL ) {
        success = mywindow->bindings.cferbind->
                            setAntialias(mywindow->bindings.cferbind, antialias);
        if ( ! success ) {
            /* grdelerrmsg already assigned */
            return 0;
        }
    }
    else if ( mywindow->bindings.pyobject != NULL ) {
        if ( antialias == 0 )
            aaobj = Py_False;
        else
            aaobj = Py_True;
        result = PyObject_CallMethod(mywindow->bindings.pyobject, "setAntialias",
                                     "O", aaobj);
        if ( result == NULL ) {
            sprintf(grdelerrmsg, "grdelWindowSetAntialias: error when calling the "
                    "Python binding's setAntiAlias method: %s", pyefcn_get_error());
            return 0;
        }
        Py_DECREF(result);
    }
    else {
        strcpy(grdelerrmsg, "grdelWindowSetAntialias: unexpected error, "
                            "no bindings associated with this Window");
        return 0;
    }

    return 1;
}

/*
 * Sets the current size of a Window.
 *
 * Arguments:
 *     window: Window to use
 *     width: width of the Window, in "device units"
 *     height: height of the window in "device units"
 *
 * "device units" is pixels at the current window DPI.
 *
 * Returns success (nonzero) or failure (zero).
 * If failure, grdelerrmsg contains an explanatory message.
 */
grdelBool grdelWindowSetSize(grdelType window, float width, float height)
{
    GDWindow *mywindow;
    grdelBool success;
    PyObject *result;

#ifdef GRDELDEBUG
    fprintf(debuglogfile, "grdelWindowSetSize called: "
            "window = %p, width = %f, height = %f\n", window, width, height);
    fflush(debuglogfile);
#endif

    if ( grdelWindowVerify(window) == NULL ) {
        strcpy(grdelerrmsg, "grdelWindowSetSize: window argument is not "
                            "a grdel Window");
        return 0;
    }
    mywindow = (GDWindow *) window;

    if ( mywindow->bindings.cferbind != NULL ) {
        success = mywindow->bindings.cferbind->
                            resizeWindow(mywindow->bindings.cferbind,
                                         (double) width, (double) height);
        if ( ! success ) {
            /* grdelerrmsg already assigned */
            return 0;
        }
    }
    else if ( mywindow->bindings.pyobject != NULL ) {
        result = PyObject_CallMethod(mywindow->bindings.pyobject, "resizeWindow",
                                     "dd", (double) width, (double) height);
        if ( result == NULL ) {
            sprintf(grdelerrmsg, "grdelWindowSetSize: error when calling the "
                    "Python binding's resizeWindow method: %s", pyefcn_get_error());
            return 0;
        }
        Py_DECREF(result);
    }
    else {
        strcpy(grdelerrmsg, "grdelWindowSetSize: unexpected error, "
                            "no bindings associated with this Window");
        return 0;
    }

    return 1;
}

/*
 * Sets the current scaling factor of a Window.
 *
 * Arguments:
 *     window: Window to use
 *     scale: scaling factor for the Window
 *
 * Returns success (nonzero) or failure (zero).
 * If failure, grdelerrmsg contains an explanatory message.
 */
grdelBool grdelWindowSetScale(grdelType window, float scale)
{
    GDWindow *mywindow;
    grdelBool success;
    PyObject *result;

#ifdef GRDELDEBUG
    fprintf(debuglogfile, "grdelWindowSetScale called: "
            "window = %p, scale = %f\n", window, scale);
    fflush(debuglogfile);
#endif

    if ( grdelWindowVerify(window) == NULL ) {
        strcpy(grdelerrmsg, "grdelWindowSetScale: window argument is not "
                            "a grdel Window");
        return 0;
    }
    mywindow = (GDWindow *) window;

    if ( mywindow->bindings.cferbind != NULL ) {
        success = mywindow->bindings.cferbind->
                            scaleWindow(mywindow->bindings.cferbind,
                                        (double) scale);
        if ( ! success ) {
            /* grdelerrmsg already assigned */
            return 0;
        }
    }
    else if ( mywindow->bindings.pyobject != NULL ) {
        result = PyObject_CallMethod(mywindow->bindings.pyobject, "scaleWindow",
                                     "d", (double) scale);
        if ( result == NULL ) {
            sprintf(grdelerrmsg, "grdelWindowSetScale: error when calling the "
                    "Python binding's scaleWindow method: %s", pyefcn_get_error());
            return 0;
        }
        Py_DECREF(result);
    }
    else {
        strcpy(grdelerrmsg, "grdelWindowSetScale: unexpected error, "
                            "no bindings associated with this Window");
        return 0;
    }

    return 1;
}

/*
 * Display or hide a Window.  A graphics engine that does not
 * have the ability to display a Window will ignore this call.
 *
 * Arguments:
 *     window: Window to use
 *     visible: display Window?
 *
 * Returns success (nonzero) or failure (zero).
 * If failure, grdelerrmsg contains an explanatory message.
 */
grdelBool grdelWindowSetVisible(grdelType window, grdelBool visible)
{
    GDWindow *mywindow;
    grdelBool success;
    PyObject *visiblebool;
    PyObject *result;

#ifdef GRDELDEBUG
    fprintf(debuglogfile, "grdelWindowSetVisible called: "
            "window = %p, visible = %d\n", window, visible);
    fflush(debuglogfile);
#endif

    if ( grdelWindowVerify(window) == NULL ) {
        strcpy(grdelerrmsg, "grdelWindowSetVisible: window argument is not "
                            "a grdel Window");
        return 0;
    }
    mywindow = (GDWindow *) window;

    if ( mywindow->bindings.cferbind != NULL ) {
        success = mywindow->bindings.cferbind->
                            showWindow(mywindow->bindings.cferbind, visible);
        if ( ! success ) {
            /* grdelerrmsg already assigned */
            return 0;
        }
    }
    else if ( mywindow->bindings.pyobject != NULL ) {
        if ( visible )
            visiblebool = Py_True;
        else
            visiblebool = Py_False;
        result = PyObject_CallMethod(mywindow->bindings.pyobject, "showWindow",
                                     "O", visiblebool);
        if ( result == NULL ) {
            sprintf(grdelerrmsg, "grdelWindowSetVisible: error when calling the "
                    "Python binding's showWindow method: %s", pyefcn_get_error());
            return 0;
        }
        Py_DECREF(result);
    }
    else {
        strcpy(grdelerrmsg, "grdelWindowSetVisible: unexpected error, "
                            "no bindings associated with this Window");
        return 0;
    }

    return 1;
}

/*
 * Save the contents of the window to a file.
 *
 * Arguments:
 *     window: Window to use
 *     filename: name of the file to create
 *     filenamelen: actual length of the file name
 *     fileformat: name of the format to use
 *     formatlen: actual length of the format name
 *     transparentbkg: make the background transparent?
 *     xinches: horizontal size of vector image in inches
 *     yinches: vertical size of vector image in inches
 *     xpixels: horizontal size of raster image in pixels
 *     ypixels: vertical size of raster image in pixels
 *     annotations: array of annotation strings;
 *                  pointers are always 8 bytes apart
 *     numannotations: number of annotation strings
 *
 * If fileformat is NULL, the fileformat is guessed from the
 * filename extension.
 *
 * Returns success (nonzero) or failure (zero).
 * If failure, grdelerrmsg contains an explanatory message.
 */
grdelBool grdelWindowSave(grdelType window, const char *filename,
                          int filenamelen, const char *fileformat,
                          int formatlen, grdelBool transparentbkg,
                          float xinches, float yinches,
                          int xpixels, int ypixels,
                          void **annotations, int numannotations)
{
    GDWindow *mywindow;
    grdelBool success;
    PyObject *transparentbool;
    PyObject *annostuple;
    PyObject *annostrobj;
    PyObject *result;
    int k;

#ifdef GRDELDEBUG
    fprintf(debuglogfile, "grdelWindowSave called: "
            "window = %p\n", window);
    fflush(debuglogfile);
#endif

    if ( grdelWindowVerify(window) == NULL ) {
        strcpy(grdelerrmsg, "grdelWindowSave: window argument is not "
                            "a grdel Window");
        return 0;
    }
    mywindow = (GDWindow *) window;

    if ( mywindow->bindings.cferbind != NULL ) {
        success = mywindow->bindings.cferbind->
                            saveWindow(mywindow->bindings.cferbind,
                                       filename, filenamelen,
                                       fileformat, formatlen, transparentbkg,
                                       xinches, yinches, xpixels, ypixels,
                                       annotations, numannotations);
        if ( ! success ) {
            /* grdelerrmsg already assigned */
            return 0;
        }
    }
    else if ( mywindow->bindings.pyobject != NULL ) {
        if ( transparentbkg )
            transparentbool = Py_True;
        else
            transparentbool = Py_False;
        if ( numannotations > 0 ) {
            annostuple = PyTuple_New((Py_ssize_t) numannotations);
            if ( annostuple == NULL ) {
                strcpy(grdelerrmsg, "grdelWindowSave: unexpected error, "
                                    "unable to create a tuple for the annotations");
                return 0;
            }
            for (k = 0; k < numannotations; k++) {
#if PY_MAJOR_VERSION > 2
                annostrobj = PyUnicode_FromString((char *) annotations[k * 8 / sizeof(void *)]);
#else
                annostrobj = PyString_FromString((char *) annotations[k * 8 / sizeof(void *)]);
#endif
                if ( annostrobj == NULL ) {
                    Py_DECREF(annostuple);
                    strcpy(grdelerrmsg, "grdelWindowSave: unexpected error, "
                                        "unable to create a annotation string object");
                    return 0;
                }
                PyTuple_SET_ITEM(annostuple, (Py_ssize_t) k, annostrobj);
            }
        }
        else {
            annostuple = Py_None;
            /* PyObect_CallMethod will steal the reference to annostuple (N instead of O) */
            Py_INCREF(Py_None);
        }
        result = PyObject_CallMethod(mywindow->bindings.pyobject,
                                     "saveWindow", "s#s#OddiiN",
                                     filename, filenamelen,
                                     fileformat, formatlen, transparentbool,
                                     (double) xinches, (double) yinches,
                                     xpixels, ypixels, annostuple);
        if ( result == NULL ) {
            sprintf(grdelerrmsg, "grdelWindowSave: error when calling the "
                    "Python binding's saveWindow method: %s", pyefcn_get_error());
            return 0;
        }
        Py_DECREF(result);
    }
    else {
        strcpy(grdelerrmsg, "grdelWindowSave: unexpected error, "
                            "no bindings associated with this Window");
        return 0;
    }

    return 1;
}


/*
 * Get information about the default screen (display) for a window.
 *
 * Input Arguments:
 *     window: Window to use
 *
 * Output Arguments:
 *     dpix: the number of dots per inch in the horizontal (X) direction
 *     dpiy: the number of dots per inch in the vertical (Y) direction
 *     screenwidth: the width of the screen (display) in pixels (dots)
 *     screenheight: the height of the screen (display) in pixels (dots)
 *
 * Returns success (nonzero) or failure (zero).
 * If failure, grdelerrmsg contains an explanatory message.
 */
grdelBool grdelWindowScreenInfo(grdelType window, float *dpix, float *dpiy,
                                int *screenwidth, int *screenheight)
{
    GDWindow *mywindow;
    grdelBool success;
    PyObject *result;

#ifdef GRDELDEBUG
    fprintf(debuglogfile, "grdelWindowScreenInfo called: "
            "window = %p\n", window);
    fflush(debuglogfile);
#endif

    if ( grdelWindowVerify(window) == NULL ) {
        strcpy(grdelerrmsg, "grdelWindowScreenInfo: window argument is not a grdel Window");
        return 0;
    }
    mywindow = (GDWindow *) window;

    if ( mywindow->bindings.cferbind != NULL ) {
        success = mywindow->bindings.cferbind->
                            windowScreenInfo(mywindow->bindings.cferbind,
                                        dpix, dpiy, screenwidth, screenheight);
        if ( success == 0 ) {
            /* grdelerrmsg already assigned */
            return 0;
        }
    }
    else if ( mywindow->bindings.pyobject != NULL ) {
        result = PyObject_CallMethod(mywindow->bindings.pyobject, "windowScreenInfo", NULL);
        if ( result == NULL ) {
            sprintf(grdelerrmsg, "grdelWindowScreenInfo: error when calling the Python "
                                 "binding's windowScreenInfo method: %s", pyefcn_get_error());
            return 0;
        }
        if ( ! PyArg_ParseTuple(result, "ffii", dpix, dpiy,
                                                screenwidth, screenheight) ) {
            Py_DECREF(result);
            sprintf(grdelerrmsg, "grdelWindowScreenInfo: Error when parsing the Python "
                                 "binding's windowScreenInfo return value: %s", pyefcn_get_error());
            return 0;
        }
        Py_DECREF(result);
    }
    else {
        strcpy(grdelerrmsg, "grdelWindowScreenInfo: unexpected error, "
                            "no bindings associated with this Window");
        return 0;
    }

#ifdef GRDELDEBUG
    fprintf(debuglogfile, "grdelWindowScreenInfo response: "
                          "dpix = %f, dpiy = %f, screenwidth = %d, screenheight = %d\n",
                          *dpix, *dpiy, *screenwidth, *screenheight);
    fflush(debuglogfile);
#endif

    return 1;
}


/*
 * Assign the window DPI.
 * Will only be successful if the window is not associated with a display.
 *
 * Input Arguments:
 *     window: Window to use
 *     newdpi: the number of dots per inch to assign
 *
 * Output Arguments:
 *     success: one if successful,
 *              zero if an error occurred,
 *              negative one if the window is associated with a display
 */
int  grdelWindowSetDpi(grdelType window, float newdpi)
{
    GDWindow *mywindow;

#ifdef GRDELDEBUG
    fprintf(debuglogfile, "grdelWindowSetDpi called: "
            "window = %p, newdpi = %f\n", window, newdpi);
    fflush(debuglogfile);
#endif

    if ( grdelWindowVerify(window) == NULL ) {
        strcpy(grdelerrmsg, "grdelWindowSetDpi: "
                            "window argument is not a grdel Window");
        return 0;
    }
    mywindow = (GDWindow *) window;

    if ( mywindow->bindings.pyobject != NULL ) {
       /* anything with Python bindings has a display associated with it */
       return -1;
    }
    if ( mywindow->bindings.cferbind == NULL ) {
        strcpy(grdelerrmsg, "grdelWindowSetDpi: unexpected error, "
                            "no bindings associated with this Window");
        return 0;
    }
    if ( mywindow->bindings.cferbind->setWindowDpi == NULL ) {
        return -1;
    }
    mywindow->bindings.cferbind->setWindowDpi(mywindow->bindings.cferbind, (double) newdpi);
    return 1;
}


/*
 * Set the scaling factor for pen widths, symbol sizes, and font sizes.
 *
 * Arguments:
 *     window: Window to use
 *     widthfactor: scaling factor to use
 *
 * Returns success (nonzero) or failure (zero).
 * If failure, grdelerrmsg contains an explanatory message.
 */
grdelBool grdelWindowSetWidthFactor(grdelType window, float widthfactor)
{
    GDWindow *mywindow;
    grdelBool success;
    PyObject *result;

#ifdef GRDELDEBUG
    fprintf(debuglogfile, "grdelWindowSetWidthFactor called: "
            "window = %p, scalingfactor = %f\n", window, widthfactor);
    fflush(debuglogfile);
#endif

    if ( grdelWindowVerify(window) == NULL ) {
        strcpy(grdelerrmsg, "grdelWindowSetWidthFactor: "
                            "window argument is not a grdel Window");
        return 0;
    }
    mywindow = (GDWindow *) window;

    if ( mywindow->bindings.cferbind != NULL ) {
        success = mywindow->bindings.cferbind->
                    setWidthFactor(mywindow->bindings.cferbind, (double) widthfactor);
        if ( ! success ) {
            /* grdelerrmsg already assigned */
            return 0;
        }
    }
    else if ( mywindow->bindings.pyobject != NULL ) {
        result = PyObject_CallMethod(mywindow->bindings.pyobject, "setWidthFactor",
                                     "d", (double) widthfactor);
        if ( result == NULL ) {
            sprintf(grdelerrmsg, "grdelWindowSetWidthFactor: error when calling the "
                    "Python binding's setWidthFactor method: %s", pyefcn_get_error());
            return 0;
        }
        Py_DECREF(result);
    }
    else {
        strcpy(grdelerrmsg, "grdelWindowSetWidthFactor: unexpected error, "
                            "no bindings associated with this Window");
        return 0;
    }

    return 1;
}

/*
 * Creates and returns a Window object.
 * Initializes the graphics engine if needed.
 *
 * Input arguments:
 *     engine: name of the graphics engine for creating the Window
 *     enginelen: actual length of the graphics engine name
 *     title: display title for the Window
 *     titlelen: actual length of the title
 *     visible: display Window on start-up? If zero, no; if non-zero, yes.
 *     noalpha: do not use the alpha channel (opacity) in colors ?
 *     rasteronly: only create raster images
 * Output Arguments:
 *     window: the window object created, or zero if failure.
 *             Use fgderrmsg_ to retreive the error message.
 */
void FORTRAN(fgdwincreate)(void **window, char *engine, int *enginelen,
                   char *title, int *titlelen, int *visible,
                   int *noalpha, int *rasteronly)
{
    grdelType mywindow;

    mywindow = grdelWindowCreate(engine, *enginelen, title, *titlelen,
                                 *visible, *noalpha, *rasteronly);
    *window = mywindow;
}

/*
 * Deletes (closes and destroys) a Window created by fgdwincreate_
 *
 * Input Arguments:
 *     window: Window to be closed
 * Output Arguments:
 *     success: non-zero if successful; zero if an error occurred.
 *              Use fgderrmsg_ to retrieve the error message.
 */
void FORTRAN(fgdwindelete)(int *success, void **window)
{
    grdelBool result;

    result = grdelWindowDelete(*window);
    *success = result;
}

/*
 * Assigns the image filename and format.  This may just be a default
 * filename when saving a window (for interactive graphics window), or
 * it may be the filename which is written as the drawing proceeds (for
 * non-interactive "batch mode" graphics without a display window).
 *
 * If formatname is empty or NULL, the format is quessed for the
 * filename extension of imagename.
 *
 * Input Arguments:
 *     window:     Window to use
 *     imagename:  filename for saving the image
 *     imgnamelen: actual length of imagename
 *     formatname: name of the format (case insensitive);
 *                 eg, "PNG", "PDF", "PS".  May be NULL.
 *     fmtnamelen: actual lenght of formatname (zero if NULL)
 * Output Arguments:
 *     success: non-zero if successful; zero if an error occurred.
 *              Use fgderrmsg_ to retrieve the error message.
 */
void FORTRAN(fgdwinimgname)(int *success, void **window, char *imagename,
                    int *imgnamelen, char *formatname, int *fmtnamelen)
{
    grdelBool result;

    result = grdelWindowSetImageName(*window, imagename, *imgnamelen,
                                     formatname, *fmtnamelen);
    *success = result;
}

/*
 * Clears the window of all drawings.  The window is filled
 * (initialized) with bkgcolor; i.e., the background color.
 *
 * Input Arguments:
 *     window: Window to be cleared
 *     bkgcolor: Color to fill (initialize) the scene
 * Output Arguments:
 *     success: non-zero if successful; zero if an error occurred.
 *              Use fgderrmsg_ to retrieve the error message.
 */
void FORTRAN(fgdwinclear)(int *success, void **window, void **bkgcolor)
{
    grdelBool result;

    result = grdelWindowClear(*window, *bkgcolor);
    *success = result;
}

/*
 * Redraws the current drawing with bkgcolor as the background Color.
 *
 * Input Arguments:
 *     window: Window to be cleared
 *     bkgcolor: Color to fill (initialize) the scene
 *               prior to redrawing the scene.
 * Output Arguments:
 *     success: non-zero if successful; zero if an error occurred.
 *              Use fgderrmsg_ to retrieve the error message.
 */
void FORTRAN(fgdwinredraw)(int *success, void **window, void **bkgcolor)
{
    grdelBool result;

    result = grdelWindowRedraw(*window, *bkgcolor);
    *success = result;
}

/*
 * Updates a Window, adding any new drawing elements and redrawing.
 *
 * Input Arguments:
 *     window: Window to be updated
 * Output Arguments:
 *     success: non-zero if successful; zero if an error occurred.
 *              Use fgderrmsg_ to retrieve the error message.
 */
void FORTRAN(fgdwinupdate)(int *success, void **window)
{
    grdelBool result;

    result = grdelWindowUpdate(*window);
    *success = result;
}

/*
 * Turns on or off anti-aliasing on items drawn after this call.
 *
 * Input Arguments:
 *     window: Window to use
 *     antialias: if zero, turn off antialiasing;
 *                otherwise, turn on antialiasing
 * Output Arguments:
 *     success: non-zero if successful; zero if an error occurred.
 *              Use fgderrmsg_ to retrieve the error message.
 */
void FORTRAN(fgdwinsetantialias)(int *success, void **window, int *antialias)
{
    grdelBool result;

    result = grdelWindowSetAntialias(*window, *antialias);
    *success = result;
}

/*
 * Sets the current size of a Window.
 *
 * Input Arguments:
 *     window: Window to use
 *     width: width of the Window, in "device units"
 *     height: height of the window in "device units"
 *   "device units" is pixels at the current window DPI.
 * Output Arguments:
 *     success: non-zero if successful; zero if an error occurred.
 *              Use fgderrmsg_ to retrieve the error message.
 */
void FORTRAN(fgdwinsetsize)(int *success, void **window, float *width, float *height)
{
    grdelBool result;

    result = grdelWindowSetSize(*window, *width, *height);
    *success = result;
}

/*
 * Sets the scaling factor of a Window.
 *
 * Input Arguments:
 *     window: Window to use
 *     scale: scaling factor for the Window
 * Output Arguments:
 *     success: non-zero if successful; zero if an error occurred.
 *              Use fgderrmsg_ to retrieve the error message.
 */
void FORTRAN(fgdwinsetscale)(int *success, void **window, float *scale)
{
    grdelBool result;

    result = grdelWindowSetScale(*window, *scale);
    *success = result;
}

/*
 * Display or hide a Window.  A graphics engine that does not
 * have the ability to display a Window will ignore the call.
 *
 * Input Arguments:
 *     window: Window to use
 *     visible: non-zero to show the window; zero to hide the window
 * Output Arguments:
 *     success: non-zero if successful; zero if an error occurred.
 *              Use fgderrmsg_ to retrieve the error message.
 */
void FORTRAN(fgdwinsetvis)(int *success, void **window, int *visible)
{
    grdelBool result;

    result = grdelWindowSetVisible(*window, *visible);
    *success = result;
}

/*
 * Save the contents of the window to a file.
 *
 * Input Arguments:
 *     window: Window to use
 *     filename: name of the file to create
 *     filenamelen: actual length of the file name
 *     fileformat: name of the format to use
 *     formatlen: actual length of the format name
 *     transparentbkg: make the background transparent?
 *     xinches: horizontal size of vector image in inches
 *     yinches: vertical size of vector image in inches
 *     xpixels: horizontal size of raster image in pixels
 *     ypixels: vertical size of raster image in pixels
 *     firststr: ferret memory pointer to the first annotation C string;
 *               pointers are always 8 bytes apart
 *     numstr: number of annotation C strings
 *
 * If formatlen is zero, the fileformat is guessed from the
 * filename extension.
 *
 * Output Arguments:
 *     success: non-zero if successful; zero if an error occurred.
 *              Use fgderrmsg_ to retrieve the error message.
 */
void FORTRAN(fgdwinsave)(int *success, void **window, char *filename, int *namelen,
                 char *fileformat, int *formatlen, int *transparentbkg,
                 float *xinches, float *yinches, int *xpixels, int *ypixels,
                 void **firststr, int *numstr)
{
    grdelBool result;

    result = grdelWindowSave(*window, filename, *namelen,
                             fileformat, *formatlen, *transparentbkg,
                             *xinches, *yinches, *xpixels, *ypixels,
                             firststr, *numstr);
    *success = result;
}


/*
 * Get information about the default screen (display) for a window.
 *
 * Input Arguments:
 *     window: Window to use
 *
 * Output Arguments:
 *     dpix: the number of dots per inch in the horizontal (X) direction
 *     dpiy: the number of dots per inch in the vertical (Y) direction
 *     screenwidth: width of the screen (display) in pixels (dots)
 *     screenheight: height of the screen (display) in pixels (dots)
 *     success: non-zero if successful; zero if an error occurred.
 *              Use fgderrmsg_ to retrieve the error message.
 */
void FORTRAN(fgdwinscreeninfo)(int *success, void **window, float *dpix, float *dpiy,
                                     int *screenwidth, int*screenheight)
{
    grdelBool result;

    result = grdelWindowScreenInfo(*window, dpix, dpiy,
                                   screenwidth, screenheight);
    *success = result;
}


/*
 * Assign the window DPI.
 * Will only be successful if the window is not associated with a display.
 *
 * Input Arguments:
 *     window: Window to use
 *     newdpi: the number of dots per inch to assign
 *
 * Output Arguments:
 *     success: one if successful,
 *              zero if an error occurred (use fgderrmsg_ to retrieve the error message),
 *              negative one if the window is associated with a display
 */
void FORTRAN(fgdwinsetdpi)(int *success, void **window, float *newdpi)
{
    int result;

    result = grdelWindowSetDpi(*window, *newdpi);
    *success = result;
}


/*
 * Assign the scaling factor for line widths, symbol sizes, and font sizes.
 *
 * Input Arguments:
 *     window: Window to use
 *     widthfactor: scaling factor to use
 *
 * Output Arguments:
 *     success: non-zero if successful; zero if an error occurred.
 *              Use fgderrmsg_ to retrieve the error message.
 */
void FORTRAN(fgdwinsetwidthfactor)(int *success, void **window, float *widthfactor)
{
    grdelBool result;

    result = grdelWindowSetWidthFactor(*window, *widthfactor);
    *success = result;
}


/*
 * Starts a View in a Window.  "View" refers to a rectangular subsection of
 * the Window (possibly the complete canvas of the Window).
 *
 * Arguments:
 *     window: Window object to use
 *     leftfrac: location of the left side of the View as a fraction
 *             [0.0, 1.0] of the total width of the Window
 *     bottomfrac: location of the bottom of the View as a fraction
 *             [0.0, 1.0] of the total height of the Window
 *     rightfrac: location of the right side of the View as a fraction
 *             [0.0, 1.0] of the total width of the Window
 *     topfrac: location of the top of the View as a fraction
 *             [0.0, 1.0] of the total height of the Window
 *     clipit: clip drawing to this View?
 *
 * The Window fractions start at the bottom left corner and increase
 * to the top right corner; thus rightfrac must be larger than leftfrac,
 * and topfrac must be larger than bottomfrac.
 *
 * Returns success (nonzero) or failure (zero).
 * If failure, grdelerrmsg contains an explanatory message.
 */
grdelBool grdelWindowViewBegin(grdelType window,
                               float leftfrac, float bottomfrac,
                               float rightfrac, float topfrac,
                               grdelBool clipit)
{
    GDWindow *mywindow;
    grdelBool success;
    PyObject *clipbool;
    PyObject *result;

#ifdef GRDELDEBUG
    fprintf(debuglogfile, "grdelWindowViewBegin called: "
            "window = %p, "
            "viewfrac  = (%f, %f, %f, %f) "
            "clipit = %d\n",
            window, leftfrac, bottomfrac, rightfrac, topfrac, clipit);
    fflush(debuglogfile);
#endif

    if ( grdelWindowVerify(window) == NULL ) {
        strcpy(grdelerrmsg, "grdelWindowViewBegin: window argument is not "
                            "a grdel Window");
        return 0;
    }
    mywindow = (GDWindow *) window;
    if ( mywindow->hasview ) {
        strcpy(grdelerrmsg, "grdelWindowViewBegin: window "
                            "already has a View defined");
        return 0;
    }

    if ( mywindow->bindings.cferbind != NULL ) {
        success = mywindow->bindings.cferbind->
                            beginView(mywindow->bindings.cferbind,
                                      (double) leftfrac, 1.0 - (double) bottomfrac,
                                      (double) rightfrac, 1.0 - (double) topfrac,
                                      clipit);
        if ( ! success ) {
            /* grdelerrmsg already assigned */
            return 0;
        }
    }
    else if ( mywindow->bindings.pyobject != NULL ) {
        if ( clipit )
            clipbool = Py_True;
        else
            clipbool = Py_False;
        result = PyObject_CallMethod(mywindow->bindings.pyobject, "beginView", "ddddO",
                                     (double) leftfrac, 1.0 - (double) bottomfrac,
                                     (double) rightfrac, 1.0 - (double) topfrac,
                                     clipbool);
        if ( result == NULL ) {
            sprintf(grdelerrmsg, "grdelWindowViewBegin: Error when calling the "
                    "Python binding's beginView method: %s", pyefcn_get_error());
            return 0;
        }
        Py_DECREF(result);
    }
    else {
        strcpy(grdelerrmsg, "grdelWindowViewBegin: unexpected error, "
                            "no bindings associated with this Window");
        return 0;
    }

    mywindow->hasview = 1;
    return 1;
}

/*
 * Enable or disable clipping to the current View.
 *
 * Arguments:
 *     clipit: clip to the current View?
 *
 * Returns success (nonzero) or failure (zero).
 * If failure, grdelerrmsg contains an explanatory message.
 */
grdelBool grdelWindowViewClip(grdelType window, grdelBool clipit)
{
    GDWindow *mywindow;
    grdelBool success;
    PyObject *clipbool;
    PyObject *result;

#ifdef GRDELDEBUG
    fprintf(debuglogfile, "grdelWindowViewClip called: "
            "window = %p "
            "clipit = %d\n",
            window, clipit);
    fflush(debuglogfile);
#endif

    if ( grdelWindowVerify(window) == NULL ) {
        strcpy(grdelerrmsg, "grdelWindowViewClip: window argument is not "
                            "a grdel Window");
        return 0;
    }
    mywindow = (GDWindow *) window;
    if ( ! mywindow->hasview ) {
        strcpy(grdelerrmsg, "grdelWindowViewClip: window does not "
                            "have a view defined");
        return 0;
    }

    if ( mywindow->bindings.cferbind != NULL ) {
        success = mywindow->bindings.cferbind->
                            clipView(mywindow->bindings.cferbind, clipit);
        if ( ! success ) {
            /* grdelerrmsg already assigned */
            return 0;
        }
    }
    else if ( mywindow->bindings.pyobject != NULL ) {
        if ( clipit )
            clipbool = Py_True;
        else
            clipbool = Py_False;
        result = PyObject_CallMethod(mywindow->bindings.pyobject, "clipView",
                                     "O", clipbool);
        if ( result == NULL ) {
            sprintf(grdelerrmsg, "grdelWindowViewClip: error when calling the "
                    "Python binding's clipView method: %s", pyefcn_get_error());
            return 0;
        }
        Py_DECREF(result);
    }
    else {
        strcpy(grdelerrmsg, "grdelWindowViewClip: unexpected error, "
                            "no bindings associated with this Window");
        return 0;
    }

    return 1;
}

/*
 * End a View created by grdelViewBegin.
 *
 * Arguments:
 *     window: Window on which the view was defined
 *
 * Returns success (nonzero) or failure (zero).
 * If failure, grdelerrmsg contains an explanatory message.
 */
grdelBool grdelWindowViewEnd(grdelType window)
{
    GDWindow *mywindow;
    grdelBool success;
    PyObject *result;

#ifdef GRDELDEBUG
    fprintf(debuglogfile, "grdelWindowViewEnd called: "
            "window = %p\n", window);
    fflush(debuglogfile);
#endif

    if ( grdelWindowVerify(window) == NULL ) {
        strcpy(grdelerrmsg, "grdelWindowViewEnd: window argument is not "
                            "a grdel Window");
        return 0;
    }
    mywindow = (GDWindow *) window;
    if ( ! mywindow->hasview ) {
        strcpy(grdelerrmsg, "grdelWindowViewEnd: window does not "
                            "have a view defined");
        return 0;
    }

    if ( mywindow->bindings.cferbind != NULL ) {
        success = mywindow->bindings.cferbind->
                            endView(mywindow->bindings.cferbind);
        if ( ! success ) {
            /* grdelerrmsg already assigned */
            return 0;
        }
    }
    else if ( mywindow->bindings.pyobject != NULL ) {
        result = PyObject_CallMethod(mywindow->bindings.pyobject, "endView", NULL);
        if ( result == NULL ) {
            sprintf(grdelerrmsg, "grdelWindowViewEnd: error when calling the "
                    "Python binding's endView method: %s", pyefcn_get_error());
            return 0;
        }
        Py_DECREF(result);
    }
    else {
        strcpy(grdelerrmsg, "grdelWindowViewEnd: unexpected error, "
                            "no bindings associated with this Window");
        return 0;
    }

    mywindow->hasview = 0;
    return 1;
}

/*
 * Starts a View in a Window.  "View" refers to a rectangular subsection of
 * the Window (possibly the complete canvas of the Window).
 *
 * Input Arguments:
 *     window: Window object to use
 *     leftfrac: location of the left side of the View as a fraction
 *             [0.0, 1.0] of the total width of the Window
 *     bottomfrac: location of the bottom of the View as a fraction
 *             [0.0, 1.0] of the total height of the Window
 *     rightfrac: location of the right side of the View as a fraction
 *             [0.0, 1.0] of the total width of the Window
 *     topfrac: location of the top of the View as a fraction
 *             [0.0, 1.0] of the total height of the Window
 *     clipit: clip drawing to this View? (zero: no, non-zero: yes)
 *
 * The Window fractions start at the bottom left corner and increase
 * to the top right corner; thus rightfrac must be larger than leftfrac,
 * and topfrac must be larger than bottomfrac.
 *
 * Output Arguments:
 *     success: non-zero if successful; zero if an error occurred.
 *              Use fgderrmsg_ to retrieve the error message.
 */
void FORTRAN(fgdviewbegin)(int *success, void **window,
                   float *leftfrac, float *bottomfrac,
                   float *rightfrac, float *topfrac,
                   int *clipit)
{
    grdelBool result;

    result = grdelWindowViewBegin(*window, *leftfrac, *bottomfrac,
                                  *rightfrac, *topfrac, *clipit);
    *success = result;
}

/*
 * Enable or disable clipping to the current View.
 *
 * Input Arguments:
 *     clipit: clip to the current View? (zero: no, non-zero: yes)
 * Output Arguments:
 *     success: non-zero if successful; zero if an error occurred.
 *              Use fgderrmsg_ to retrieve the error message.
 */
void FORTRAN(fgdviewclip)(int *success, void **window, int *clipit)
{
    grdelBool result;

    result = grdelWindowViewClip(*window, *clipit);
    *success = result;
}

/*
 * End a View created by fgdviewbegin_
 *
 * Input Arguments:
 *     window: Window on which the view was defined
 * Output Arguments:
 *     success: non-zero if successful; zero if an error occurred.
 *              Use fgderrmsg_ to retrieve the error message.
 */
void FORTRAN(fgdviewend)(int *success, void **window)
{
    grdelBool result;

    result = grdelWindowViewEnd(*window);
    *success = result;
}

/*
 * Starts a Segment in a Window.
 * A "Segment" is a group of drawing commands.
 *
 * Input Arguments:
 *     window: Window object to use
 *     segid: ID for the Segment
 * Output Arguments:
 *     success: non-zero if successful; zero if an error occurred.
 *              Use fgderrmsg_ to retrieve the error message.
 */
grdelBool grdelWindowSegmentBegin(grdelType window, int segid)
{
    GDWindow *mywindow;
    grdelBool success;
    PyObject *result;

#ifdef GRDELDEBUG
    fprintf(debuglogfile, "grdelWindowSegmentBegin called: "
            "window = %p, "
            "segid = %d\n",
            window, segid);
    fflush(debuglogfile);
#endif

    if ( grdelWindowVerify(window) == NULL ) {
        strcpy(grdelerrmsg, "grdelWindowSegmentBegin: window argument "
                            "is not a grdel Window");
        return 0;
    }
    mywindow = (GDWindow *) window;
    if ( mywindow->hasseg ) {
        strcpy(grdelerrmsg, "grdelWindowSegmentBegin: window "
                            "already has a Segment defined");
        return 0;
    }

    if ( mywindow->bindings.cferbind != NULL ) {
        success = mywindow->bindings.cferbind->
                            beginSegment(mywindow->bindings.cferbind, segid);
        if ( ! success ) {
            /* grdelerrmsg already assigned */
            return 0;
        }
    }
    else if ( mywindow->bindings.pyobject != NULL ) {
        result = PyObject_CallMethod(mywindow->bindings.pyobject,
                                     "beginSegment", "i", segid);
        if ( result == NULL ) {
            sprintf(grdelerrmsg, "grdelWindowSegmentBegin: Error when calling the "
                    "Python binding's beginSegment method: %s", pyefcn_get_error());
            return 0;
        }
        Py_DECREF(result);
    }
    else {
        strcpy(grdelerrmsg, "grdelWindowSegmentBegin: unexpected error, "
                            "no bindings associated with this Window");
        return 0;
    }

    mywindow->hasseg = 1;
    return 1;
}

/*
 * Ends the current Segment in a Window.
 *
 * Arguments:
 *     window: Window on which the segment was defined
 *
 * Returns success (nonzero) or failure (zero).
 * If failure, grdelerrmsg contains an explanatory message.
 */
grdelBool grdelWindowSegmentEnd(grdelType window)
{
    GDWindow *mywindow;
    grdelBool success;
    PyObject *result;

#ifdef GRDELDEBUG
    fprintf(debuglogfile, "grdelWindowSegmentEnd called: "
            "window = %p\n", window);
    fflush(debuglogfile);
#endif

    if ( grdelWindowVerify(window) == NULL ) {
        strcpy(grdelerrmsg, "grdelWindowSegmentEnd: window argument "
                            "is not a grdel Window");
        return 0;
    }
    mywindow = (GDWindow *) window;
    if ( ! mywindow->hasseg ) {
        strcpy(grdelerrmsg, "grdelWindowSegmentEnd: window does not "
                            "have a segment defined");
        return 0;
    }

    if ( mywindow->bindings.cferbind != NULL ) {
        success = mywindow->bindings.cferbind->
                            endSegment(mywindow->bindings.cferbind);
        if ( ! success ) {
            /* grdelerrmsg already assigned */
            return 0;
        }
    }
    else if ( mywindow->bindings.pyobject != NULL ) {
        result = PyObject_CallMethod(mywindow->bindings.pyobject, "endSegment", NULL);
        if ( result == NULL ) {
            sprintf(grdelerrmsg, "grdelWindowSegmentEnd: error when calling the "
                    "Python binding's endSegment method: %s", pyefcn_get_error());
            return 0;
        }
        Py_DECREF(result);
    }
    else {
        strcpy(grdelerrmsg, "grdelWindowSegmentEnd: unexpected error, "
                            "no bindings associated with this Window");
        return 0;
    }

    mywindow->hasseg = 0;
    return 1;
}

/*
 * Deletes the drawing commands in the indicated Segment of a Window.
 *
 * Arguments:
 *     window: Window on which the segment was defined
 *     segid: ID for the Segment to delete
 *
 * Returns success (nonzero) or failure (zero).
 * If failure, grdelerrmsg contains an explanatory message.
 */
grdelBool grdelWindowSegmentDelete(grdelType window, int segid)
{
    GDWindow *mywindow;
    grdelBool success;
    PyObject *result;

#ifdef GRDELDEBUG
    fprintf(debuglogfile, "grdelWindowSegmentDelete called: "
            "window = %p, "
            "segid = %d\n",
            window, segid);
    fflush(debuglogfile);
#endif

    if ( grdelWindowVerify(window) == NULL ) {
        strcpy(grdelerrmsg, "grdelWindowSegmentDelete: window argument "
                            "is not a grdel Window");
        return 0;
    }
    mywindow = (GDWindow *) window;

    if ( mywindow->bindings.cferbind != NULL ) {
        success = mywindow->bindings.cferbind->
                            deleteSegment(mywindow->bindings.cferbind, segid);
        if ( ! success ) {
            /* grdelerrmsg already assigned */
            return 0;
        }
    }
    else if ( mywindow->bindings.pyobject != NULL ) {
        result = PyObject_CallMethod(mywindow->bindings.pyobject,
                                     "deleteSegment", "i", segid);
        if ( result == NULL ) {
            sprintf(grdelerrmsg, "grdelWindowSegmentDelete: error when calling the "
                    "Python binding's deleteSegment method: %s", pyefcn_get_error());
            return 0;
        }
        Py_DECREF(result);
    }
    else {
        strcpy(grdelerrmsg, "grdelWindowSegmentDelete: unexpected error, "
                            "no bindings associated with this Window");
        return 0;
    }

    mywindow->hasseg = 0;
    return 1;
}

grdelBool grdelWindowSetWmark(grdelType window, char *filename, int len_filename,
                              float xloc, float yloc, float scalefrac, float opacity)
{
    GDWindow *mywindow;
    grdelBool success;
    PyObject *result;

#ifdef GRDELDEBUG
    fprintf(debuglogfile, "grdelWindowSetWmark called: "
            "window = %p, "
            "filename = %*c, "
            "xloc = %f, "
            "yloc = %f, "
            "scalefrac = %f, "
            "opacity = %f\n",
            window, len_filename, filename, xloc, yloc, scalefrac, opacity);
    fflush(debuglogfile);
#endif

    if ( grdelWindowVerify(window) == NULL ) {
        strcpy(grdelerrmsg, "grdelWindowSetWmark: window argument "
                            "is not a grdel Window");
        return 0;
    }
    mywindow = (GDWindow *) window;

    if ( mywindow->bindings.cferbind != NULL ) {
        success = mywindow->bindings.cferbind->
                            setWaterMark(mywindow->bindings.cferbind, filename, len_filename, xloc, yloc, scalefrac, opacity);
        if ( ! success ) {
            /* grdelerrmsg already assigned */
            return 0;
        }
    }
    else if ( mywindow->bindings.pyobject != NULL ) {
        result = PyObject_CallMethod(mywindow->bindings.pyobject,
                                     "setWaterMark", "s#iffff", filename, len_filename, xloc, yloc, scalefrac, opacity);
        if ( result == NULL ) {
            sprintf(grdelerrmsg, "grdelWindowSetWmark: Error when calling the "
                    "Python binding's setWaterMark method: %s", pyefcn_get_error());
            return 0;
        }
        Py_DECREF(result);
    }
    else {
        strcpy(grdelerrmsg, "grdelWindowSetWmark: unexpected error, "
                            "no bindings associated with this Window");
        return 0;
    }

    return 1;
}

/*
 * Start a Segment in a Window.
 * A "Segment" is a group of drawing commands.
 *
 * Input Arguments:
 *     window: Window object to use
 *     segid: ID for the Segment
 * Output Arguments:
 *     success: non-zero if successful; zero if an error occurred.
 *              Use fgderrmsg_ to retrieve the error message.
 */
void FORTRAN(fgdsegbegin)(int *success, void **window, int *segid)
{
    grdelBool result;

    result = grdelWindowSegmentBegin(*window, *segid);
    *success = result;
}

/*
 * Ends the current Segment in a Window.
 *
 * Input Arguments:
 *     window: Window object to use
 * Output Arguments:
 *     success: non-zero if successful; zero if an error occurred.
 *              Use fgderrmsg_ to retrieve the error message.
 */
void FORTRAN(fgdsegend)(int *success, void **window)
{
    grdelBool result;

    result = grdelWindowSegmentEnd(*window);
    *success = result;
}

/*
 * Deletes the drawing commands in the indicated Segment of a Window.
 *
 * Input Arguments:
 *     window: Window object to use
 *     segid: ID for the Segment
 * Output Arguments:
 *     success: non-zero if successful; zero if an error occurred.
 *              Use fgderrmsg_ to retrieve the error message.
 */
void FORTRAN(fgdsegdelete)(int *success, void **window, int *segid)
{
    grdelBool result;

    result = grdelWindowSegmentDelete(*window, *segid);
    *success = result;
}

/*
 * Sets watermark image as contents of image specified by filename.
 *
 * Input Arguments:
 *     window: Window object to use
 *     filename: path to watermark image
 *     len_filename: number of characters in filename string
 *     xloc: horizontal position of watermark on final image
 *     yloc: vertical position of watermakr on final image
 *     scalefrac: proportion of displayed image to real image
 *     opacity: percentage of transparency to display watermark with
 * Output Arguments:
 *     success: non-zero if successful; zero if an error occurred.
 *              Use fgderrmsg_ to retrieve the error message.
 */
void FORTRAN(fgdwinsetwmark)(int *success, void **window, char *filename, int *len_filename,
                             float *xloc, float *yloc, float *scalefrac, float *opacity)
{
    grdelBool result;

    result = grdelWindowSetWmark(*window, filename, *len_filename, *xloc, *yloc, *scalefrac, *opacity);
    *success = result;
}
