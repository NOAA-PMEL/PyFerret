/*
 * "Window" refers to the full canvas; however, no drawing, except
 * possibly clearing the window, is done on the full window.  Instead,
 * a "View" of the Window is "begun", and drawing is done in that View
 * of the Window.
 *
 * "View" refers to a rectangular subsection of the Window (possibly
 * the complete canvas of the Window).  A View has its own coordinate
 * system where the longer side has coordinates [0, 1000], and the
 * shorter side has coordinates [0, N] where N is such that the aspect
 * ratio of the Window is maintained.
 *
 * In order to draw in a Window, a View must first have been specified
 * using grdelViewBegin.  All coordinates and sizes in drawing methods
 * use View coordinates with the one exception of Font size in points.
 * When drawing in a View is complete, grdelViewEnd is called, at which
 * point the Window will be updated.
 *
 * Only one View can be active at any time.  So a switch between views
 * requires ending one view and beginning a another view.
 */
#include <Python.h> /* make sure Python.h is first */
#include <string.h>
#include "grdel.h"
#include "pyferret.h"


#ifdef VERBOSEDEBUG
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
    debuglogfile = fopen("pyferretdebug.log", "w");
    if ( debuglogfile == NULL ) {
        fputs("Open of pyferretdebug.log failed", stderr);
        fflush(stderr);
        exit(1);
    }
    atexit(closelogfile);
}
#endif /* VERBOSEDEBUG */


static const char *grdelwindowid = "GRDEL_WINDOW";

/* Instantiate the global error message string */
char grdelerrmsg[2048];

typedef struct GDWindow_ {
    const char *id;
    PyObject *bindings;
    grdelBool hasview;
} GDWindow;

/*
 * Copies of the error message in grdelerrmsg to errmsg, but will not null
 * terminated.  The argument errmsg should be at least 2048 characters in
 * length and initialized to all blank prior to calling this function for
 * return of a proper Fortran string.  The argument errmsglen is assigned
 * the actual length of the message returned in errmsg.
 */
void fgderrmsg_(char *errmsg, int *errmsglen)
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
 *     width: width of the Window, in 0.001 inches
 *     height: height of the Window, in 0.001 inches
 *     visible: display Window on start-up?
 *
 * Returns a pointer to the window object created.
 * If an error occurs, NULL is returned and
 * grdelerrmsg contains an explanatory message.
 */
grdelType grdelWindowCreate(const char *engine, int enginelen,
               const char *title, int titlelen, float width,
               float height, grdelBool visible)
{
    GDWindow *window;
    PyObject *modulename;
    PyObject *module;
    PyObject *visiblebool;

    /* Allocate memory for this GDWindow */
    window = (GDWindow *) PyMem_Malloc(sizeof(GDWindow));
    if ( window == NULL ) {
        strcpy(grdelerrmsg, "grdelWindowCreate: out of memory for a new Window");
        return NULL;
    }
    window->id = grdelwindowid;
    window->hasview = 0;

    /*
     * Call pyferret.graphbind.createWindow, which will create (and return)
     * an instance of the bindings for this graphics engine and then call
     * the createWindow method of that bindings instance to create the window.
     */
    modulename = PyString_FromString("pyferret.graphbind");
    if ( modulename == NULL ) {
        PyErr_Clear();
        strcpy(grdelerrmsg, "grdelWindowCreate: problems creating "
               "a Python string from the module name: pyferret.graphbind");
        PyMem_Free(window);
        return NULL;
    }
    module = PyImport_Import(modulename);
    Py_DECREF(modulename);
    if ( module == NULL ) {
        PyErr_Clear();
        strcpy(grdelerrmsg, "grdelWindowCreate: unable to import module: "
                            "pyferret.graphbind");
        PyMem_Free(window);
        return NULL;
    }
    if ( visible )
        visiblebool = Py_True;
    else
        visiblebool = Py_False;
    window->bindings = PyObject_CallMethod(module, "createWindow", "s#s#ddO",
                                engine, enginelen, title, titlelen,
                                (double) width, (double) height, visiblebool);
    Py_DECREF(module);
    if ( window->bindings == NULL ) {
        sprintf(grdelerrmsg, "grdelWindowCreate: error when calling createWindow "
                             "in pyferret.graphbind: %s", pyefcn_get_error());
        PyMem_Free(window);
        return NULL;
    }

#ifdef VERBOSEDEBUG
    /* 
     * Since a window has to be created before anything else can happen
     * the initialization check of debuglogfile only needs to happen here.
     */
    if ( debuglogfile == NULL )
        openlogfile();
    fprintf(debuglogfile, "grdelWindow created: "
            "window = %X\n", window);
    fflush(debuglogfile);
#endif

    /* return the pointer to the GDWindow */
    grdelerrmsg[0] = '\0';
    return window;
}


/*
 * Verifies the argument is a grdel Window.
 * Returns the graphic engine's window bindings
 * object if it is.  Returns NULL if it is not.
 */
PyObject *grdelWindowVerify(grdelType window)
{
    GDWindow *mywindow;

    if ( window == NULL )
        return NULL;
    mywindow = (GDWindow *) window;
    if ( mywindow->id != grdelwindowid )
        return NULL;
    return mywindow->bindings;
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
    PyObject *modulename;
    PyObject *module;
    PyObject *result;

#ifdef VERBOSEDEBUG
    fprintf(debuglogfile, "grdelWindowDelete called: "
            "window = %X\n", window);
    fflush(debuglogfile);
#endif


    if ( grdelWindowVerify(window) == NULL ) {
        strcpy(grdelerrmsg, "grdelWindowDelete: window argument is not "
                            "a grdel Window");
        return (grdelBool) 0;
    }
    mywindow = (GDWindow *) window;

    /* if a view is still active, end it */
    if ( mywindow->hasview ) {
        if ( ! grdelWindowViewEnd(window) ) {
            /* grdelerrmsg already assigned */
            return (grdelBool) 0;
        }
    }

    /*
     * Call the deleteWindow method of the bindings instance.
     * If True is returned, delete (decrement the reference to)
     * this bindings instance and return True.
     */
    result = PyObject_CallMethod(mywindow->bindings, "deleteWindow", NULL);
    if ( result == NULL ) {
        sprintf(grdelerrmsg, "grdelWindowDelete: error when calling "
                "the binding's deleteWindow method: %s", pyefcn_get_error());
        return (grdelBool) 0;
    }
    Py_DECREF(result);
    /* Py_True and Py_False are singleton objects */
    if ( result != Py_True ) {
        sprintf(grdelerrmsg, "grdelWindowDelete: deleteWindow method "
                             "returned False");
        return (grdelBool) 0;
    }
    Py_DECREF(mywindow->bindings);

    /* Free the memory for the GDWindow */
    mywindow->id = NULL;
    mywindow->hasview = 0;
    mywindow->bindings = NULL;
    PyMem_Free(window);

    grdelerrmsg[0] = '\0';
    return (grdelBool) 1;
}

/*
 * Clears the window of all drawings.  The window is filled
 * (initialized) with fillcolor.
 *
 * Arguments:
 *     window: Window to be cleared
 *     fillcolor: Color to fill (initialize) the scene
 *
 * Returns success (nonzero) or failure (zero).
 * If failure, grdelerrmsg contains an explanatory message.
 */
grdelBool grdelWindowClear(grdelType window, grdelType fillcolor)
{
    GDWindow *mywindow;
    PyObject *colorobj;
    PyObject *result;

#ifdef VERBOSEDEBUG
    fprintf(debuglogfile, "grdelWindowClear called: "
            "window = %X, fillcolor = %X\n", window, fillcolor);
    fflush(debuglogfile);
#endif

    if ( grdelWindowVerify(window) == NULL ) {
        strcpy(grdelerrmsg, "grdelWindowClear: window argument is not "
                            "a grdel Window");
        return (grdelBool) 0;
    }
    mywindow = (GDWindow *) window;
    colorobj = grdelColorVerify(fillcolor, window);
    if ( colorobj == NULL ) {
    	strcpy(grdelerrmsg, "grdelWindowClear: fillcolor argument is not "
    	                    "a valid grdel Color for the window");
    	return (grdelBool) 0;
    }

    result = PyObject_CallMethod(mywindow->bindings, "clearWindow", "O",
                                 colorobj);
    if ( result == NULL ) {
        sprintf(grdelerrmsg, "grdelWindowClear: Error when calling "
                "the binding's clearWindow method: %s", pyefcn_get_error());
        return (grdelBool) 0;
    }
    Py_DECREF(result);

    grdelerrmsg[0] = '\0';
    return (grdelBool) 1;
}

/*
 * Sets the current size of a Window.
 *
 * Arguments:
 *     window: Window to use
 *     width: width of the Window, in units of 0.001 inches
 *     height: height of the window in units of 0.001 inches
 *
 * Returns success (nonzero) or failure (zero).
 * If failure, grdelerrmsg contains an explanatory message.
 */
grdelBool grdelWindowSetSize(grdelType window, float width, float height)
{
    GDWindow *mywindow;
    PyObject *result;

#ifdef VERBOSEDEBUG
    fprintf(debuglogfile, "grdelWindowSetSize called: "
            "window = %X, width = %f, height = %f\n", window, width, height);
    fflush(debuglogfile);
#endif

    if ( grdelWindowVerify(window) == NULL ) {
        strcpy(grdelerrmsg, "grdelWindowSetSize: window argument is not "
                            "a grdel Window");
        return (grdelBool) 0;
    }
    mywindow = (GDWindow *) window;

    result = PyObject_CallMethod(mywindow->bindings, "resizeWindow", "dd",
                                 (double) width, (double) height);
    if ( result == NULL ) {
        sprintf(grdelerrmsg, "grdelWindowSetSize: error when calling "
                "the binding's resizeWindow method: %s", pyefcn_get_error());
        return (grdelBool) 0;
    }
    Py_DECREF(result);

    grdelerrmsg[0] = '\0';
    return (grdelBool) 1;
}

/*
 * Display or hide a Window.  A graphics engine that does not
 * have the ability to display a Window will always return
 * failure.
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
    PyObject *visiblebool;
    PyObject *result;

#ifdef VERBOSEDEBUG
    fprintf(debuglogfile, "grdelWindowSetVisible called: "
            "window = %X, visible = %d\n", window, visible);
    fflush(debuglogfile);
#endif

    if ( grdelWindowVerify(window) == NULL ) {
        strcpy(grdelerrmsg, "grdelWindowSetVisible: window argument is not "
                            "a grdel Window");
        return (grdelBool) 0;
    }
    mywindow = (GDWindow *) window;

    if ( visible )
        visiblebool = Py_True;
    else
        visiblebool = Py_False;

    result = PyObject_CallMethod(mywindow->bindings, "showWindow", "O",
                                 visiblebool);
    if ( result == NULL ) {
        sprintf(grdelerrmsg, "grdelWindowSetVisible: error when calling "
                "the binding's showWindow method: %s", pyefcn_get_error());
        return (grdelBool) 0;
    }
    Py_DECREF(result);

    grdelerrmsg[0] = '\0';
    return (grdelBool) 1;
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
 *
 * If fileformat is NULL, the fileformat is guessed from the
 * filename extension.
 *
 * Returns success (nonzero) or failure (zero).
 * If failure, grdelerrmsg contains an explanatory message.
 */
grdelBool grdelWindowSave(grdelType window, const char *filename,
		          int filenamelen, const char *fileformat,
                          int formatlen, int transparentbkg)
{
    GDWindow *mywindow;
    PyObject *transparentbool;
    PyObject *result;

#ifdef VERBOSEDEBUG
    fprintf(debuglogfile, "grdelWindowSave called: "
            "window = %X\n", window);
    fflush(debuglogfile);
#endif

    if ( grdelWindowVerify(window) == NULL ) {
        strcpy(grdelerrmsg, "grdleWindowSave: window argument is not "
                            "a grdel Window");
        return (grdelBool) 0;
    }
    mywindow = (GDWindow *) window;

    if ( transparentbkg == 0 )
        transparentbool = Py_False;
    else
        transparentbool = Py_True;

    result = PyObject_CallMethod(mywindow->bindings, "saveWindow", "s#s#O",
                                 filename, filenamelen,
                                 fileformat, formatlen, transparentbool);
    if ( result == NULL ) {
        sprintf(grdelerrmsg, "grdleWindowSave: error when calling "
                "the binding's saveWindow method: %s", pyefcn_get_error());
        return (grdelBool) 0;
    }
    Py_DECREF(result);

    grdelerrmsg[0] = '\0';
    return (grdelBool) 1;
}


/*
 * Get the screen resolution of a window.
 *
 * Input Arguments:
 *     window: Window to use
 *
 * Output Arguments:
 *     dpix: the number of dots per inch in the horizontal (X) direction
 *     dpiy: the number of dots per inch in the vertical (Y) direction
 *
 * Returns success (nonzero) or failure (zero).
 * If failure, grdelerrmsg contains an explanatory message.
 */
grdelBool grdelWindowDpi(grdelType window, float *dpix, float *dpiy)
{
    GDWindow *mywindow;
    PyObject *result;

#ifdef VERBOSEDEBUG
    fprintf(debuglogfile, "grdelWindowDpi called: "
            "window = %X\n", window);
    fflush(debuglogfile);
#endif

    if ( grdelWindowVerify(window) == NULL ) {
        strcpy(grdelerrmsg, "grdelWindowDpi: window argument is not a grdel Window");
        return (grdelBool) 0;
    }
    mywindow = (GDWindow *) window;

    result = PyObject_CallMethod(mywindow->bindings, "windowDpi", NULL);
    if ( result == NULL ) {
        sprintf(grdelerrmsg, "grdelWindowDpi: error when calling the binding's "
                             "windowDpi method: %s", pyefcn_get_error());
        return (grdelBool) 0;
    }

    if ( ! PyArg_ParseTuple(result, "ff", dpix, dpiy) ) {
        Py_DECREF(result);
        sprintf(grdelerrmsg, "grdelWindowDpi: Error when parsing the binding's "
                             "windowDpi return value: %s", pyefcn_get_error());
        return (grdelBool) 0;
    }
    Py_DECREF(result);
    grdelerrmsg[0] = '\0';
    return (grdelBool) 1;
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
 *     width: width of the Window, in 0.001 inches
 *     height: height of the Window, in 0.001 inches
 *     visible: display Window on start-up? If zero, no; if non-zero, yes.
 * Output Arguments:
 *     window: the window object created, or zero if failure.
 *             Use fgderrmsg_ to retreive the error message.
 */
void fgdwincreate_(void **window, char *engine, int *enginelen, char *title,
                   int *titlelen, float *width, float *height, int *visible)
{
    grdelType mywindow;

    mywindow = grdelWindowCreate(engine, *enginelen, title, *titlelen,
                                 *width, *height, *visible);
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
void fgdwindelete_(int *success, void **window)
{
    grdelBool result;

    result = grdelWindowDelete(*window);
    *success = result;
}

/*
 * Clears the window of all drawings.  The window is filled
 * (initialized) with fillcolor.
 *
 * Input Arguments:
 *     window: Window to be cleared
 *     fillcolor: Color to fill (initialize) the scene
 * Output Arguments:
 *     success: non-zero if successful; zero if an error occurred.
 *              Use fgderrmsg_ to retrieve the error message.
 */
void fgdwinclear_(int *success, void **window, void **fillcolor)
{
    grdelBool result;

    result = grdelWindowClear(*window, *fillcolor);
    *success = result;
}

/*
 * Sets the current size of a Window.
 *
 * Input Arguments:
 *     window: Window to use
 *     width: width of the Window, in units of 0.001 inches
 *     height: height of the window in units of 0.001 inches
 * Output Arguments:
 *     success: non-zero if successful; zero if an error occurred.
 *              Use fgderrmsg_ to retrieve the error message.
 */
void fgdwinsetsize_(int *success, void **window, float *width, float *height)
{
    grdelBool result;

    result = grdelWindowSetSize(*window, *width, *height);
    *success = result;
}

/*
 * Display or hide a Window.  A graphics engine that does not
 * have the ability to display a Window will fail.
 *
 * Input Arguments:
 *     window: Window to use
 *     visible: non-zero to show the window; zero to hide the window
 * Output Arguments:
 *     success: non-zero if successful; zero if an error occurred.
 *              Use fgderrmsg_ to retrieve the error message.
 */
void fgdwinsetvis_(int *success, void **window, int *visible)
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
 *
 * If formatlen is zero, the fileformat is guessed from the
 * filename extension.
 *
 * Output Arguments:
 *     success: non-zero if successful; zero if an error occurred.
 *              Use fgderrmsg_ to retrieve the error message.
 */
void fgdwinsave_(int *success, void **window, char *filename, int *namelen,
                 char *fileformat, int *formatlen, int *transparentbkg)
{
    grdelBool result;

    result = grdelWindowSave(*window, filename, *namelen,
                             fileformat, *formatlen, *transparentbkg);
    *success = result;
}


/*
 * Get the screen resolution of a window.
 *
 * Input Arguments:
 *     window: Window to use
 *
 * Output Arguments:
 *     dpix: the number of dots per inch in the horizontal (X) direction
 *     dpiy: the number of dots per inch in the vertical (Y) direction
 *     success: non-zero if successful; zero if an error occurred.
 *              Use fgderrmsg_ to retrieve the error message.
 */
void fgdwindpi_(int *success, void **window, float *dpix, float *dpiy)
{
    grdelBool result;

    result = grdelWindowDpi(*window, dpix, dpiy);
    *success = result;
}


/*
 * Starts a View in a Window.  "View" refers to a rectangular subsection of
 * the Window (possibly the complete canvas of the Window).  A View has its
 * own coordinate system (user or world coordinates).
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
 *     leftcoord: user coordinate of the left side of the View
 *     bottomcoord: user coordinate of the bottom side of the View
 *     rightcoord: user coordinate of the right side of the View
 *     topcoord: user coordinate of the top side of the View
 *
 * The Window and View coordinates start at the bottom left corner and
 * increase to the top right corner; thus rightfrac must be larger than
 * leftfrac, and topfrac must be larger than bottomfrac.  The user (world)
 * coordinates could be treated as integer values, so the values for
 * opposite sides should be significantly different.
 *
 * Returns success (nonzero) or failure (zero).
 * If failure, grdelerrmsg contains an explanatory message.
 */
grdelBool grdelWindowViewBegin(grdelType window,
                               float leftfrac, float bottomfrac,
                               float rightfrac, float topfrac,
                               float leftcoord, float bottomcoord,
                               float rightcoord, float topcoord)
{
    GDWindow *mywindow;
    PyObject *result;

#ifdef VERBOSEDEBUG
    fprintf(debuglogfile, "grdelWindowViewBegin called: "
            "window = %X, "
            "viewfrac  = (%f, %f, %f, %f)"
            "usercoord = (%f, %f, %f, %f)\n",
            window, leftfrac, bottomfrac, rightfrac, topfrac,
                    leftcoord, bottomcoord, rightcoord, topcoord);
    fflush(debuglogfile);
#endif

    if ( grdelWindowVerify(window) == NULL ) {
        strcpy(grdelerrmsg, "grdelWindowViewBegin: window argument is not "
                            "a grdel Window");
        return (grdelBool) 0;
    }
    mywindow = (GDWindow *) window;
    if ( mywindow->hasview ) {
        strcpy(grdelerrmsg, "grdelWindowViewBegin: window "
                            "already has a View defined");
        return (grdelBool) 0;
    }
    if ( (0.0 > leftfrac) || (leftfrac >= rightfrac) || (rightfrac > 1.0) ) {
        strcpy(grdelerrmsg, "grdelWindowViewBegin: leftfrac and rightfrac "
               "must be in [0.0, 1.0] with leftfrac < rightfrac");
        return (grdelBool) 0;
    }
    if ( (0.0 > bottomfrac) || (bottomfrac >= topfrac) || (topfrac > 1.0) ) {
        strcpy(grdelerrmsg, "grdelWindowViewBegin: bottomfrac and topfrac "
               "must be in [0.0, 1.0] with bottomfrac < topfrac");
        return (grdelBool) 0;
    }

    result = PyObject_CallMethod(mywindow->bindings, "beginView", "dddddddd",
                                 (double) leftfrac, (double) bottomfrac,
                                 (double) rightfrac, (double) topfrac,
                                 (double) leftcoord, (double) bottomcoord,
                                 (double) rightcoord, (double) topcoord);
    if ( result == NULL ) {
        sprintf(grdelerrmsg, "grdelWindowViewBegin: Error when calling "
                "the binding's beginView method: %s", pyefcn_get_error());
        return (grdelBool) 0;
    }
    Py_DECREF(result);

    mywindow->hasview = (grdelBool) 1;
    grdelerrmsg[0] = '\0';
    return (grdelBool) 1;
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
    PyObject *result;

#ifdef VERBOSEDEBUG
    fprintf(debuglogfile, "grdelWindowViewEnd called: "
            "window = %X\n", window);
    fflush(debuglogfile);
#endif

    if ( grdelWindowVerify(window) == NULL ) {
        strcpy(grdelerrmsg, "grdelWindowViewEnd: window argument is not "
                            "a grdel Window");
        return (grdelBool) 0;
    }
    mywindow = (GDWindow *) window;
    if ( ! mywindow->hasview ) {
        strcpy(grdelerrmsg, "grdelWindowViewEnd: window does not "
                            "have a view defined");
        return (grdelBool) 0;
    }

    result = PyObject_CallMethod(mywindow->bindings, "endView", NULL);
    if ( result == NULL ) {
        sprintf(grdelerrmsg, "grdelWindowViewEnd: error when calling "
                "the binding's endView method: %s", pyefcn_get_error());
        return (grdelBool) 0;
    }
    Py_DECREF(result);

    mywindow->hasview = (grdelBool) 0;
    grdelerrmsg[0] = '\0';
    return (grdelBool) 1;
}

/*
 * Starts a View in a Window.  "View" refers to a rectangular subsection of
 * the Window (possibly the complete canvas of the Window).  A View has its
 * own coordinate system (user or world coordinates).
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
 *     leftcoord: user coordinate of the left side of the View
 *     bottomcoord: user coordinate of the bottom side of the View
 *     rightcoord: user coordinate of the right side of the View
 *     topcoord: user coordinate of the top side of the View
 *
 * The Window and View coordinates start at the bottom left corner and
 * increase to the top right corner; thus rightfrac must be larger than
 * leftfrac, and topfrac must be larger than bottomfrac.
 *
 * Output Arguments:
 *     success: non-zero if successful; zero if an error occurred.
 *              Use fgderrmsg_ to retrieve the error message.
 */
void fgdviewbegin_(int *success, void **window,
                   float *leftfrac, float *bottomfrac,
                   float *rightfrac, float *topfrac,
                   float *leftcoord, float *bottomcoord,
                   float *rightcoord, float *topcoord)
{
    grdelBool result;

    result = grdelWindowViewBegin(*window,
                  *leftfrac, *bottomfrac, *rightfrac, *topfrac,
                  *leftcoord, *bottomcoord, *rightcoord, *topcoord);
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
void fgdviewend_(int *success, void **window)
{
    grdelBool result;

    result = grdelWindowViewEnd(*window);
    *success = result;
}

