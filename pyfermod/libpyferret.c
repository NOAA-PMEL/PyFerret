/*
 *  This software was developed by the Thermal Modeling and Analysis
 *  Project(TMAP) of the National Oceanographic and Atmospheric
 *  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
 *  hereafter referred to as NOAA/PMEL/TMAP.
 *
 *  Access and use of this software shall impose the following
 *  obligations and understandings on the user. The user is granted the
 *  right, without any fee or cost, to use, copy, modify, alter, enhance
 *  and distribute this software, and any derivative works thereof, and
 *  its supporting documentation for any purpose whatsoever, provided
 *  that this entire notice appears in all copies of the software,
 *  derivative works and supporting documentation.  Further, the user
 *  agrees to credit NOAA/PMEL/TMAP in any publications that result from
 *  the use of this software or in any product that includes this
 *  software. The names TMAP, NOAA and/or PMEL, however, may not be used
 *  in any advertising or publicity to endorse or promote any products
 *  or commercial entity unless specific written permission is obtained
 *  from NOAA/PMEL/TMAP. The user also understands that NOAA/PMEL/TMAP
 *  is not obligated to provide the user with any support, consulting,
 *  training or assistance of any kind with regard to the use, operation
 *  and performance of this software nor to provide the user with any
 *  updates, revisions, new versions or "bug fixes".
 *
 *  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP "AS IS" AND ANY EXPRESS
 *  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY
 *  SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
 *  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
 *  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
 *  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL pyferret_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <ctype.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ferret.h"
#include "ferret_shared_buffer.h"
#include "EF_Util.h"
#include "grdel.h"
#include "pyferret.h"
#include "pplmem.h"
#include "FerMem.h"

/* For older versions of NumPy (v1.4 with RHEL6), define this flag as the deprecated flag */
#ifndef NPY_ARRAY_OWNDATA
#define NPY_ARRAY_OWNDATA NPY_OWNDATA
#endif

/* global pyferret Python module object used for readline */
PyObject *pyferret_module_pyobject = NULL;

/* global pyferret.graphbind Python module object used for createWindow */
PyObject *pyferret_graphbind_module_pyobject = NULL;

/* Ferret's OK return status value */
#define FERR_OK 3

/* Special return value from libpyferret._run indicating the program should shut down */
#define FERR_EXIT_PROGRAM -3

/* Ferret's unspecified integer value */
#define UNSPECIFIED_INT4 -999

/* Length given to the abstract axis */
#define ABSTRACT_AXIS_LEN 9999999

/* 
 * String used as the missing value for String arrays in Python.
 * Note that this is NULL ('\0') terminated. 
 */
static char *PYTHON_STRING_MISSING_VALUE = "\004";

/* Flag of this Ferret's start/stop state */
static int ferretInitialized = 0;

/* Memory for PPLUS */
static float  *pplMemory = NULL;

/* for recovering from problems in external function calls */
static void (*pyefcn_segv_handler)(int);
static jmp_buf pyefcn_jumpbuffer;
static void pyefcn_signal_handler(int signum)
{
    longjmp(pyefcn_jumpbuffer, signum);
}

/* 
 * Ctrl-C handler that just calls the CTRLC_AST Fortran subroutine 
 * defined in fer/gnl/ctrl_c.F (which has no arguments). 
 */
static void ferret_sigint_handler(int signum) 
{
    /* ignore any further Ctrl-C entries until done */
    signal(SIGINT, SIG_IGN);
    /* Now call the Fortran routine */
    FORTRAN(ctrlc_ast)();
    /* Go back to catching Ctrl-C */
    signal(SIGINT, ferret_sigint_handler);
}

static jmp_buf crash_jumpbuffer;
#ifdef NDEBUG
/*
 * Signal handler for program-quiting signals other than SIGINT.
 * For exiting gracefully to shut down any displayed viewers
 * and for generating a stderr message for LAS.
 * Only for production (not debug); for debug allow the crash to happen.
 */
static void crash_signal_handler(int signum) 
{
    longjmp(crash_jumpbuffer, signum);
}
#endif

/* 
 * Storage for original signal handlers when in the ferret engine.  
 * Largest ANSI/POSIX/BSD value of signals caught is SIGTERM = 15.
 */
#define MAX_SIGHANDLERS 32
static char *(signal_names[MAX_SIGHANDLERS]);
static void (*(orig_signal_handlers[MAX_SIGHANDLERS]))(int);

/*
 * Remove signal handlers assigned for the Ferret engine, 
 * restoring the original signal handlers.  If Ferret signal 
 * handlers are currently assigned, this call does nothing.
 */
static void remove_ferret_signal_handlers(void)
{
    int k;

    /* restore the original signal handlers */
    for (k = 0; k < MAX_SIGHANDLERS; k++) {
        if ( signal_names[k] != NULL ) {
            signal(k, orig_signal_handlers[k]);
        }
    }
    /* clear the signal names to indicate state of the signal handlers */
    for (k = 0; k < MAX_SIGHANDLERS; k++) {
        signal_names[k] = NULL;
    }
}

/*
 * Assign the signal handlers for the Ferret engine, saving the 
 * original signal handlers.  If the Ferret signal handlers are 
 * already assigned, this call does nothing.
 *
 * If a problem arose while reassigning a signal, all the original
 * signal handlers are restored, PyErr_StrString is called to set 
 * an appropriate error message and value, and -1 is returned.
 *
 * If successful, zero is returned. 
 */
static int assign_ferret_signal_handlers(void)
{
    /* Check if the handlers are already in place */
    if ( signal_names[SIGINT] != NULL )
        return 0;

    /* Let ferret deal with ctrl-C while in ferret mode */
    orig_signal_handlers[SIGINT] = signal(SIGINT, ferret_sigint_handler);
    if ( orig_signal_handlers[SIGINT] == SIG_ERR ) {
        remove_ferret_signal_handlers();
        PyErr_SetString(PyExc_SystemError, "Unable to catch SIGINT while in Ferret");
        return -1;
    }
    signal_names[SIGINT] = "SIGINT";

    /* Only catch other signals when compiled optimized - for debug, let them crash */
#ifdef NDEBUG
    /* Catch other program termination signals to gracefully return an error */
    orig_signal_handlers[SIGHUP] = signal(SIGHUP, crash_signal_handler);
    if ( orig_signal_handlers[SIGHUP] == SIG_ERR ) {
        remove_ferret_signal_handlers();
        PyErr_SetString(PyExc_SystemError, "Unable to catch SIGHUP while in Ferret");
        return -1;
    }
    signal_names[SIGHUP] = "SIGHUP";

    orig_signal_handlers[SIGQUIT] = signal(SIGQUIT, crash_signal_handler);
    if ( orig_signal_handlers[SIGQUIT] == SIG_ERR ) {
        remove_ferret_signal_handlers();
        PyErr_SetString(PyExc_SystemError, "Unable to catch SIGQUIT while in Ferret");
        return -1;
    }
    signal_names[SIGQUIT] = "SIGQUIT";

    orig_signal_handlers[SIGILL] = signal(SIGILL, crash_signal_handler);
    if ( orig_signal_handlers[SIGILL] == SIG_ERR ) {
        remove_ferret_signal_handlers();
        PyErr_SetString(PyExc_SystemError, "Unable to catch SIGILL while in Ferret");
        return -1;
    }
    signal_names[SIGILL] = "SIGILL";

#ifdef SIGBUS
    orig_signal_handlers[SIGBUS] = signal(SIGBUS, crash_signal_handler);
    if ( orig_signal_handlers[SIGBUS] == SIG_ERR ) {
        remove_ferret_signal_handlers();
        PyErr_SetString(PyExc_SystemError, "Unable to catch SIGBUS while in Ferret");
        return -1;
    }
    signal_names[SIGBUS] = "SIGBUS";
#endif

    orig_signal_handlers[SIGABRT] = signal(SIGABRT, crash_signal_handler);
    if ( orig_signal_handlers[SIGABRT] == SIG_ERR ) {
        remove_ferret_signal_handlers();
        PyErr_SetString(PyExc_SystemError, "Unable to catch SIGABRT while in Ferret");
        return -1;
    }
    signal_names[SIGABRT] = "SIGABRT";

    orig_signal_handlers[SIGFPE] = signal(SIGFPE, crash_signal_handler);
    if ( orig_signal_handlers[SIGFPE] == SIG_ERR ) {
        remove_ferret_signal_handlers();
        PyErr_SetString(PyExc_SystemError, "Unable to catch SIGFPE while in Ferret");
        return -1;
    }
    signal_names[SIGFPE] = "SIGFPE";

    orig_signal_handlers[SIGSEGV] = signal(SIGSEGV, crash_signal_handler);
    if ( orig_signal_handlers[SIGSEGV] == SIG_ERR ) {
        remove_ferret_signal_handlers();
        PyErr_SetString(PyExc_SystemError, "Unable to catch SIGSEGV while in Ferret");
        return -1;
    }
    signal_names[SIGSEGV] = "SIGSEGV";

    orig_signal_handlers[SIGTERM] = signal(SIGTERM, crash_signal_handler);
    if ( orig_signal_handlers[SIGTERM] == SIG_ERR ) {
        remove_ferret_signal_handlers();
        PyErr_SetString(PyExc_SystemError, "Unable to catch SIGTERM while in Ferret");
        return -1;
    }
    signal_names[SIGTERM] = "SIGTERM";
#endif

    return 0;
}


static char pyferretStartDocstring[] =
    "Initializes Ferret.  This allocates the initial amount of memory for \n"
    "Ferret (from Python-managed memory), opens the journal file, if requested, \n"
    "and sets Ferret's verify mode.  If restrict is True, some Ferret commands \n"
    "will not be available (to provide a secured session).  Once restrict is set, \n"
    "it cannot be unset.  If server is True, Ferret will be run in server mode. \n"
    "If metaname is empty (and not in server mode), Ferret's graphics will be \n"
    "displayed by default;  otherwise, this value is used as the initial filename \n"
    "for output graphics.  This routine does NOT run any user initialization scripts. \n"
    "\n"
    "Required arguments: \n"
    " (none) \n"
    "\n"
    "Optional arguments: \n"
    "    memsize = <float>: the size, in megadoubles (where a double is 8 bytes), \n"
    "                       to allocate for Ferret's memory cache (default 125 == 1Gb) \n"
    "    journal = <bool>: journal Ferret commands? (default True) \n"
    "    verify = <bool>: echo Ferret commands? (default True) \n"
    "    restrict = <bool>: restrict Ferret's capabilities? (default False) \n"
    "    server = <bool>: run Ferret in server mode? (default False) \n"
    "    metaname = <string>: filename for Ferret graphics (default empty) \n"
    "    unmapped = <bool>: hide the graphics viewer? (default False unless png is True) \n"
    "    pngonly = <bool>: write directly to a PNG? (default False; True implies unmapped) \n"
    "    quiet = <bool>: do not print the ferret header? (default False) \n"
    "    linebuffer = <bool>: use line buffering for stdout and stderr? (default False) \n"
    "            Note: \n"
    "                the enviroment variable GFORTRAN_UNBUFFERED_PRECONNECTED \n"
    "                needs to be set to 1 in order to unbuffer the Fortran \n"
    "                units for output and error messages\n"
    "\n"
    "Returns: \n"
    "    True is successful \n"
    "    False if Ferret has already been started \n"
    "\n"
    "Raises: \n"
    "    MemoryError if unable to allocate the needed memory \n"
    "    IOError if unable to open the journal file \n";

static PyObject *pyferretStart(PyObject *self, PyObject *args, PyObject *kwds)
{
    static char *argNames[] = {"memsize", "journal", "verify", "restrict", "server",
                               "metaname", "transparent", "unmapped", "pngonly", 
                               "quiet", "linebuffer", NULL};
    double mwMemSize = 125.0;
    PyObject *pyoJournal = NULL;
    PyObject *pyoVerify = NULL;
    PyObject *pyoRestrict = NULL;
    PyObject *pyoServer = NULL;
    PyObject *pyoTransparent = NULL;
    PyObject *pyoUnmapped = NULL;
    PyObject *pyoPngonly = NULL;
    PyObject *pyoQuiet = NULL;
    PyObject *pyoLineBuffer = NULL;
    char *metaname = NULL;
    int journalFlag = 1;
    int verifyFlag = 1;
    int restrictFlag = 0;
    int serverFlag = 0;
    int transparentFlag = 0;
    int unmappedFlag = 0;
    int pngonlyFlag = 0;
    int quietFlag = 0;
    int lineBufferFlag = 0;
    int pplMemSize;
    int status;
    int ttoutLun = TTOUT_LUN;
    int one_cmnd_mode_int;
    PyObject *modulename;

    /* If already initialized, return False */
    if ( ferretInitialized ) {
        Py_INCREF(Py_False);
	return Py_False;
    }

    /* Import the function-pointer table for the PyArray_* functions */
    import_array1(NULL);

    /* Parse the arguments, checking if an Exception was raised */
    if ( ! PyArg_ParseTupleAndKeywords(args, kwds, "|dO!O!O!O!sO!O!O!O!O!",
                 argNames, &mwMemSize, &PyBool_Type, &pyoJournal,
                 &PyBool_Type, &pyoVerify, &PyBool_Type, &pyoRestrict,
                 &PyBool_Type, &pyoServer, &metaname,
                 &PyBool_Type, &pyoTransparent, &PyBool_Type, &pyoUnmapped,
                 &PyBool_Type, &pyoPngonly, &PyBool_Type, &pyoQuiet, 
                 &PyBool_Type, &pyoLineBuffer) )
        return NULL;

    /* Interpret the booleans - Py_False and Py_True are singleton non-NULL objects, so just use == */
    if ( pyoJournal == Py_False )
        journalFlag = 0;
    if ( pyoVerify == Py_False )
        verifyFlag = 0;
    if ( pyoRestrict == Py_True )
        restrictFlag = 1;
    if ( pyoServer == Py_True )
        serverFlag = 1;
    if ( pyoTransparent == Py_True )
        transparentFlag = 1;
    if ( pyoUnmapped == Py_True )
        unmappedFlag = 1;
    if ( pyoPngonly == Py_True )
        pngonlyFlag = 1;
    if ( pyoQuiet == Py_True )
        quietFlag = 1;
    if ( pyoLineBuffer == Py_True )
        lineBufferFlag = 1;

    if ( pngonlyFlag != 0 )
        unmappedFlag = 1;
    if ( metaname[0] == '\0' )
        metaname = NULL;

    /* Deal with line buffering */
    if ( lineBufferFlag != 0 ) {
        /* Set line buffering on stdout and stderr; ignore failures */
        setvbuf(stdout, NULL, _IOLBF, BUFSIZ);
        setvbuf(stderr, NULL, _IOLBF, BUFSIZ);
    }
    /* Deal with the restrict and server flags right away */
    if ( restrictFlag != 0 )
        set_secure();
    if ( serverFlag != 0 )
        set_server();

    /* Initialize the shared buffer sBuffer */
    set_shared_buffer();

    /* Initial allocation of PPLUS memory */
    pplMemSize = 1024 * 1024;
    pplMemory = (float *) FerMem_Malloc((size_t)pplMemSize * (size_t)sizeof(float), __FILE__, __LINE__);
    if ( pplMemory == NULL )
        return PyErr_NoMemory();
    set_ppl_memory(pplMemory, pplMemSize);
    FORTRAN(init_memory)(&mwMemSize);

    if ( (metaname != NULL) || (unmappedFlag != 0) ) {
       /*
        * Set the default graphics filename for saving before ending.
        * Make a copy of the name just in case set_batch_graphics changes
        * something.  This also hides the graphics viewer.
        */
       char my_meta_name[2048];
       if ( metaname != NULL ) {
           strncpy(my_meta_name, metaname, 2048);
           my_meta_name[2047] = '\0';
       }
       else {
           my_meta_name[0] = '\0';
       }
       set_batch_graphics(my_meta_name, &pngonlyFlag);
    }

    /* Set the default autosave transparency */
    FORTRAN(fgd_set_transparency)(&transparentFlag);

    /* Initialize stuff: keyboard, todays date, grids, GFDL terms, PPL brain */
    FORTRAN(initialize_ferret)();

    /* Open the output journal file, if appropriate */
    if ( journalFlag != 0 ) {
        FORTRAN(init_journal)(&status);
        if ( status != FERR_OK ) {
            PyErr_SetString(PyExc_IOError, "Unable to open the journal file ferret.jnl");
            return NULL;
        }
    }
    else
        FORTRAN(no_journal)();

    /* Set the verify flag */
    if ( verifyFlag == 0 )
        FORTRAN(turnoff_verify)(&status);

    /* Get the PyObject representing the pyferret module */
#if PY_MAJOR_VERSION > 2
    modulename = PyUnicode_FromString("pyferret");
#else
    modulename = PyString_FromString("pyferret");
#endif
    if ( modulename == NULL ) {
        return NULL;
    }
    pyferret_module_pyobject = PyImport_Import(modulename);
    Py_DECREF(modulename);
    if ( pyferret_module_pyobject == NULL ) {
        return NULL;
    }

    /* Get the PyObject representing the pyferret.graphbind module */
#if PY_MAJOR_VERSION > 2
    modulename = PyUnicode_FromString("pyferret.graphbind");
#else
    modulename = PyString_FromString("pyferret.graphbind");
#endif
    if ( modulename == NULL ) {
        Py_DECREF(pyferret_module_pyobject);
        return NULL;
    }
    pyferret_graphbind_module_pyobject = PyImport_Import(modulename);
    Py_DECREF(modulename);
    if ( pyferret_graphbind_module_pyobject == NULL ) {
        return NULL;
    }

    /* Set and possibly output program name and revision number */
    FORTRAN(proclaim_c)(&ttoutLun, "\t", &quietFlag);

    /* Set so that ferret_dispatch returns after every command */
    one_cmnd_mode_int = 1;
    FORTRAN(set_one_cmnd_mode)(&one_cmnd_mode_int);

    /* Success - return True */
    ferretInitialized = 1;
    Py_INCREF(Py_True);
    return Py_True;
}


/*
 * Called by the Ferret core to reallocate for more PPL memory
 * Preface the message with **ERROR so it gets shown as an error within PyFerret by LAS
 */
void reallo_ppl_memory(int new_size)
{
    if ( pplMemory != NULL )
        FerMem_Free(pplMemory, __FILE__, __LINE__);
    pplMemory = (float *) FerMem_Malloc((size_t)new_size * sizeof(float), __FILE__, __LINE__);
    if ( pplMemory == NULL ) {
        printf("**ERROR: PyFerret. Unable to allocate the requested %d words of PLOT memory.\n", new_size);
        exit(1);
    }
    set_ppl_memory(pplMemory, new_size);
}


static char pyferretRunCommandDocstring[] =
    "Runs a Ferret command just as if entering a command at the Ferret prompt. \n"
    "\n"
    "If an empty string is given, Ferret will prompt you for commands \n"
    "until either the \"EXIT\" or the \"EXIT /TOPYTHON\" command is given. \n"
    "If \"EXIT /TOPYTHON\" is given, the return tuple will be for the last \n"
    "error, if any, that occurred in the sequence of commands submitted. \n"
    "\n"
    "Required arguments: \n"
    "    command = <string>: the Ferret command to be run \n"
    "\n"
    "Optional arguments: \n"
    "    (none) \n"
    "\n"
    "Returns: \n"
    "   (err_int, err_string) \n"
    "       err_int: one of the FERR_* data values (FERR_OK if there are no errors) \n"
    "       err_string: error or warning message (can be empty) \n"
    "   Error messages normally start with \"**ERROR\" \n"
    "   Warning messages normally start with \"*** NOTE:\" \n"
    "\n"
    "Raises: \n"
    "    MemoryError if Ferret has not been started or has been stopped \n";

static PyObject *pyferretRunCommand(PyObject *self, PyObject *args, PyObject *kwds)
{
    static char *argNames[] = {"command", NULL};
    char *command;
    char *iter_command;
    int  one_cmnd_mode_int;
    int  cmnd_stack_level;
    char errmsg[2112];
    int  errval;

    /* If not initialized, raise a MemoryError */
    if ( ! ferretInitialized ) {
        PyErr_SetString(PyExc_MemoryError, "Ferret not started");
        return NULL;
    }

    /* Parse the arguments, checking if an Exception was raised */
    if ( ! PyArg_ParseTupleAndKeywords(args, kwds, "s", argNames, &command) )
        return NULL;

    /* Clear the last error message and value */
    FORTRAN(clear_fer_last_error_info)();

    /* If an empty string, temporarily turn off the one-command mode */
    if ( command[0] == '\0' ) {
        one_cmnd_mode_int = 0;
        FORTRAN(set_one_cmnd_mode)(&one_cmnd_mode_int);
    }
    else
        one_cmnd_mode_int = 1;

    errval = setjmp(crash_jumpbuffer);
    if ( errval != 0 ) {
        /* 
         * If we get here, a signal was caught with crash_signal_handler 
         * as its handler.  The value errval is the signal number. 
         */

        /* 
         * The following will exit completely but leaves up any displays:
         *
         *    fprintf(stderr, "**ERROR Ferret crash; signal = %d (%s)\n", 
         *                    errval, signal_names[errval]);
         *    remove_ferret_signal_handlers();
         *    exit(-1);
         *
         *
         * Instead, raise a RuntimeError with an appropriate error message
         * so a proper shutdown can be performed.
         */
        sprintf(errmsg, "\n\n"
                        "**ERROR Ferret crash; signal = %d (%s)\n"
                        "Enter Ctrl-D to exit Python\n", 
                        errval, signal_names[errval]);
        remove_ferret_signal_handlers();
        /* 
         * Clear any problem status that might have arisen in Python.
         * The RuntimeError will be raised instead.
         */
        PyErr_Clear();
        PyErr_SetString(PyExc_RuntimeError, errmsg);
        return NULL;
    }
    /* Assign appropriate signal handlers */
    if ( assign_ferret_signal_handlers() != 0 ) {
        /* 
         * Problems assigning signal handlers; signals all
         * restored to original and PyErr_SetString called 
         */
        return NULL;
    }

    /* do-loop only for dealing with Ferret "SET MEMORY /SIZE=..." resize command */
    iter_command = command;
    do {
        cmnd_stack_level = 0;
        /* Run the Ferret command */
	ferret_dispatch_c(iter_command, sBuffer);

        if ( sBuffer->flags[FRTN_ACTION] == FACTN_MEM_RECONFIGURE ) {
            /* Now handled internally */
            cmnd_stack_level = sBuffer->flags[FRTN_IDATA2];
        }
        else {
            /*
             * Not a memory resize command; probably an exit command.
             * Do not allow return to the Python prompt if in restricted mode
             * (is_secure returns non-zero).
             */
            if ( FORTRAN(is_secure)() == 0 )
               break;
            if ( sBuffer->flags[FRTN_ACTION] == FACTN_EXIT ) {
               remove_ferret_signal_handlers();
               exit(0);
            }
        }
        /* submit an empty command to continue on with whaterever was going on */
        iter_command = "";
    } while ( (one_cmnd_mode_int == 0) || (cmnd_stack_level > 0) );

    /* Restore all the signal handlers that were changed */
    remove_ferret_signal_handlers();

    /* Set back to single command mode */
    if ( one_cmnd_mode_int == 0 ) {
        one_cmnd_mode_int = 1;
        FORTRAN(set_one_cmnd_mode)(&one_cmnd_mode_int);
    }

    if ( sBuffer->flags[FRTN_ACTION] == FACTN_EXIT ) {
        /* 
         * plain "EXIT" Ferret command - exit completely
         *
         * python -i -c ... intercepts the Python sys.exit() call and stays in python,
         * so just do a C exit() from python - 
         * after doing some clean-up for memory leak detection.
         */
        if ( ferretInitialized ) {
            /* Set to uninitialized */
            ferretInitialized = 0;

            /* Release the references to the pyferret and pyferret.graphbind modules */
            Py_DECREF(pyferret_graphbind_module_pyobject);
            pyferret_graphbind_module_pyobject = NULL;
            Py_DECREF(pyferret_module_pyobject);
            pyferret_module_pyobject = NULL;

            /* Free memory allocated inside Ferret */
            FORTRAN(finalize_ferret)();

            /* Free memory allocated for PPLUS */
            FerMem_Free(pplMemory, __FILE__, __LINE__);
            pplMemory = NULL;
        }

#ifdef MEMORYDEBUG
        (void) ReportAnyMemoryLeaks();
#endif

        exit(0);
    }

    /* Get the last error message (null terminated) and value */
    FORTRAN(get_fer_last_error_info)(&errval, errmsg, 2112);

    /* Return the tuple of the last error value and message */
    return Py_BuildValue("is", errval, errmsg);
}


static char pyferretGetDataDocstring[] =
    "Returns the numeric data array described in the argument. \n"
    "\n"
    "Required arguments: \n"
    "    name = <string>: the name of the numeric data array to return \n"
    "\n"
    "Optional arguments: \n"
    "    (none) \n"
    "\n"
    "Returns: \n"
    "    A tuple containing: \n"
    "        a NumPy float64 ndarray containing a copy of the numeric data requested, \n"
    "        a NumPy float64 ndarray containing the bad-data-flag value for the data, \n"
    "        a string giving the units for the data \n"
    "        a tuple of six integers giving the AXISTYPE codes of the axes, \n"
    "        a tuple of six strings giving the names of the axes, \n"
    "        a tuple of six strings giving the units of a non-calendar-time data axis, or \n"
    "                                       the CALTYPE_ calendar name of a calendar-time axis, \n"
    "        a tuple of six ndarrays giving the coordinates for the data axes \n"
    "            (ndarray of N doubles for non-calendar-time, non-normal axes, \n"
    "             ndarray of (N,6) integers for calendar-time axes, or \n"
    "             None for normal axes) \n"
    "\n"
    "Raises: \n"
    "    ValueError if the data name is invalid \n"
    "    MemoryError if Ferret has not been started or has been stopped \n";

static PyObject *pyferretGetData(PyObject *self, PyObject *args, PyObject *kwds)
{
    static char   *argNames[] = {"name", NULL};
    char          *name;
    int            lendataname;
    char           dataname[1024];
    double        *arraystart;
    int            memlo[MAX_FERRET_NDIM], memhi[MAX_FERRET_NDIM];
    int            steplo[MAX_FERRET_NDIM], stephi[MAX_FERRET_NDIM], incr[MAX_FERRET_NDIM];
    char           dataunit[64];
    int            lendataunit;
    AXISTYPE       axis_types[MAX_FERRET_NDIM];
    char           errmsg[2112];
    int            lenerrmsg;
    double         badval;
    int            i, j, k, l, m, n, q;
    npy_intp       shape[MAX_FERRET_NDIM];
    npy_intp       new_shape[2];
    int            strides[MAX_FERRET_NDIM];
    PyArrayObject *data_ndarray;
    double        *npydata;
    PyArrayObject *badval_ndarray;
    PyArrayObject *axis_coords[MAX_FERRET_NDIM];
    char           axis_units[MAX_FERRET_NDIM][64];
    char           axis_names[MAX_FERRET_NDIM][64];
    CALTYPE        calendar_type;

    /* If not initialized, raise a MemoryError */
    if ( ! ferretInitialized ) {
        PyErr_SetString(PyExc_MemoryError, "Ferret not started");
        return NULL;
    }

    /* Parse the arguments, checking if an Exception was raised */
    if ( ! PyArg_ParseTupleAndKeywords(args, kwds, "s", argNames, &name) )
        return NULL;

    /* Make a copy of dataname just to be sure it isn't altered */
    lendataname = strlen(name);
    if ( lendataname > 1020 ) {
        PyErr_SetString(PyExc_ValueError, "name too long");
        return NULL;
    }
    strcpy(dataname, name);

    /*
     * Retrieve the memory parameters describing the data array requested.
     * Assumes Unix standard for passing strings to Fortran (appended array lengths).
     */
    FORTRAN(get_data_array_params)(dataname, &lendataname, &arraystart, memlo, memhi,
                           steplo, stephi, incr, dataunit, &lendataunit, axis_types,
                           &badval, errmsg, &lenerrmsg, 1024, 64, 2112);
    if ( lenerrmsg > 0 ) {
        errmsg[lenerrmsg] = '\0';
        PyErr_SetString(PyExc_ValueError, errmsg);
        return NULL;
    }

    /* null terminate the data unit name */
    dataunit[lendataunit] = '\0';

    /* Get the shape of the array */
    for (k = 0; k < MAX_FERRET_NDIM; k++)
        shape[k] = (npy_intp) ((stephi[k] - steplo[k] + incr[k]) / (incr[k]));

    /* Get the strides through the memory (as a double *) */
    strides[0] = 1;
    for (k = 1; k < MAX_FERRET_NDIM; k++)
        strides[k] = strides[k-1] * (memhi[k-1] - memlo[k-1] + 1);

    /* Get the actual starting point in the array */
    for (k = 0; k < MAX_FERRET_NDIM; k++)
        arraystart += (strides[k]) * (steplo[k] - memlo[k]);

    /* Convert to strides through places in memory to be read */
    for (k = 0; k < MAX_FERRET_NDIM; k++)
        strides[k] *= incr[k];

    /* Create a new NumPy double ndarray (Fortran ordering) with the same shape */
    data_ndarray = (PyArrayObject *) PyArray_EMPTY(MAX_FERRET_NDIM, shape, NPY_DOUBLE, 1);
    if ( data_ndarray == NULL ) {
        return NULL;
    }

    /*
     * Assign the data in the new ndarray.
     * Note: if MAX_FERRET_NDIM changes, this needs editing.
     */
    npydata = (double *)PyArray_DATA(data_ndarray);
    q = 0;
    for (n = 0; n < (int)(shape[5]); n++) {
      for (m = 0; m < (int)(shape[4]); m++) {
        for (l = 0; l < (int)(shape[3]); l++) {
          for (k = 0; k < (int)(shape[2]); k++) {
            for (j = 0; j < (int)(shape[1]); j++) {
              for (i = 0; i < (int)(shape[0]); i++) {
                npydata[q] = arraystart[ i * strides[0] + 
                                         j * strides[1] + 
                                         k * strides[2] + 
                                         l * strides[3] +
                                         m * strides[4] +
                                         n * strides[5] ];
                q++;
              }
            }
          }
        }
      }
    }

    /* Create a new NumPy float ndarray with the bad-data-flag value(s) */
    new_shape[0] = 1;
    badval_ndarray = (PyArrayObject *) PyArray_SimpleNew(1, new_shape, NPY_DOUBLE);
    if ( badval_ndarray == NULL ) {
       Py_DECREF(data_ndarray);
       return NULL;
    }
    npydata = (double *)PyArray_DATA(badval_ndarray);
    npydata[0] = badval;

    /* Create the axis coordinates array objects */
    for (k = 0; k < MAX_FERRET_NDIM; k++) {
        switch( axis_types[k] ) {
        case AXISTYPE_LONGITUDE:
        case AXISTYPE_LATITUDE:
        case AXISTYPE_LEVEL:
        case AXISTYPE_CUSTOM:
        case AXISTYPE_ABSTRACT:
            /* array of doubles, possibly with a units string */
            axis_coords[k] = (PyArrayObject *) PyArray_SimpleNew(1, &(shape[k]), NPY_DOUBLE);
            if ( axis_coords[k] == NULL ) {
                while ( k > 0 ) {
                    k--;
                    Py_DECREF(axis_coords[k]);
                }
                Py_DECREF(badval_ndarray);
                Py_DECREF(data_ndarray);
                return NULL;
            }
            /* get the coordinates and the units string */
            q = k+1;
            j = (int) shape[k];
            FORTRAN(get_data_array_coords)((double *)PyArray_DATA(axis_coords[k]), axis_units[k],
                                   axis_names[k], &q, &j, errmsg, &lenerrmsg, 64, 64, 2112);
            if ( lenerrmsg > 0 ) {
                errmsg[lenerrmsg] = '\0';
                PyErr_SetString(PyExc_ValueError, errmsg);
                Py_DECREF(axis_coords[k]);
                while ( k > 0 ) {
                    k--;
                    Py_DECREF(axis_coords[k]);
                }
                Py_DECREF(badval_ndarray);
                Py_DECREF(data_ndarray);
                return NULL;
            }
            break;
        case AXISTYPE_TIME:
            /* array of 6-tuples of integers in C order, so: [N][6] in C or (6,N) in Fortran */
            new_shape[0] = shape[k];
            new_shape[1] = 6;
            axis_coords[k] = (PyArrayObject *) PyArray_SimpleNew(2, new_shape, NPY_INT);
            if ( axis_coords[k] == NULL ) {
                while ( k > 0 ) {
                    k--;
                    Py_DECREF(axis_coords[k]);
                }
                Py_DECREF(badval_ndarray);
                Py_DECREF(data_ndarray);
                return NULL;
            }
            /* get the time coordinate integers */
            q = k+1;
            j = (int) shape[k];
            FORTRAN(get_data_array_time_coords)((int (*)[6])PyArray_DATA(axis_coords[k]), &calendar_type, axis_names[k],
                                        &q, &j, errmsg, &lenerrmsg, 64, 2112);
            if ( lenerrmsg > 0 ) {
                errmsg[lenerrmsg] = '\0';
                PyErr_SetString(PyExc_ValueError, errmsg);
                Py_DECREF(axis_coords[k]);
                while ( k > 0 ) {
                    k--;
                    Py_DECREF(axis_coords[k]);
                }
                Py_DECREF(badval_ndarray);
                Py_DECREF(data_ndarray);
                return NULL;
            }
            /* set the axis units to the name of the calendar */
            switch( calendar_type ) {
            case CALTYPE_NONE:
                strcpy(axis_units[k], CALTYPE_NONE_STR);
                break;
            case CALTYPE_360DAY:
                strcpy(axis_units[k], CALTYPE_360DAY_STR);
                break;
            case CALTYPE_NOLEAP:
                strcpy(axis_units[k], CALTYPE_NOLEAP_STR);
                break;
            case CALTYPE_GREGORIAN:
                strcpy(axis_units[k], CALTYPE_GREGORIAN_STR);
                break;
            case CALTYPE_JULIAN:
                strcpy(axis_units[k], CALTYPE_JULIAN_STR);
                break;
            case CALTYPE_ALLLEAP:
                strcpy(axis_units[k], CALTYPE_ALLLEAP_STR);
                break;
            default:
                sprintf(errmsg, "Unexpected calendar type of %d", calendar_type);
                PyErr_SetString(PyExc_ValueError, errmsg);
                Py_DECREF(axis_coords[k]);
                while ( k > 0 ) {
                    k--;
                    Py_DECREF(axis_coords[k]);
                }
                Py_DECREF(badval_ndarray);
                Py_DECREF(data_ndarray);
                return NULL;
            }
            break;
        case AXISTYPE_NORMAL:
            /* axis normal to the results - no coordinates */
            Py_INCREF(Py_None);
            axis_coords[k] = (PyArrayObject *) Py_None;
            axis_units[k][0] = '\0';
            axis_names[k][0] = '\0';
            break;
        default:
            sprintf(errmsg, "Unexpected axis type of %d", axis_types[k]);
            PyErr_SetString(PyExc_RuntimeError, errmsg);
            while ( k > 0 ) {
                k--;
                Py_DECREF(axis_coords[k]);
            }
            Py_DECREF(badval_ndarray);
            Py_DECREF(data_ndarray);
            return NULL;
        }
    }

    /*
     * Return a tuple (stealing references for PyObjects) of data_ndarray,
     * badval_ndarray, dataunit, axis_types, axis_names, axis_units, and axis_coords.
     * Note: if MAX_FERRET_NDIM changes, this needs editing.
     */
    return Py_BuildValue("NNs(iiiiii)(ssssss)(ssssss)(NNNNNN)", data_ndarray, badval_ndarray, dataunit,
              axis_types[0], axis_types[1], axis_types[2], axis_types[3], axis_types[4], axis_types[5],
              axis_names[0], axis_names[1], axis_names[2], axis_names[3], axis_names[4], axis_names[5],
              axis_units[0], axis_units[1], axis_units[2], axis_units[3], axis_units[4], axis_units[5],
              axis_coords[0], axis_coords[1], axis_coords[2], axis_coords[3], axis_coords[4], axis_coords[5]);
}


static char pyferretPutDataDocstring[] =
    "Creates a Ferret data variable with the numeric data array described in the arguments. \n"
    "\n"
    "Required arguments: \n"
    "    codename = <string>: the code name of the Ferret data variable to create (eg, \"SST\") \n"
    "    title = <string>: the title of the Ferret data variable to create (eg, \"Sea Surface Temperature\") \n"
    "    data = <ndarray>: the array containing the numeric data \n"
    "    bdfval = <ndarray>: the bad-data-flag value for the data \n"
    "    units = <string>: the units for the data \n"
    "    dset = <string>: the dataset name or number to be associates with this variable; \n"
    "                     give an empty strip associated with the current dataset \n"
    "                     or 'None' to not associate with any dataset \n"
    "    axis_types = <6-tuple of int>: the AXISTYPE codes for the axes \n"
    "    axis_names = <6-tuple of string>: the names of the axes \n"
    "    axis_units = <6-tuple of string>: the units of a non-calendar-time axis, or \n"
    "                                      the CALTYPE_ calendar name of a calendar-time axis \n"
    "    axis_coords = <6-tuple of ndarray>: the axis coordinates \n"
    "                                        (ndarray of N doubles for a non-calendar-time, non-normal axis, or \n"
    "                                         ndarray of (N,6) integers for a calendar-time axis; \n"
    "                                         None - or any object - for a normal axis) \n"
    "\n"
    "Optional arguments: \n"
    "    (none) \n"
    "\n"
    "Returns: \n"
    "    None \n"
    "\n"
    "Raises: \n"
    "    ValueError if there is a problem with the argument data passed \n"
    "    MemoryError if Ferret has not been started or has been stopped \n";

static PyObject *pyferretPutData(PyObject *self, PyObject *args, PyObject *kwds)
{
    static char *argNames[] = {"codename", "title", "data", "bdfval", "units", "dset",
                               "axis_types", "axis_names", "axis_units", "axis_coords", NULL};
    char          *codename;
    char          *title;
    PyArrayObject *data_ndarray;
    PyArrayObject *bdfval_ndarray;
    char          *units;
    char          *dset;
    PyObject      *axis_types_tuple;
    PyObject      *axis_names_tuple;
    PyObject      *axis_units_tuple;
    PyObject      *axis_coords_tuple;
    double         bdfval;
    int            k;
    PyObject      *seqitem;
    AXISTYPE       axis_types[MAX_FERRET_NDIM];
    char          *strptr;
    char           axis_names[MAX_FERRET_NDIM][64];
    char           axis_units[MAX_FERRET_NDIM][64];
    int            num_coords[MAX_FERRET_NDIM];
    void          *axis_coords[MAX_FERRET_NDIM];
    CALTYPE        calendar_type;
    int            axis_nums[MAX_FERRET_NDIM];
    int            axis_starts[MAX_FERRET_NDIM];
    int            axis_ends[MAX_FERRET_NDIM];
    int            len_codename;
    int            len_title;
    int            len_units;
    int            len_dset;
    char           errmsg[2048];
    int            len_errmsg;

    /* If not initialized, raise a MemoryError */
    if ( ! ferretInitialized ) {
        PyErr_SetString(PyExc_MemoryError, "Ferret not started");
        return NULL;
    }

    /* Parse the arguments, checking if an Exception was raised - borrowed references to the PyObjects */
    if ( ! PyArg_ParseTupleAndKeywords(args, kwds, "ssOOssOOOO", argNames, &codename, &title,
                                       &data_ndarray, &bdfval_ndarray, &units, &dset, &axis_types_tuple,
                                       &axis_names_tuple, &axis_units_tuple, &axis_coords_tuple) )
        return NULL;

    /* PyArray_Size returns 0 if the object is not an appropriate type */
    /* ISFARRAY_RO checks if it is F-contiguous, aligned, and in machine byte-order */
    if ( (PyArray_Size((PyObject *) data_ndarray) < 1) || (PyArray_TYPE(data_ndarray) != NPY_DOUBLE) ||
         (! PyArray_ISFARRAY_RO(data_ndarray)) || (! PyArray_CHKFLAGS(data_ndarray, NPY_ARRAY_OWNDATA)) ) {
        PyErr_SetString(PyExc_ValueError, "data is not an appropriate ndarray of type float64");
        return NULL;
    }

    /* PyArray_Size returns 0 if the object is not an appropriate type */
    /* ISBEHAVED_RO checks if it is aligned and in machine byte-order */
    if ( (PyArray_Size((PyObject *) bdfval_ndarray) < 1) || (PyArray_TYPE(bdfval_ndarray) != NPY_DOUBLE) ||
         (! PyArray_ISBEHAVED_RO(bdfval_ndarray)) ) {
        PyErr_SetString(PyExc_ValueError, "bdfval is not an appropriate ndarray of type float64");
        return NULL;
    }
    /* Just get bdfval from the data in bdfval_ndarray */
    bdfval = ((double *)PyArray_DATA(bdfval_ndarray))[0];

    /* Get the axis types out of the tuple */
    axis_types_tuple = PySequence_Fast(axis_types_tuple, "axis_types is not a tuple or list");
    if ( axis_types_tuple == NULL ) {
        return NULL;
    }
    if ( PySequence_Fast_GET_SIZE(axis_types_tuple) != MAX_FERRET_NDIM ) {
        PyErr_SetString(PyExc_ValueError, "axis_types does not have the expected number of items");
        Py_DECREF(axis_types_tuple);
        return NULL;
    }
    for (k = 0; k < MAX_FERRET_NDIM; k++) {
        seqitem = PySequence_Fast_GET_ITEM(axis_types_tuple, k); /* borrowed reference */
#if PY_MAJOR_VERSION > 2
        axis_types[k] = (int) PyLong_AsLong(seqitem);
#else
        axis_types[k] = (int) PyInt_AsLong(seqitem);
#endif
        if ( (axis_types[k] != AXISTYPE_LONGITUDE) &&
             (axis_types[k] != AXISTYPE_LATITUDE) &&
             (axis_types[k] != AXISTYPE_LEVEL) &&
             (axis_types[k] != AXISTYPE_TIME) &&
             (axis_types[k] != AXISTYPE_CUSTOM) &&
             (axis_types[k] != AXISTYPE_ABSTRACT) &&
             (axis_types[k] != AXISTYPE_NORMAL) ) {
            PyErr_SetString(PyExc_ValueError, "Invalid axis_types item");
            Py_DECREF(axis_types_tuple);
            return NULL;
        }
    }
    Py_DECREF(axis_types_tuple);

    /* Get the axis names out of the tuple */
    axis_names_tuple = PySequence_Fast(axis_names_tuple, "axis_names is not a tuple or list");
    if ( axis_names_tuple == NULL ) {
        return NULL;
    }
    if ( PySequence_Fast_GET_SIZE(axis_names_tuple) != MAX_FERRET_NDIM ) {
        PyErr_SetString(PyExc_ValueError, "axis_names does not have the expected number of items");
        Py_DECREF(axis_names_tuple);
        return NULL;
    }
    for (k = 0; k < MAX_FERRET_NDIM; k++) {
        seqitem = PySequence_Fast_GET_ITEM(axis_names_tuple, k); /* borrowed reference */
#if PY_MAJOR_VERSION > 2
        strptr = PyUnicode_AsUTF8(seqitem);
#else
        strptr = PyString_AsString(seqitem);
#endif
        if ( strptr == NULL ) {
            PyErr_Clear();
            PyErr_SetString(PyExc_ValueError, "Invalid axis_names item");
            Py_DECREF(axis_names_tuple);
            return NULL;
        }
        strncpy(axis_names[k], strptr, 64);
        axis_names[k][63] = '\0';
    }
    Py_DECREF(axis_names_tuple);

    /* Get the axis units out of the tuple */
    axis_units_tuple = PySequence_Fast(axis_units_tuple, "axis_units is not a tuple or list");
    if ( axis_units_tuple == NULL ) {
        return NULL;
    }
    if ( PySequence_Fast_GET_SIZE(axis_units_tuple) != MAX_FERRET_NDIM ) {
        PyErr_SetString(PyExc_ValueError, "axis_units does not have the expected number of items");
        Py_DECREF(axis_units_tuple);
        return NULL;
    }
    for (k = 0; k < MAX_FERRET_NDIM; k++) {
        seqitem = PySequence_Fast_GET_ITEM(axis_units_tuple, k); /* borrowed reference */
#if PY_MAJOR_VERSION > 2
        strptr = PyUnicode_AsUTF8(seqitem);
#else
        strptr = PyString_AsString(seqitem);
#endif
        if ( strptr == NULL ) {
            PyErr_Clear();
            PyErr_SetString(PyExc_ValueError, "Invalid axis_units item");
            Py_DECREF(axis_units_tuple);
            return NULL;
        }
        strncpy(axis_units[k], strptr, 64);
        axis_units[k][63] = '\0';
    }
    Py_DECREF(axis_units_tuple);

    /* Get the axis coordinates ndarray out of the tuple */
    axis_coords_tuple = PySequence_Fast(axis_coords_tuple, "axis_coords is not a tuple or list");
    if ( axis_coords_tuple == NULL ) {
        return NULL;
    }
    if ( PySequence_Fast_GET_SIZE(axis_coords_tuple) != MAX_FERRET_NDIM ) {
        PyErr_SetString(PyExc_ValueError, "axis_coords does not have the expected number of items");
        Py_DECREF(axis_coords_tuple);
        return NULL;
    }
    for (k = 0; k < MAX_FERRET_NDIM; k++) {
        seqitem = PySequence_Fast_GET_ITEM(axis_coords_tuple, k); /* borrowed reference */
        switch( axis_types[k] ) {
        case AXISTYPE_LONGITUDE:
        case AXISTYPE_LATITUDE:
        case AXISTYPE_LEVEL:
        case AXISTYPE_CUSTOM:
        case AXISTYPE_ABSTRACT:
            /* float64 N-ndarray containing the axis coordinates */
            /* PyArray_Size returns 0 if the object is not an appropriate type */
            /* ISCARRAY_RO checks if it is C-contiguous, aligned and in machine byte-order */
            num_coords[k] = PyArray_Size(seqitem);
            if ( num_coords[k] < 1 ) {
                PyErr_SetString(PyExc_ValueError, "a standard axis of axis_coords has an invalid number of coordinates");
                Py_DECREF(axis_coords_tuple);
                return NULL;
            }
            if ( PyArray_TYPE((PyArrayObject *) seqitem) != NPY_DOUBLE ) {
                PyErr_SetString(PyExc_ValueError, "a standard axis of axis_coords has an invalid type");
                Py_DECREF(axis_coords_tuple);
                return NULL;
            }
            if ( ! PyArray_ISCARRAY_RO((PyArrayObject *) seqitem) ) {
                PyErr_SetString(PyExc_ValueError, "a standard axis of axis_coords is not an appropriate ndarray");
                Py_DECREF(axis_coords_tuple);
                return NULL;
            }
            axis_coords[k] = (int *)PyArray_DATA((PyArrayObject *) seqitem);
            FORTRAN(get_axis_num)(&(axis_nums[k]), &(axis_starts[k]), &(axis_ends[k]), axis_names[k],
                          axis_units[k], axis_coords[k], &(num_coords[k]), &(axis_types[k]),
                          errmsg, &len_errmsg, strlen(axis_names[k]), strlen(axis_units[k]), 2048);
            if ( len_errmsg > 0 ) {
                errmsg[len_errmsg] = '\0';
                PyErr_SetString(PyExc_ValueError, errmsg);
                Py_DECREF(axis_coords_tuple);
                return NULL;
            }
            break;
        case AXISTYPE_TIME:
            /* int32 (N,6)-ndarray containing component time values; the calendar given in axis_units */
            /* PyArray_Size returns 0 if the object is not an appropriate type */
            /* ISCARRAY_RO checks if it is C-contiguous, aligned and in machine byte-order */
            num_coords[k] = PyArray_Size(seqitem);
            if ( (num_coords[k] < 1) || ((num_coords[k] % 6) != 0) ) {
                PyErr_SetString(PyExc_ValueError, "an absolute-time axis of axis_coords has an invalid number of coordinates");
                Py_DECREF(axis_coords_tuple);
                return NULL;
            }
            if ( (PyArray_TYPE((PyArrayObject *) seqitem) != NPY_INT) &&
                 ((PyArray_TYPE((PyArrayObject *) seqitem) != NPY_LONG) || (NPY_SIZEOF_LONG != 4)) ) {
                PyErr_SetString(PyExc_ValueError, "an absolute-time axis of axis_coords has an invalid type");
                Py_DECREF(axis_coords_tuple);
                return NULL;
            }
            if ( ! PyArray_ISCARRAY_RO((PyArrayObject *) seqitem) ) {
                PyErr_SetString(PyExc_ValueError, "an absolute-time axis of axis_coords is not an appropriate ndarray");
                Py_DECREF(axis_coords_tuple);
                return NULL;
            }
            num_coords[k] /= 6;
            if ( strcmp(axis_units[k], CALTYPE_NONE_STR) == 0 ) {
                calendar_type = CALTYPE_NONE;
            }
            else if ( strcmp(axis_units[k], CALTYPE_360DAY_STR) == 0 ) {
                calendar_type = CALTYPE_360DAY;
            }
            else if ( strcmp(axis_units[k], CALTYPE_NOLEAP_STR) == 0 ) {
                calendar_type = CALTYPE_NOLEAP;
            }
            else if ( strcmp(axis_units[k], CALTYPE_GREGORIAN_STR) == 0 ) {
                calendar_type = CALTYPE_GREGORIAN;
            }
            else if ( strcmp(axis_units[k], CALTYPE_JULIAN_STR) == 0 ) {
                calendar_type = CALTYPE_JULIAN;
            }
            else if ( strcmp(axis_units[k], CALTYPE_ALLLEAP_STR) == 0 ) {
                calendar_type = CALTYPE_ALLLEAP;
            }
            else {
                PyErr_SetString(PyExc_ValueError, "unknown calendar");
                Py_DECREF(axis_coords_tuple);
                return NULL;
            }
            axis_coords[k] = (int *)PyArray_DATA((PyArrayObject *) seqitem);
            FORTRAN(get_time_axis_num)(&(axis_nums[k]), &(axis_starts[k]), &(axis_ends[k]),
                               axis_names[k], &calendar_type, axis_coords[k], &(num_coords[k]),
                               errmsg, &len_errmsg, strlen(axis_names[k]), 2048);
            if ( len_errmsg > 0 ) {
                errmsg[len_errmsg] = '\0';
                PyErr_SetString(PyExc_ValueError, errmsg);
                Py_DECREF(axis_coords_tuple);
                return NULL;
            }
            break;
        case AXISTYPE_NORMAL:
            /* axis normal to the results - ignore sequence item (probably None) */
            axis_nums[k] = 0;   /* ferret.parm value for a normal line (mnormal) */
            axis_starts[k] = 0;
            axis_ends[k] = 0;
            break;
        default:
            PyErr_SetString(PyExc_ValueError, "Unexpected axis_type when processing axis coordinates");
            Py_DECREF(axis_coords_tuple);
            return NULL;
        }
    }

    /* The information in axis_coords_tuple no longer needed */
    Py_DECREF(axis_coords_tuple);

    /* Assign the data in the XPYVAR_INFO common block */
    len_codename = strlen(codename);
    len_title = strlen(title);
    len_units = strlen(units);
    len_dset = strlen(dset);
    FORTRAN(add_pystat_var)(&data_ndarray, codename, title, units, &bdfval, dset,
                    axis_nums, axis_starts, axis_ends, errmsg, &len_errmsg,
                    len_codename, len_title, len_units, len_dset, 2048);
    if ( len_errmsg > 0 ) {
        errmsg[len_errmsg] = '\0';
        PyErr_SetString(PyExc_ValueError, errmsg);
        return NULL;
    }

    /*
     * Increase the reference count to data_ndarray to keep it around.
     * A pointer to it is stored in the XPYVAR_INFO common block.
     * The reference count will be decremented by Ferret when no longer needed.
     */
    Py_INCREF(data_ndarray);

    Py_INCREF(Py_None);
    return Py_None;
}

static char pyferretGetStrDataDocstring[] =
    "Returns the String data array described in the argument. \n"
    "\n"
    "Required arguments: \n"
    "    name = <string>: the name of the String data array to return \n"
    "\n"
    "Optional arguments: \n"
    "    (none) \n"
    "\n"
    "Returns: \n"
    "    A tuple containing: \n"
    "        a NumPy String ndarray containing a copy of the numeric data requested, \n"
    "        a NumPy String ndarray containing the bad-data-flag value for the data, \n"
    "        a tuple of six integers giving the AXISTYPE codes of the axes, \n"
    "        a tuple of six strings giving the names of the axes, \n"
    "        a tuple of six strings giving the units of a non-calendar-time data axis, or \n"
    "                                       the CALTYPE_ calendar name of a calendar-time axis, \n"
    "        a tuple of six ndarrays giving the coordinates for the data axes \n"
    "            (ndarray of N doubles for non-calendar-time, non-normal axes, \n"
    "             ndarray of (N,6) integers for calendar-time axes, or \n"
    "             None for normal axes) \n"
    "\n"
    "Raises: \n"
    "    ValueError if the data name is invalid \n"
    "    MemoryError if Ferret has not been started or has been stopped \n";

static PyObject *pyferretGetStrData(PyObject *self, PyObject *args, PyObject *kwds)
{
    static char   *argNames[] = {"name", NULL};
    char          *name;
    int            lendataname;
    char           dataname[1024];
    char         **arraystart;
    int            memlo[MAX_FERRET_NDIM], memhi[MAX_FERRET_NDIM];
    int            steplo[MAX_FERRET_NDIM], stephi[MAX_FERRET_NDIM], incr[MAX_FERRET_NDIM];
    AXISTYPE       axis_types[MAX_FERRET_NDIM];
    char           errmsg[2112];
    int            lenerrmsg;
    int            i, j, k, l, m, n, q;
    npy_intp       shape[MAX_FERRET_NDIM];
    npy_intp       new_shape[2];
    int            strides[MAX_FERRET_NDIM];
    PyArrayObject *data_ndarray;
    int            factor;
    char          *strptr;
    int            maxstrlen;
    int            thisstrlen;
    PyArray_Descr *strarraydescript;
    char          *npydata;
    PyArrayObject *badval_ndarray;
    PyArrayObject *axis_coords[MAX_FERRET_NDIM];
    char           axis_units[MAX_FERRET_NDIM][64];
    char           axis_names[MAX_FERRET_NDIM][64];
    CALTYPE        calendar_type;

    /* If not initialized, raise a MemoryError */
    if ( ! ferretInitialized ) {
        PyErr_SetString(PyExc_MemoryError, "Ferret not started");
        return NULL;
    }

    /* Parse the arguments, checking if an Exception was raised */
    if ( ! PyArg_ParseTupleAndKeywords(args, kwds, "s", argNames, &name) )
        return NULL;

    /* Make a copy of dataname just to be sure it isn't altered */
    lendataname = strlen(name);
    if ( lendataname > 1020 ) {
        PyErr_SetString(PyExc_ValueError, "name too long");
        return NULL;
    }
    strcpy(dataname, name);

    /*
     * Retrieve the memory parameters describing the data array requested.
     * Assumes Unix standard for passing strings to Fortran (appended array lengths).
     */
    FORTRAN(get_str_data_array_params)(dataname, &lendataname, &arraystart, 
                               memlo, memhi, steplo, stephi, incr, axis_types,
                               errmsg, &lenerrmsg, 1024, 2112);
    if ( lenerrmsg > 0 ) {
        errmsg[lenerrmsg] = '\0';
        PyErr_SetString(PyExc_ValueError, errmsg);
        return NULL;
    }

    /* Get the shape of the array */
    for (k = 0; k < MAX_FERRET_NDIM; k++)
        shape[k] = (npy_intp) ((stephi[k] - steplo[k] + incr[k]) / (incr[k]));

    /* Get the strides through the memory (as a double *) */
    strides[0] = 1;
    for (k = 1; k < MAX_FERRET_NDIM; k++)
        strides[k] = strides[k-1] * (memhi[k-1] - memlo[k-1] + 1);

    /* Get the actual starting point in the array */
    for (k = 0; k < MAX_FERRET_NDIM; k++)
        arraystart += (strides[k]) * (steplo[k] - memlo[k]);

    /* Convert to strides through places in memory to be read */
    for (k = 0; k < MAX_FERRET_NDIM; k++)
        strides[k] *= incr[k];

    /* 
     * Note: for string arrays, each "double" is a pointer 
     * to allocated memory containing a null-terminated string;
     * or null for missing strings.
     * This means pointer are spaced 8 bytes apart regardless of pointer size,
     * so need to double for 4-byte (32-bit) pointers.
     */
    factor = 8 / sizeof(char *);

    /* Get the maximum string length of all the strings in this array */
    /* Use PYTHON_STRING_MISSING_VALUE for missing strings */
    maxstrlen = (int)(strlen(PYTHON_STRING_MISSING_VALUE) + 1);
    for (n = 0; n < (int)(shape[5]); n++) {
      for (m = 0; m < (int)(shape[4]); m++) {
        for (l = 0; l < (int)(shape[3]); l++) {
          for (k = 0; k < (int)(shape[2]); k++) {
            for (j = 0; j < (int)(shape[1]); j++) {
              for (i = 0; i < (int)(shape[0]); i++) {
                strptr = arraystart[ factor * ( 
                                     i * strides[0] + 
                                     j * strides[1] + 
                                     k * strides[2] + 
                                     l * strides[3] +
                                     m * strides[4] +
                                     n * strides[5] ) ];
                if ( strptr != NULL ) {
                  /* add one so always null-terminated */
                  thisstrlen = (int)(strlen(strptr) + 1);
                  if ( maxstrlen < thisstrlen ) {
                    maxstrlen = thisstrlen;
                  }
                }
              }
            }
          }
        }
      }
    }

    /* Create a new NumPy String ndarray (Fortran ordering) with the same shape */
    strarraydescript = PyArray_DescrNewFromType(NPY_STRING);
    strarraydescript->elsize = maxstrlen;
    data_ndarray = (PyArrayObject *) PyArray_Empty(MAX_FERRET_NDIM, shape, strarraydescript, 1);
    if ( data_ndarray == NULL ) {
        return NULL;
    }
    /* PyArray_Empty steals the reference to strarraydescript so do not free or reuse */

    /*
     * Assign the data in the new ndarray.
     * Note: if MAX_FERRET_NDIM changes, this needs editing.
     */
    npydata = (char *)PyArray_DATA(data_ndarray);
    q = 0;
    for (n = 0; n < (int)(shape[5]); n++) {
      for (m = 0; m < (int)(shape[4]); m++) {
        for (l = 0; l < (int)(shape[3]); l++) {
          for (k = 0; k < (int)(shape[2]); k++) {
            for (j = 0; j < (int)(shape[1]); j++) {
              for (i = 0; i < (int)(shape[0]); i++) {
                strptr = arraystart[ factor * (
                                     i * strides[0] + 
                                     j * strides[1] + 
                                     k * strides[2] + 
                                     l * strides[3] +
                                     m * strides[4] +
                                     n * strides[5] ) ];
                if ( strptr == NULL ) {
                   strptr = PYTHON_STRING_MISSING_VALUE;
                }
                /* because of +1 to strlen, these will always be null-terminated */
                strncpy(&(npydata[q]), strptr, maxstrlen);
                q += maxstrlen;
              }
            }
          }
        }
      }
    }

    /* Create a new NumPy String ndarray with the bad-data-flag value */
    new_shape[0] = 1;
    strarraydescript = PyArray_DescrNewFromType(NPY_STRING);
    strarraydescript->elsize = maxstrlen;
    badval_ndarray = (PyArrayObject *) PyArray_Empty(1, new_shape, strarraydescript, 0);
    if ( badval_ndarray == NULL ) {
       Py_DECREF(data_ndarray);
       return NULL;
    }
    /* PyArray_Empty steals the reference to strarraydescript so do not free or reuse */
    npydata = (char *)PyArray_DATA(badval_ndarray);
    strncpy(npydata, PYTHON_STRING_MISSING_VALUE, maxstrlen);

    /* Create the axis coordinates array objects */
    for (k = 0; k < MAX_FERRET_NDIM; k++) {
        switch( axis_types[k] ) {
        case AXISTYPE_LONGITUDE:
        case AXISTYPE_LATITUDE:
        case AXISTYPE_LEVEL:
        case AXISTYPE_CUSTOM:
        case AXISTYPE_ABSTRACT:
            /* array of doubles, possibly with a units string */
            axis_coords[k] = (PyArrayObject *) PyArray_SimpleNew(1, &(shape[k]), NPY_DOUBLE);
            if ( axis_coords[k] == NULL ) {
                while ( k > 0 ) {
                    k--;
                    Py_DECREF(axis_coords[k]);
                }
                Py_DECREF(badval_ndarray);
		Py_DECREF(data_ndarray);
                return NULL;
            }
            /* get the coordinates and the units string */
            q = k+1;
            j = (int) shape[k];
            FORTRAN(get_data_array_coords)((double *)PyArray_DATA(axis_coords[k]), axis_units[k],
                                   axis_names[k], &q, &j, errmsg, &lenerrmsg, 64, 64, 2112);
            if ( lenerrmsg > 0 ) {
                errmsg[lenerrmsg] = '\0';
                PyErr_SetString(PyExc_ValueError, errmsg);
                Py_DECREF(axis_coords[k]);
                while ( k > 0 ) {
                    k--;
                    Py_DECREF(axis_coords[k]);
                }
                Py_DECREF(badval_ndarray);
                Py_DECREF(data_ndarray);
                return NULL;
            }
            break;
        case AXISTYPE_TIME:
            /* array of 6-tuples of integers in C order, so: [N][6] in C or (6,N) in Fortran */
            new_shape[0] = shape[k];
            new_shape[1] = 6;
            axis_coords[k] = (PyArrayObject *) PyArray_SimpleNew(2, new_shape, NPY_INT);
            if ( axis_coords[k] == NULL ) {
                while ( k > 0 ) {
                    k--;
                    Py_DECREF(axis_coords[k]);
                }
                Py_DECREF(badval_ndarray);
                Py_DECREF(data_ndarray);
                return NULL;
            }
            /* get the time coordinate integers */
            q = k+1;
            j = (int) shape[k];
            FORTRAN(get_data_array_time_coords)((int (*)[6])PyArray_DATA(axis_coords[k]), &calendar_type, axis_names[k],
                                        &q, &j, errmsg, &lenerrmsg, 64, 2112);
            if ( lenerrmsg > 0 ) {
                errmsg[lenerrmsg] = '\0';
                PyErr_SetString(PyExc_ValueError, errmsg);
                Py_DECREF(axis_coords[k]);
                while ( k > 0 ) {
                    k--;
                    Py_DECREF(axis_coords[k]);
                }
                Py_DECREF(badval_ndarray);
                Py_DECREF(data_ndarray);
                return NULL;
            }
            /* set the axis units to the name of the calendar */
            switch( calendar_type ) {
            case CALTYPE_NONE:
                strcpy(axis_units[k], CALTYPE_NONE_STR);
                break;
            case CALTYPE_360DAY:
                strcpy(axis_units[k], CALTYPE_360DAY_STR);
                break;
            case CALTYPE_NOLEAP:
                strcpy(axis_units[k], CALTYPE_NOLEAP_STR);
                break;
            case CALTYPE_GREGORIAN:
                strcpy(axis_units[k], CALTYPE_GREGORIAN_STR);
                break;
            case CALTYPE_JULIAN:
                strcpy(axis_units[k], CALTYPE_JULIAN_STR);
                break;
            case CALTYPE_ALLLEAP:
                strcpy(axis_units[k], CALTYPE_ALLLEAP_STR);
                break;
            default:
                sprintf(errmsg, "Unexpected calendar type of %d", calendar_type);
                PyErr_SetString(PyExc_ValueError, errmsg);
                Py_DECREF(axis_coords[k]);
                while ( k > 0 ) {
                    k--;
                    Py_DECREF(axis_coords[k]);
                }
                Py_DECREF(badval_ndarray);
                Py_DECREF(data_ndarray);
                return NULL;
            }
            break;
        case AXISTYPE_NORMAL:
            /* axis normal to the results - no coordinates */
            Py_INCREF(Py_None);
            axis_coords[k] = (PyArrayObject *) Py_None;
            axis_units[k][0] = '\0';
            axis_names[k][0] = '\0';
            break;
        default:
            sprintf(errmsg, "Unexpected axis type of %d", axis_types[k]);
            PyErr_SetString(PyExc_RuntimeError, errmsg);
            while ( k > 0 ) {
                k--;
                Py_DECREF(axis_coords[k]);
            }
            Py_DECREF(badval_ndarray);
            Py_DECREF(data_ndarray);
            return NULL;
        }
    }

    /*
     * Return a tuple (stealing references for PyObjects) of data_ndarray,
     * axis_types, axis_names, axis_units, and axis_coords.
     * Note: if MAX_FERRET_NDIM changes, this needs editing.
     */
    return Py_BuildValue("NN(iiiiii)(ssssss)(ssssss)(NNNNNN)", data_ndarray, badval_ndarray,
              axis_types[0], axis_types[1], axis_types[2], axis_types[3], axis_types[4], axis_types[5],
              axis_names[0], axis_names[1], axis_names[2], axis_names[3], axis_names[4], axis_names[5],
              axis_units[0], axis_units[1], axis_units[2], axis_units[3], axis_units[4], axis_units[5],
              axis_coords[0], axis_coords[1], axis_coords[2], axis_coords[3], axis_coords[4], axis_coords[5]);
}


static char pyferretStopDocstring[] =
    "Returns Ferret to its default state, shuts down Ferret, and \n"
    "releases all memory used by Ferret.  After calling this \n"
    "function do not call any Ferret functions except start, \n"
    "which will restart Ferret and re-enable the other functions. \n"
    "\n"
    "Required arguments: \n"
    "    (none) \n"
    "\n"
    "Optional arguments: \n"
    "    (none) \n"
    "\n"
    "Returns: \n"
    "    False if Ferret has not been started or has already been stopped; \n"
    "    True otherwise \n";

static PyObject *pyferretStop(PyObject *self)
{
    /* If not initialized, return False */
    if ( ! ferretInitialized ) {
        Py_INCREF(Py_False);
        return Py_False;
    }

    /* Set to uninitialized */
    ferretInitialized = 0;

    /* Release the references to the pyferret and pyferret.graphbind modules */
    Py_DECREF(pyferret_graphbind_module_pyobject);
    pyferret_graphbind_module_pyobject = NULL;
    Py_DECREF(pyferret_module_pyobject);
    pyferret_module_pyobject = NULL;

    /* Clear/reset Ferret's state and free memory allocated inside Ferret */
    FORTRAN(finalize_ferret)();

    /* Free memory allocated for PPLUS */
    FerMem_Free(pplMemory, __FILE__, __LINE__);
    pplMemory = NULL;

#ifdef MEMORYDEBUG
    (void) ReportAnyMemoryLeaks();
#endif

    /* Return True */
    Py_INCREF(Py_True);
    return Py_True;
}


static char pyferretQuitDocstring[] =
    "Returns Ferret to its default state, shuts down Ferret, and \n"
    "releases all memory used by Ferret.  After calling this \n"
    "function do not call any Ferret functions except start, \n"
    "which will restart Ferret and re-enable the other functions. \n"
    "\n"
    "Except for the return value, this function is now identical \n"
    "to the _stop method.  This method is retained for backwards \n"
    "compatibility - to be used with the atexit module to ensure \n"
    "an open viewer window does not hang Python shutdown. \n"
    "\n"
    "Required arguments: \n"
    "    (none) \n"
    "\n"
    "Optional arguments: \n"
    "    (none) \n"
    "\n"
    "Returns: \n"
    "    None \n";

static PyObject *pyferretQuit(PyObject *self)
{
    PyObject *result;

    /* Just call the _stop method */
    result = pyferretStop(self);
    Py_DECREF(result);

    /* But return None */
    Py_INCREF(Py_None);
    return Py_None;
}


static char pyefcnGetAxisCoordinatesDocstring[] =
    "Returns the \"world\" coordinates for an axis of an argument to an external function\n"
    "\n"
    "Required arguments: \n"
    "    id = <int>: the ferret id of the external function \n"
    "    arg = <int>: the index (zero based) of the argument (can use ARG1, ARG2, ..., ARG9) \n"
    "    axis = <int>: the index (zero based) of the axis (can use X_AXIS, Y_AXIS, Z_AXIS, \n"
    "                                                              T_AXIS, E_AXIS, F_AXIS) \n"
    "\n"
    "Optional arguments: \n"
    "    (none) \n"
    "\n"
    "Returns: \n"
    "    a NumPy float64 ndarray containing the \"world\" coordinates, \n"
    "    or None if the values cannot be determined at the time this was called \n"
    "\n"
    "Raises: \n"
    "    ValueError if id, arg, or axis is invalid \n";

static PyObject *pyefcnGetAxisCoordinates(PyObject *self, PyObject *args, PyObject *kwds)
{
    static char      *argNames[] = {"id", "arg", "axis", NULL};
    int               id, arg, axis;
    ExternalFunction *ef_ptr;
    int               steplo[EF_MAX_COMPUTE_ARGS][MAX_FERRET_NDIM];
    int               stephi[EF_MAX_COMPUTE_ARGS][MAX_FERRET_NDIM];
    int               incr[EF_MAX_COMPUTE_ARGS][MAX_FERRET_NDIM];
    int               lo, hi;
    npy_intp          shape[1];
    PyArrayObject    *coords_ndarray;

    /* Parse the arguments, checking if an Exception was raised */
    if ( ! PyArg_ParseTupleAndKeywords(args, kwds, "iii", argNames, &id, &arg, &axis) )
        return NULL;

    /* Check for obvious errors in the arguments passed */
    ef_ptr = ef_ptr_from_id_ptr(&id);
    if ( (ef_ptr == NULL) || ! ef_ptr->already_have_internals ) {
        PyErr_SetString(PyExc_ValueError, "Invalid ferret external function id");
        return NULL;
    }
    if ( (arg < 0) || (arg >= EF_MAX_ARGS) ||
         ((arg >= ef_ptr->internals_ptr->num_reqd_args) && ! ef_ptr->internals_ptr->has_vari_args) ) {
        PyErr_SetString(PyExc_ValueError, "Invalid argument index");
        return NULL;
    }
    if ( (axis < 0) || (axis >= MAX_FERRET_NDIM) ) {
        PyErr_SetString(PyExc_ValueError, "Invalid axis index");
        return NULL;
    }

    /* Catch seg faults from indiscriminately calling this function */
    if ( setjmp(pyefcn_jumpbuffer) != 0 ) {
        signal(SIGSEGV, pyefcn_segv_handler);
        PyErr_SetString(PyExc_ValueError, "Invalid function call - probably not from a ferret external function call");
        return NULL;
    }
    pyefcn_segv_handler = signal(SIGSEGV, pyefcn_signal_handler);
    if ( pyefcn_segv_handler == SIG_ERR ) {
        PyErr_SetString(PyExc_ValueError, "Unable to catch SIGSEGV");
        return NULL;
    }

    /* Get the subscripts for all of the arguments */
    FORTRAN(ef_get_arg_subscripts_6d)(&id, steplo, stephi, incr);

    /* Restore the original segv handler */
    signal(SIGSEGV, pyefcn_segv_handler);

    /* Check the indices for the coordinates of the desired axis of the argument */
    if ( (steplo[arg][axis] == UNSPECIFIED_INT4) || (stephi[arg][axis] == UNSPECIFIED_INT4) ||
         ((steplo[arg][axis] == 1) && (stephi[arg][axis] == ABSTRACT_AXIS_LEN)) ) {
        Py_INCREF(Py_None);
        return Py_None;
    }

    /* Create a NumPy float64 ndarray to get the memory for the coordinates */
    if ( incr[arg][axis] == 0 ) {
        if ( steplo[arg][axis] <= stephi[arg][axis] )
            incr[arg][axis] = 1;
        else
            incr[arg][axis] = -1;
    }
    shape[0] = (Py_ssize_t) ((stephi[arg][axis] - steplo[arg][axis] + incr[arg][axis]) / incr[arg][axis]);
    coords_ndarray = (PyArrayObject *) PyArray_SimpleNew(1, shape, NPY_DOUBLE);
    if ( coords_ndarray == NULL ) {
        return NULL;
    }

    /* Get the full range of world coordinates for the requested axis */
    lo = steplo[arg][axis];
    hi = stephi[arg][axis];
    arg++;
    axis++;
    FORTRAN(ef_get_coordinates)(&id, &arg, &axis, &lo, &hi, (double *)PyArray_DATA(coords_ndarray));

    return (PyObject *) coords_ndarray;
}


static char pyefcnGetAxisBoxSizesDocstring[] =
    "Returns the \"box sizes\", in \"world\" coordinate units, \n"
    "for an axis of an argument to an external function \n"
    "\n"
    "Required arguments: \n"
    "    id = <int>: the ferret id of the external function \n"
    "    arg = <int>: the index (zero based) of the argument (can use ARG1, ARG2, ..., ARG9) \n"
    "    axis = <int>: the index (zero based) of the axis (can use X_AXIS, Y_AXIS, Z_AXIS, \n"
    "                                                              T_AXIS, E_AXIS, F_AXIS) \n"
    "\n"
    "Optional arguments: \n"
    "    (none) \n"
    "\n"
    "Returns: \n"
    "    a NumPy float64 ndarray containing the \"box sizes\", \n"
    "    or None if the values cannot be determined at the time this was called \n"
    "\n"
    "Raises: \n"
    "    ValueError if id, arg, or axis is invalid \n";

static PyObject *pyefcnGetAxisBoxSizes(PyObject *self, PyObject *args, PyObject *kwds)
{
    static char      *argNames[] = {"id", "arg", "axis", NULL};
    int               id, arg, axis;
    ExternalFunction *ef_ptr;
    int               steplo[EF_MAX_COMPUTE_ARGS][MAX_FERRET_NDIM];
    int               stephi[EF_MAX_COMPUTE_ARGS][MAX_FERRET_NDIM];
    int               incr[EF_MAX_COMPUTE_ARGS][MAX_FERRET_NDIM];
    int               lo, hi;
    npy_intp          shape[1];
    PyArrayObject    *sizes_ndarray;

    /* Parse the arguments, checking if an Exception was raised */
    if ( ! PyArg_ParseTupleAndKeywords(args, kwds, "iii", argNames, &id, &arg, &axis) )
        return NULL;

    /* Check for obvious errors in the arguments passed */
    ef_ptr = ef_ptr_from_id_ptr(&id);
    if ( (ef_ptr == NULL) || ! ef_ptr->already_have_internals ) {
        PyErr_SetString(PyExc_ValueError, "Invalid ferret external function id");
        return NULL;
    }
    if ( (arg < 0) || (arg >= EF_MAX_ARGS) ||
         ((arg >= ef_ptr->internals_ptr->num_reqd_args) && ! ef_ptr->internals_ptr->has_vari_args) ) {
        PyErr_SetString(PyExc_ValueError, "Invalid argument index");
        return NULL;
    }
    if ( (axis < 0) || (axis >= MAX_FERRET_NDIM) ) {
        PyErr_SetString(PyExc_ValueError, "Invalid axis index");
        return NULL;
    }

    /* Catch seg faults from indiscriminately calling this function */
    if ( setjmp(pyefcn_jumpbuffer) != 0 ) {
        signal(SIGSEGV, pyefcn_segv_handler);
        PyErr_SetString(PyExc_ValueError, "Invalid function call - probably not from a ferret external function call");
        return NULL;
    }
    pyefcn_segv_handler = signal(SIGSEGV, pyefcn_signal_handler);
    if ( pyefcn_segv_handler == SIG_ERR ) {
        PyErr_SetString(PyExc_ValueError, "Unable to catch SIGSEGV");
        return NULL;
    }

    /* Get the subscripts for all of the arguments */
    FORTRAN(ef_get_arg_subscripts_6d)(&id, steplo, stephi, incr);

    /* Restore the original segv handler */
    signal(SIGSEGV, pyefcn_segv_handler);

    /* Check the indices for the coordinates of the desired axis of the argument */
    if ( (steplo[arg][axis] == UNSPECIFIED_INT4) || (stephi[arg][axis] == UNSPECIFIED_INT4) ||
         ((steplo[arg][axis] == 1) && (stephi[arg][axis] == ABSTRACT_AXIS_LEN)) ) {
        Py_INCREF(Py_None);
        return Py_None;
    }

    /* Create a NumPy float64 ndarray to get the memory for the box sizes */
    if ( incr[arg][axis] == 0 ) {
        if ( steplo[arg][axis] <= stephi[arg][axis] )
            incr[arg][axis] = 1;
        else
            incr[arg][axis] = -1;
    }
    shape[0] = (Py_ssize_t) ((stephi[arg][axis] - steplo[arg][axis] + incr[arg][axis]) / incr[arg][axis]);
    sizes_ndarray = (PyArrayObject *) PyArray_SimpleNew(1, shape, NPY_DOUBLE);
    if ( sizes_ndarray == NULL ) {
        return NULL;
    }

    /* Get the full range of box sizes for the requested axis */
    lo = steplo[arg][axis];
    hi = stephi[arg][axis];
    arg++;
    axis++;
    FORTRAN(ef_get_box_size)(&id, &arg, &axis, &lo, &hi, (double *)PyArray_DATA(sizes_ndarray));

    return (PyObject *) sizes_ndarray;
}


static char pyefcnGetAxisBoxLimitsDocstring[] =
    "Returns the \"box limits\", in \"world\" coordinate units, \n"
    "for an axis of an argument to an external function \n"
    "\n"
    "Required arguments: \n"
    "    id = <int>: the ferret id of the external function \n"
    "    arg = <int>: the index (zero based) of the argument (can use ARG1, ARG2, ..., ARG9) \n"
    "    axis = <int>: the index (zero based) of the axis (can use X_AXIS, Y_AXIS, Z_AXIS, \n"
    "                                                              T_AXIS, E_AXIS, F_AXIS) \n"
    "\n"
    "Optional arguments: \n"
    "    (none) \n"
    "\n"
    "Returns: \n"
    "    a tuple of two NumPy float64 ndarrays containing the low and high \"box limits\", \n"
    "    or None if the values cannot be determined at the time this was called \n"
    "\n"
    "Raises: \n"
    "    ValueError if id, arg, or axis is invalid \n";

static PyObject *pyefcnGetAxisBoxLimits(PyObject *self, PyObject *args, PyObject *kwds)
{
    static char      *argNames[] = {"id", "arg", "axis", NULL};
    int               id, arg, axis;
    ExternalFunction *ef_ptr;
    int               steplo[EF_MAX_COMPUTE_ARGS][MAX_FERRET_NDIM];
    int               stephi[EF_MAX_COMPUTE_ARGS][MAX_FERRET_NDIM];
    int               incr[EF_MAX_COMPUTE_ARGS][MAX_FERRET_NDIM];
    int               lo, hi;
    npy_intp          shape[1];
    PyArrayObject    *low_limits_ndarray, *high_limits_ndarray;

    /* Parse the arguments, checking if an Exception was raised */
    if ( ! PyArg_ParseTupleAndKeywords(args, kwds, "iii", argNames, &id, &arg, &axis) )
        return NULL;

    /* Check for obvious errors in the arguments passed */
    ef_ptr = ef_ptr_from_id_ptr(&id);
    if ( (ef_ptr == NULL) || ! ef_ptr->already_have_internals ) {
        PyErr_SetString(PyExc_ValueError, "Invalid ferret external function id");
        return NULL;
    }
    if ( (arg < 0) || (arg >= EF_MAX_ARGS) ||
         ((arg >= ef_ptr->internals_ptr->num_reqd_args) && ! ef_ptr->internals_ptr->has_vari_args) ) {
        PyErr_SetString(PyExc_ValueError, "Invalid argument index");
        return NULL;
    }
    if ( (axis < 0) || (axis >= MAX_FERRET_NDIM) ) {
        PyErr_SetString(PyExc_ValueError, "Invalid axis index");
        return NULL;
    }

    /* Catch seg faults from indiscriminately calling this function */
    if ( setjmp(pyefcn_jumpbuffer) != 0 ) {
        signal(SIGSEGV, pyefcn_segv_handler);
        PyErr_SetString(PyExc_ValueError, "Invalid function call - probably not from a ferret external function call");
        return NULL;
    }
    pyefcn_segv_handler = signal(SIGSEGV, pyefcn_signal_handler);
    if ( pyefcn_segv_handler == SIG_ERR ) {
        PyErr_SetString(PyExc_ValueError, "Unable to catch SIGSEGV");
        return NULL;
    }

    /* Get the subscripts for all of the arguments */
    FORTRAN(ef_get_arg_subscripts_6d)(&id, steplo, stephi, incr);

    /* Restore the original segv handler */
    signal(SIGSEGV, pyefcn_segv_handler);

    /* Check the indices for the coordinates of the desired axis of the argument */
    if ( (steplo[arg][axis] == UNSPECIFIED_INT4) || (stephi[arg][axis] == UNSPECIFIED_INT4) ||
         ((steplo[arg][axis] == 1) && (stephi[arg][axis] == ABSTRACT_AXIS_LEN)) ) {
        Py_INCREF(Py_None);
        return Py_None;
    }

    /* Create two NumPy float64 ndarrays to get the memory for the box limits */
    if ( incr[arg][axis] == 0 ) {
        if ( steplo[arg][axis] <= stephi[arg][axis] )
            incr[arg][axis] = 1;
        else
            incr[arg][axis] = -1;
    }
    shape[0] = (Py_ssize_t) ((stephi[arg][axis] - steplo[arg][axis] + incr[arg][axis]) / incr[arg][axis]);
    low_limits_ndarray = (PyArrayObject *) PyArray_SimpleNew(1, shape, NPY_DOUBLE);
    if ( low_limits_ndarray == NULL ) {
        return NULL;
    }
    high_limits_ndarray = (PyArrayObject *) PyArray_SimpleNew(1, shape, NPY_DOUBLE);
    if ( high_limits_ndarray == NULL ) {
        Py_DECREF(low_limits_ndarray);
        return NULL;
    }

    /* Get the full range of box limits for the requested axis */
    lo = steplo[arg][axis];
    hi = stephi[arg][axis];
    arg++;
    axis++;
    FORTRAN(ef_get_box_limits)(&id, &arg, &axis, &lo, &hi, (double *)PyArray_DATA(low_limits_ndarray), 
                                                   (double *)PyArray_DATA(high_limits_ndarray));

    return Py_BuildValue("NN", low_limits_ndarray, high_limits_ndarray); /* Steals the references to the two ndarrays */
}


static char pyefcnGetAxisInfoDocstring[] =
    "Returns information about the axis of an argument to an external function \n"
    "\n"
    "Required arguments: \n"
    "    id = <int>: the ferret id of the external function \n"
    "    arg = <int>: the index (zero based) of the argument (can use ARG1, ARG2, ..., ARG9) \n"
    "    axis = <int>: the index (zero based) of the axis (can use X_AXIS, Y_AXIS, Z_AXIS, \n"
    "                                                              T_AXIS, E_AXIS, F_AXIS) \n"
    "\n"
    "Optional arguments: \n"
    "    (none) \n"
    "\n"
    "Returns: \n"
    "    a dictionary defining the following keys: \n"
    "        \"name\": name string for the axis coordinate \n"
    "        \"unit\": name string for the axis unit \n"
    "        \"backwards\": boolean - reversed axis? \n"
    "        \"modulo\": float - length of a modulo (periodic,wrapping) axis, \n"
    "                          or 0.0 if not a modulo axis\n"
    "        \"regular\": boolean - evenly spaced axis? \n"
    "        \"size\": number of coordinates on this axis, or -1 if the value \n"
    "                  cannot be determined at the time this was called \n"
    "\n"
    "Raises: \n"
    "    ValueError if id, arg, or axis is invalid \n";

static PyObject *pyefcnGetAxisInfo(PyObject *self, PyObject *args, PyObject *kwds)
{
    static char      *argNames[] = {"id", "arg", "axis", NULL};
    int               id, arg, axis;
    ExternalFunction *ef_ptr;
    int               steplo[EF_MAX_COMPUTE_ARGS][MAX_FERRET_NDIM];
    int               stephi[EF_MAX_COMPUTE_ARGS][MAX_FERRET_NDIM];
    int               incr[EF_MAX_COMPUTE_ARGS][MAX_FERRET_NDIM];
    int               num_coords;
    char              name[80];
    char              unit[80];
    int               backwards;
    int               modulo;
    int               regular;
    double            modulolen;
    PyObject         *backwards_bool;
    PyObject         *regular_bool;

    /* Parse the arguments, checking if an Exception was raised */
    if ( ! PyArg_ParseTupleAndKeywords(args, kwds, "iii", argNames, &id, &arg, &axis) )
        return NULL;

    /* Check for obvious errors in the arguments passed */
    ef_ptr = ef_ptr_from_id_ptr(&id);
    if ( (ef_ptr == NULL) || ! ef_ptr->already_have_internals ) {
        PyErr_SetString(PyExc_ValueError, "Invalid ferret external function id");
        return NULL;
    }
    if ( (arg < 0) || (arg >= EF_MAX_ARGS) ||
         ((arg >= ef_ptr->internals_ptr->num_reqd_args) && ! ef_ptr->internals_ptr->has_vari_args) ) {
        PyErr_SetString(PyExc_ValueError, "Invalid argument index");
        return NULL;
    }
    if ( (axis < 0) || (axis >= MAX_FERRET_NDIM) ) {
        PyErr_SetString(PyExc_ValueError, "Invalid axis index");
        return NULL;
    }

    /* Catch seg faults from indiscriminately calling this function */
    if ( setjmp(pyefcn_jumpbuffer) != 0 ) {
        signal(SIGSEGV, pyefcn_segv_handler);
        PyErr_SetString(PyExc_ValueError, "Invalid function call - probably not from a ferret external function call");
        return NULL;
    }
    pyefcn_segv_handler = signal(SIGSEGV, pyefcn_signal_handler);
    if ( pyefcn_segv_handler == SIG_ERR ) {
        PyErr_SetString(PyExc_ValueError, "Unable to catch SIGSEGV");
        return NULL;
    }

    /* Get the subscripts for all of the arguments */
    FORTRAN(ef_get_arg_subscripts_6d)(&id, steplo, stephi, incr);

    /* Restore the original segv handler */
    signal(SIGSEGV, pyefcn_segv_handler);

    /* Check the indices for the coordinates of the desired axis of the argument */
    if ( (steplo[arg][axis] == UNSPECIFIED_INT4) || (stephi[arg][axis] == UNSPECIFIED_INT4) ||
         ((steplo[arg][axis] == 1) && (stephi[arg][axis] == ABSTRACT_AXIS_LEN)) ) {
        num_coords = -1;
    }
    else {
        if ( incr[arg][axis] == 0 ) {
            if ( steplo[arg][axis] <= stephi[arg][axis] )
                incr[arg][axis] = 1;
            else
                incr[arg][axis] = -1;
        }
        num_coords = (stephi[arg][axis] - steplo[arg][axis] + incr[arg][axis]) / incr[arg][axis];
    }

    /* Get the rest of the info */
    arg++;
    axis++;
    FORTRAN(ef_get_single_axis_info)(&id, &arg, &axis, name, unit, &backwards, &modulo, &regular, 80, 80);
    if ( modulo != 0 )
        FORTRAN(ef_get_axis_modulo_len)(&id, &arg, &axis, &modulolen);
    else
        modulolen = 0.0;

    /* Assign the Python bool objects */
    if ( backwards != 0 )
        backwards_bool = Py_True;
    else
        backwards_bool = Py_False;
    if ( regular != 0 )
        regular_bool = Py_True;
    else
        regular_bool = Py_False;

    /* Using O for the booleans to increment the references to these objects */
    return Py_BuildValue("{sssssOsdsOsi}", "name", name, "unit", unit,
                                           "backwards", backwards_bool, "modulo", modulolen,
                                           "regular", regular_bool, "size", num_coords);
}


static char pyefcnGetArgOneValDocstring[] =
    "Returns the value of the indicated FLOAT_ONEVAL or STRING_ONEVAL argument. \n"
    "\n"
    "Required arguments: \n"
    "    id = <int>: the ferret id of the external function \n"
    "    arg = <int>: the index (zero based) of the argument (can use ARG1, ARG2, ..., ARG9) \n"
    "\n"
    "Optional arguments: \n"
    "    (none) \n"
    "\n"
    "Returns: \n"
    "    the value of the argument, either as a float (if a FLOAT_ONEVAL) \n"
    "    or a string (if STRING_ONEVAL) \n"
    "\n"
    "Raises: \n"
    "    ValueError if id or arg is invalid, or if the argument type is not \n"
    "               FLOAT_ONEVAL or STRING_ONEVAL \n";

static PyObject *pyefcnGetArgOneVal(PyObject *self, PyObject *args, PyObject *kwds)
{
    static char      *argNames[] = {"id", "arg", NULL};
    int               id, arg;
    ExternalFunction *ef_ptr;
    PyObject         *modname;
    PyObject         *usermod;
    PyObject         *initdict;
    PyObject         *typetuple;
    PyObject         *typeobj;
    double            float_val;
    PyObject         *valobj;
    char              str_val[2048];
    int               valtype;
    int               k;

    /* Parse the arguments, checking if an Exception was raised */
    if ( ! PyArg_ParseTupleAndKeywords(args, kwds, "ii", argNames, &id, &arg) )
        return NULL;

    /* Check for obvious errors in the arguments passed */
    ef_ptr = ef_ptr_from_id_ptr(&id);
    if ( (ef_ptr == NULL) || ! ef_ptr->already_have_internals ) {
        PyErr_SetString(PyExc_ValueError, "Invalid ferret external function id");
        return NULL;
    }
    if ( (arg < 0) || (arg >= EF_MAX_ARGS) ||
         ((arg >= ef_ptr->internals_ptr->num_reqd_args) && ! ef_ptr->internals_ptr->has_vari_args) ) {
        PyErr_SetString(PyExc_ValueError, "Invalid argument index");
        return NULL;
    }

    /* Get the Python module (should already be imported) */
#if PY_MAJOR_VERSION > 2
    modname = PyUnicode_FromString(ef_ptr->path);
#else
    modname = PyString_FromString(ef_ptr->path);
#endif
    if ( modname == NULL )
        return NULL;
    usermod = PyImport_Import(modname);
    Py_DECREF(modname);
    if ( usermod == NULL )
        return NULL;

     /* Call the initialization method to get the argument types */
    initdict = PyObject_CallMethod(usermod, INIT_METHOD_NAME, "i", id);
    Py_DECREF(usermod);
    if ( initdict == NULL )
        return NULL;
    typetuple = PyDict_GetItemString(initdict, "argtypes"); /* borrowed reference */
    if ( typetuple == NULL ) {
        /* Key not present; no exception raised */
        Py_DECREF(initdict);
        PyErr_SetString(PyExc_ValueError, "argtype is neither FLOAT_ONEVAL nor STRING_ONEVAL");
        return NULL;
    }

    /* Get the type of this argument */
    typeobj = PySequence_GetItem(typetuple, (Py_ssize_t) arg);
    if ( typeobj == NULL ) {
        PyErr_Clear();
        Py_DECREF(initdict);
        PyErr_SetString(PyExc_ValueError, "argtype is neither FLOAT_ONEVAL nor STRING_ONEVAL");
        return NULL;
    }
#if PY_MAJOR_VERSION > 2
    valtype = (int) PyLong_AsLong(typeobj);
#else
    valtype = (int) PyInt_AsLong(typeobj);
#endif
    switch( valtype ) {
        case FLOAT_ONEVAL:
            k = arg + 1;
            FORTRAN(ef_get_one_val)(&id, &k, &float_val);
            valobj = PyFloat_FromDouble(float_val);
            break;
        case STRING_ONEVAL:
        case STRING_ARG:
            k = arg + 1;
            /* Assumes gcc standard for passing Hollerith strings */
            FORTRAN(ef_get_arg_string)(&id, &k, str_val, 2048);
            for (k = 2048; k > 0; k--)
                if ( ! isspace(str_val[k-1]) )
                    break;
#if PY_MAJOR_VERSION > 2
            valobj = PyUnicode_FromStringAndSize(str_val, k);
#else
            valobj = PyString_FromStringAndSize(str_val, k);
#endif
            break;
        default:
            PyErr_Clear();   /* Just to be safe */
            PyErr_SetString(PyExc_ValueError, "argtype is neither FLOAT_ONEVAL nor STRING_ONEVAL");
            valobj = NULL;
    }
    Py_DECREF(typeobj);
    Py_DECREF(initdict);
    return valobj;
}


/* List of Python functions and their docstrings available in this module */
static struct PyMethodDef pyferretMethods[] = {
    {"_start", (PyCFunction) pyferretStart, METH_VARARGS | METH_KEYWORDS, pyferretStartDocstring},
    {"_run", (PyCFunction) pyferretRunCommand, METH_VARARGS | METH_KEYWORDS, pyferretRunCommandDocstring},
    {"_get", (PyCFunction) pyferretGetData, METH_VARARGS | METH_KEYWORDS, pyferretGetDataDocstring},
    {"_getstrdata", (PyCFunction) pyferretGetStrData, METH_VARARGS | METH_KEYWORDS, pyferretGetStrDataDocstring},
    {"_put", (PyCFunction) pyferretPutData, METH_VARARGS | METH_KEYWORDS, pyferretPutDataDocstring},
    {"_stop", (PyCFunction) pyferretStop, METH_NOARGS, pyferretStopDocstring},
    {"_quit", (PyCFunction) pyferretQuit, METH_NOARGS, pyferretQuitDocstring},
    {"_get_axis_coordinates", (PyCFunction) pyefcnGetAxisCoordinates, METH_VARARGS | METH_KEYWORDS, pyefcnGetAxisCoordinatesDocstring},
    {"_get_axis_box_sizes", (PyCFunction) pyefcnGetAxisBoxSizes, METH_VARARGS | METH_KEYWORDS, pyefcnGetAxisBoxSizesDocstring},
    {"_get_axis_box_limits", (PyCFunction) pyefcnGetAxisBoxLimits, METH_VARARGS | METH_KEYWORDS, pyefcnGetAxisBoxLimitsDocstring},
    {"_get_axis_info", (PyCFunction) pyefcnGetAxisInfo, METH_VARARGS | METH_KEYWORDS, pyefcnGetAxisInfoDocstring},
    {"_get_arg_one_val", (PyCFunction) pyefcnGetArgOneVal, METH_VARARGS | METH_KEYWORDS, pyefcnGetArgOneValDocstring},
    {NULL, (PyCFunction) NULL, 0, NULL}
};

/* Add constants to the libpyferret module */
static void AddConstantsToPyFerret(PyObject *mod)
{
    char names[64][32];
    int  values[64];
    int  numvals;
    int  k;

    /* Add ferret parameter values */
    FORTRAN(get_ferret_params)(names, values, &numvals);
    for (k = 0; k < numvals; k++) {
        PyModule_AddIntConstant(mod, names[k], values[k]);
    }

    /* Add parameters for the python EF argument types */
    PyModule_AddIntConstant(mod, "FLOAT_ARRAY", FLOAT_ARRAY);
    PyModule_AddIntConstant(mod, "FLOAT_ONEVAL", FLOAT_ONEVAL);
    PyModule_AddIntConstant(mod, "STRING_ARRAY", STRING_ARRAY);
    PyModule_AddIntConstant(mod, "STRING_ONEVAL", STRING_ONEVAL);

    /* Add parameters for the python axis functions */
    PyModule_AddIntConstant(mod, "X_AXIS", 0);
    PyModule_AddIntConstant(mod, "Y_AXIS", 1);
    PyModule_AddIntConstant(mod, "Z_AXIS", 2);
    PyModule_AddIntConstant(mod, "T_AXIS", 3);
    PyModule_AddIntConstant(mod, "E_AXIS", 4);
    PyModule_AddIntConstant(mod, "F_AXIS", 5);
    PyModule_AddIntConstant(mod, "ARG1", 0);
    PyModule_AddIntConstant(mod, "ARG2", 1);
    PyModule_AddIntConstant(mod, "ARG3", 2);
    PyModule_AddIntConstant(mod, "ARG4", 3);
    PyModule_AddIntConstant(mod, "ARG5", 4);
    PyModule_AddIntConstant(mod, "ARG6", 5);
    PyModule_AddIntConstant(mod, "ARG7", 6);
    PyModule_AddIntConstant(mod, "ARG8", 7);
    PyModule_AddIntConstant(mod, "ARG9", 8);

    /* Parameters for interpreting axis data */
    PyModule_AddIntConstant(mod, "AXISTYPE_LONGITUDE",    AXISTYPE_LONGITUDE);
    PyModule_AddIntConstant(mod, "AXISTYPE_LATITUDE",     AXISTYPE_LATITUDE);
    PyModule_AddIntConstant(mod, "AXISTYPE_LEVEL",        AXISTYPE_LEVEL);
    PyModule_AddIntConstant(mod, "AXISTYPE_TIME",         AXISTYPE_TIME);
    PyModule_AddIntConstant(mod, "AXISTYPE_CUSTOM",       AXISTYPE_CUSTOM);
    PyModule_AddIntConstant(mod, "AXISTYPE_ABSTRACT",     AXISTYPE_ABSTRACT);
    PyModule_AddIntConstant(mod, "AXISTYPE_NORMAL",       AXISTYPE_NORMAL);
    PyModule_AddIntConstant(mod, "TIMEARRAY_DAYINDEX",    TIMEARRAY_DAYINDEX);
    PyModule_AddIntConstant(mod, "TIMEARRAY_MONTHINDEX",  TIMEARRAY_MONTHINDEX);
    PyModule_AddIntConstant(mod, "TIMEARRAY_YEARINDEX",   TIMEARRAY_YEARINDEX);
    PyModule_AddIntConstant(mod, "TIMEARRAY_HOURINDEX",   TIMEARRAY_HOURINDEX);
    PyModule_AddIntConstant(mod, "TIMEARRAY_MINUTEINDEX", TIMEARRAY_MINUTEINDEX);
    PyModule_AddIntConstant(mod, "TIMEARRAY_SECONDINDEX", TIMEARRAY_SECONDINDEX);

    /* Parameters for the calendar types */
    PyModule_AddStringConstant(mod, "CALTYPE_360DAY", CALTYPE_360DAY_STR);
    PyModule_AddStringConstant(mod, "CALTYPE_NOLEAP", CALTYPE_NOLEAP_STR);
    PyModule_AddStringConstant(mod, "CALTYPE_GREGORIAN", CALTYPE_GREGORIAN_STR);
    PyModule_AddStringConstant(mod, "CALTYPE_JULIAN", CALTYPE_JULIAN_STR);
    PyModule_AddStringConstant(mod, "CALTYPE_ALLLEAP", CALTYPE_ALLLEAP_STR);
    PyModule_AddStringConstant(mod, "CALTYPE_NONE", CALTYPE_NONE_STR);

    /* Parameter giving the maximum number of axis allowed in Ferret */
    PyModule_AddIntConstant(mod, "MAX_FERRET_NDIM", MAX_FERRET_NDIM);

    /* Parameter giving the String used as the missing value for String arrays */
    PyModule_AddStringConstant(mod, "STRING_MISSING_VALUE", PYTHON_STRING_MISSING_VALUE);

    /* Private parameter return value from libpyferret._run indicating the program should shut down */
    PyModule_AddIntConstant(mod, "_FERR_EXIT_PROGRAM", FERR_EXIT_PROGRAM);
}

static char pyferretModuleName[] = "libpyferret";

static char pyferretModuleDocstring[] =
"An extension module enabling the use of Ferret from Python \n";

#if PY_MAJOR_VERSION > 2

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    pyferretModuleName,
    pyferretModuleDocstring,
    -1,
    pyferretMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_libpyferret(void)
{
    PyObject *mod = PyModule_Create(&moduledef);
    if ( mod == NULL )
        return NULL;

    AddConstantsToPyFerret(mod);
    return mod;
}

#else

/* For the libpyferret module, this function must be named initlibpyferret */
PyMODINIT_FUNC initlibpyferret(void)
{
    /* Create the module with the indicated methods */
    PyObject *mod = Py_InitModule3(pyferretModuleName, pyferretMethods, pyferretModuleDocstring);
    if ( mod == NULL )
        return;

    AddConstantsToPyFerret(mod);
    return;
}

#endif

