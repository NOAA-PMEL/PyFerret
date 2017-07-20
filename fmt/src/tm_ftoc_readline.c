/*
*
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
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE.
*
*/



/* tm_ftoc_readline -- based on "manexamp.c" in the readline distribution. */
/* c jacket  to make gnu readline callable from FORTRAN */

/* had to add ifdef check for trailing underscore in routine name
   for aix port *kob* 10/94 */

/* Readline is very slow for piped I/O, so run w/o readline for Ferret server
 * *js* 12/98
 */

/* v51 *kob* - upgraded to new version of readline, which is now seperate
   from tmap library - modified readline include to that end */

/* *acm   9/06 v600 - add stdlib.h wherever there is stdio.h for altix build*/
/*
 * *kms* 10/11 use a static memory line in server mode
 *             (eliminate many small malloc/free calls);
 *             make end-of-line clean-up cleaner and more portable;
 *             do not add lines to readline history when in server mode.
 * *kms* 10/11 change to using pyferret._readline to get the line
 *             when not in server mode.
 */

#include <Python.h> /* make sure Python.h is first */
#include <ctype.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "fmtprotos.h"
#include "ferret.h" /* for is_server */
#include "pyferret.h"

/* Static memory to contain the line read */
#define STATIC_LINE_LEN 2048
static char static_line[STATIC_LINE_LEN];

/*
 * Prompt the user, using the input argument prompt, for the next input
 * line, then read and return that line.  Calls pyferret._readline, which
 * uses the raw_input function in Python to do this so other Python
 * operations are not blocked by our own call to a readline function.
 * Returns NULL on error or EOF.
 */
static char *pyferret_readline(char *prompt)
{
    PyObject *resultobj;
    char *resultstr;
    int   resultstrlen;

    /* call pyferret._readline - a NULL prompt turns into a Python None argument */
    resultobj = PyObject_CallMethod(pyferret_module_pyobject, 
                                    "_readline", "s", prompt);
    if ( resultobj == NULL ) {
        /* Exception - should not happen but treat as if EOF */
        PyErr_Clear();
        return NULL;
    }

    /* first check if None was returned == EOF */
    if ( resultobj == Py_None ) {
        Py_DECREF(resultobj);
        return NULL;
    }

    /* get the string out of the result object */
#if PY_MAJOR_VERSION > 2
    resultstr = PyUnicode_AsUTF8(resultobj);
#else
    resultstr = PyString_AsString(resultobj);
#endif
    if ( resultstr == NULL ) {
        /* Exception (not a string object) - should not happen but treat as if EOF */
        PyErr_Clear();
        Py_DECREF(resultobj);
        return NULL;
    }

    /* just truncate if the string is too long */
    resultstrlen = strlen(resultstr);
    if ( resultstrlen >= STATIC_LINE_LEN )
        resultstrlen = STATIC_LINE_LEN - 1;

    /* trim off any trailing whitespace */
    resultstrlen--;
    while ( (resultstrlen >= 0) && isspace(resultstr[resultstrlen]) ) {
        resultstrlen--;
    }
    resultstrlen++;

    /* make a copy of the string since resultstr belongs to resultobj */
    strncpy(static_line, resultstr, resultstrlen);
    static_line[resultstrlen] = '\0';

    /* done with resultobj */
    Py_DECREF(resultobj);

    /* return the static memory copy of the line */
    return static_line;
}


/* Read a string, and return a pointer to it.  Returns NULL on EOF. */
static char *do_gets(char *prompt)
{
    char *line_read;

    /* Get a line from the user. */
    if ( FORTRAN(is_server)() ) {
        /* Server mode - just read the next line directly */
        int linelen;

        /* Prompt the user */
        fputs(prompt, stdout);
        fflush(stdout);

        /* Get the answer */
        if ( fgets(static_line, STATIC_LINE_LEN - 1, stdin) == NULL )
            return NULL;

        /* Trim off any trailing whitespace */
        linelen = strlen(static_line);
        linelen--;
        while ( (linelen >= 0) && isspace(static_line[linelen]) ) {
            linelen--;
        }
        linelen++;
        static_line[linelen] = '\0';

        /* Set the line read to the static memory line */
        line_read = static_line;
    }
    else {
        /* Use pyferret._readline to get user input */
        line_read = pyferret_readline(prompt);
    }

    return line_read;
}


/* c jacket routine to make gnu readline callable from FORTRAN */
void FORTRAN(tm_ftoc_readline)(char *prompt, char *buff)
{
  char *line_read;

    /* invoke either gets or Python readline */
    line_read = do_gets(prompt);

    /* copy the string into the buffer provided from FORTRAN */
    if ( line_read != NULL ) {
        strcpy( buff, line_read );
    }
    else {
        buff[0] = '\004';   /* ^D  */
        buff[1] = '\0';
    }

    return;
}

