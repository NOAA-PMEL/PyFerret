#include <udposix.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <X11/Xlib.h>
#include "xgks.h"
#include "fortxgks.h"

/*
 * Strictly C-callable function for setting the name of the application.  This
 * name will be subsequently used to obtain application-specific X resources.
 */
    void
cgesspn(name, name_len)
    char           *name;
    size_t	    name_len;
{
    int             i;
    char           *p;

    /*
     * Adjust "length" to account for possibly trailing blanks -- an annoying
     * characteristic of FORTRAN character variables.
     */
    for (i = 0; i < name_len; ++i)
	if (name[i] == ' ') {
	    name_len = i;
	    break;
	}

    /* Copy the name into local storage. */
    (void) strncpy(p = (char *) malloc((size_t) (name_len + 1)), name, 
		   name_len);
    p[name_len] = 0;

    /* Call the nominal C function.  */
    (void) gescsetprogname(p);

    /* Free local storage.  */
    (void) free((voidp)p);
}


