/*
 * $Id$
 */

/*LINTLIBRARY*/


#include "udposix.h"
#include "stdlib.h"


    int
atexit(func)
    void	(*func)UD_PROTO((void));
{
#   ifdef HAVE_ON_EXIT
	return on_exit(func, (char*)0);
#   endif
}
