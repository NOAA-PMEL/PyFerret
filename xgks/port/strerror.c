/*
 * $Id$
 */

/*LINTLIBRARY*/


#include "udposix.h"
#include "string.h"


/*
 * Return the string corresponding to a given error number.
 */
    char*
strerror(errnum)
    int		errnum;
{
    extern int	sys_nerr;
    extern char	*sys_errlist[];

    return errnum >= 0 && errnum < sys_nerr
		? sys_errlist[errnum]
		: "unknown error";
}
