/*
 * $Id$
 */

#include "udposix.h"
#include <stddef.h>		/* for NULL */
#include <signal.h>
#include <errno.h>


/*
 * Clear a signal-mask.
 */
    int
sigemptyset(mask)
    sigset_t	*mask;
{
    int		retval;

    if (mask == NULL) {
	errno	= EINVAL;
	retval	= -1;
    } else {
	*mask	= 0;
	retval	= 0;
    }

    return retval;
}
