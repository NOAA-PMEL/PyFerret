/*
 * $Id$
 */

#include "udposix.h"
#include <stddef.h>		/* for NULL */
#include <signal.h>
#include <errno.h>

#undef	SIGMASK
#define	SIGMASK(num)		((sigset_t)1 << (num)-1)


/*
 * Add a signal to a signal-mask.
 */
    int
sigaddset(mask, num)
    sigset_t	*mask;
    int		num;
{
    int		retval;

    if (num < 1 || mask == NULL) {
	errno	= EINVAL;
	retval	= -1;
    } else {
	*mask	|= SIGMASK(num);
	retval	= 0;
    }

    return retval;
}
