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
 * Delete a signal from a signal-mask.
 */
    int
sigdelset(mask, num)
    sigset_t	*mask;
    int		num;
{
    int		retval;

    if (num < 1 || mask == NULL) {
	errno	= EINVAL;
	retval	= -1;
    } else {
	*mask	&= ~SIGMASK(num);
	retval	= 0;
    }

    return retval;
}
