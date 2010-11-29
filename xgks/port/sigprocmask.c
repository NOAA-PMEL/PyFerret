/*
 * $Id$
 *
 * POSIX signal interface layered atop the native signal interface.
 */

#include "udposix.h"
#include <stddef.h>		/* for NULL */
#include <signal.h>
#include <errno.h>


/*
 * Process a signal-mask.
 */
    int
sigprocmask(action, in_mask, out_mask)
    int		action;
    sigset_t	*in_mask;
    sigset_t	*out_mask;
{
    int		MyErrno	= errno;

    errno	= 0;

    if (out_mask != NULL)
	*out_mask	= sigblock((sigset_t)0);

    if (in_mask != NULL) {
	switch (action) {
	case SIG_BLOCK:
	    (void)sigblock(*in_mask);
	    break;
	case SIG_UNBLOCK:
	    (void)sigsetmask(sigblock((sigset_t)0) & ~*in_mask);
	    break;
	case SIG_SETMASK:
	    (void)sigsetmask(*in_mask);
	    break;
	default:
	    errno	= EINVAL;
	}
    }

    if (errno == 0) {
	errno	= MyErrno;
	MyErrno	= 0;
    } else {
	MyErrno	= -1;
    }

    return MyErrno;
}
