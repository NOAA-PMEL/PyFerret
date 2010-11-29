/*
 * $Id$
 */

#include "udposix.h"
#include <signal.h>


/*
 * Block the signals of a signal-mask and suspend the process.
 */
    int
sigsuspend(mask)
    sigset_t	*mask;
{
#   ifdef HAVE_BSDSIGP
	return bsdsigp(*mask);
#   else
#       ifdef HAVE_SIGPAUSE
	    return sigpause(*mask);
#       else
#           include "don't know how to implement sigsuspend(2)"
#       endif
#   endif
}
