/*
 * $Id$
 */

#ifndef UD_FORTC_H_INCLUDED
#define UD_FORTC_H_INCLUDED

#include "udposix.h"
#include <stddef.h>	/* for size_t */

/*
 *	Interface to the Unidata FORTRAN-support abstraction:
 */
UD_EXTERN_FUNC(size_t fclen, (const char *s, int max));
UD_EXTERN_FUNC(char  *fcdup, (const char *s, int max));

#endif	/* !UD_FORTC_H_INCLUDED */
