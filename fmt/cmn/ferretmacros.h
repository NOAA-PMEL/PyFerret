/* ferretmacros.h in fmt/cmn 

   Define macros for handling underscores on c-to-fortran calls
   and the ifdef for double-precision Ferret.
/* acm 12/2012 */

/* Easier way of handling FORTRAN calls with underscore/no underscore */
#ifndef FORTRAN
#ifdef NO_ENTRY_NAME_UNDERSCORES
#define FORTRAN(a) a
#else
#define FORTRAN(a) a##_
#endif
#endif

/* Easier way of handling single/double floating-point declarations */

#ifdef double_p
#define DFTYPE double
#else
#define DFTYPE float
#endif
