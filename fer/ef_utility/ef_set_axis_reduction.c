/* Make sure Python.h is included first */
#include <Python.h>
#include "ferret.h"
#include "EF_Util.h"

/* Calls the 6D version with RETAINED for the E and F axes */
void FORTRAN(ef_set_axis_reduction)(int *id_ptr, int *xax, int *yax, int *zax, int *tax)
{
   int eax_val, fax_val;

   eax_val = RETAINED;
   fax_val = RETAINED;
   FORTRAN(ef_set_axis_reduction_6d)(id_ptr, xax, yax, zax, tax, &eax_val, &fax_val);
}

