/* Make sure Python.h is included first */
#include <Python.h>
#include "EF_Util.h"

/* calls the 6D function with NO for the E and F axes */
void FORTRAN(ef_set_axis_influence)(int *id_ptr, int *arg, int *xax, int *yax, int *zax, int *tax)
{
   int eax_val, fax_val;

   eax_val = NO;
   fax_val = NO;
   FORTRAN(ef_set_axis_influence_6d)(id_ptr, arg, xax, yax, zax, tax, &eax_val, &fax_val);
}

