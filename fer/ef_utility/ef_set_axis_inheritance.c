/* Make sure Python.h is included first */
#include <Python.h>
#include "ferret.h"
#include "EF_Util.h"

/* Calls the 6D version with NORMAL for the E and F axis */
void FORTRAN(ef_set_axis_inheritance)(int *id_ptr, int *xax, int *yax, int *zax, int *tax)
{
   int eax_val, fax_val;

   eax_val = NORMAL;
   fax_val = NORMAL;
   FORTRAN(ef_set_axis_inheritance_6d)(id_ptr, xax, yax, zax, tax, &eax_val, &fax_val);
}

