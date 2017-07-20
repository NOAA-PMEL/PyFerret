/* Make sure Python.h is included first */
#include <Python.h>
#include "ferret.h"
#include "EF_Util.h"

/*
 * Set the requested size (in words) for a specific work array.
 * Calls the 6D function with 1 for the length of the E and F axes.
 */
void FORTRAN(ef_set_work_array_lens)(int *id_ptr, int *iarray,
                    int *xlen, int *ylen, int *zlen, int *tlen)
{
   int elen, flen;

   elen = 1;
   flen = 1;
   FORTRAN(ef_set_work_array_lens_6d)(id_ptr, iarray,
                                      xlen, ylen, zlen,
                                      tlen, &elen, &flen);
}

