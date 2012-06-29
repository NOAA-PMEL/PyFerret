#include "EF_Util.h"

/* Calls the 6D function with 1 for the low and high dimensions of the E and F axes */
void FORTRAN(ef_set_work_array_dims)(int *id_ptr, int *iarray, 
                         int *xlo, int *ylo, int *zlo, int *tlo,
                         int *xhi, int *yhi, int *zhi, int *thi)
{
   int elo, ehi, flo, fhi;

   elo = 1;
   ehi = 1;
   flo = 1;
   fhi = 1;
   FORTRAN(ef_set_work_array_dims_6d)(id_ptr, iarray, 
                                      xlo, ylo, zlo, tlo, &elo, &flo,
                                      xhi, yhi, zhi, thi, &ehi, &fhi);
}

