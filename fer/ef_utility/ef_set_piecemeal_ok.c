#include "ferret.h"
#include "EF_Util.h"

/* Calls the 6D function with YES for the E and F axes */
void FORTRAN(ef_set_piecemeal_ok)(int *id_ptr, int *xax, int *yax, int *zax, int *tax)
{
   int eax_val, fax_val;

   eax_val = YES;
   fax_val = YES;
   FORTRAN(ef_set_piecemeal_ok_6d)(id_ptr, xax, yax, zax, tax, &eax_val, &fax_val);
}

