/* Make sure Python.h is included first */
#include <Python.h>
#include <stdlib.h>
#include "ferret.h"
#include "EF_Util.h"

void FORTRAN(ef_set_piecemeal_ok_6d)(int *id_ptr, int *xax, int *yax, int *zax,
                                                  int *tax, int *eax, int *fax)
{
   ExternalFunction *ef_ptr;

   ef_ptr = ef_ptr_from_id_ptr(id_ptr);
   if ( ef_ptr == NULL ) {
      abort();
   }

   ef_ptr->internals_ptr->piecemeal_ok[0] = *xax;
   ef_ptr->internals_ptr->piecemeal_ok[1] = *yax;
   ef_ptr->internals_ptr->piecemeal_ok[2] = *zax;
   ef_ptr->internals_ptr->piecemeal_ok[3] = *tax;
   ef_ptr->internals_ptr->piecemeal_ok[4] = *eax;
   ef_ptr->internals_ptr->piecemeal_ok[5] = *fax;
}

