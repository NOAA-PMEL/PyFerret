/* Make sure Python.h is included first */
#include <Python.h>
#include <stdlib.h>
#include "ferret.h"
#include "EF_Util.h"

void FORTRAN(ef_set_axis_influence_6d)(int *id_ptr, int *arg,
                                       int *xax, int *yax, int *zax,
                                       int *tax, int *eax, int *fax)
{
   ExternalFunction *ef_ptr;

   ef_ptr = ef_ptr_from_id_ptr(id_ptr);
   if ( ef_ptr == NULL ) {
      abort();
   }

   if ( *xax != YES && *xax != NO ) {
      FORTRAN(ef_err_bail_out)(id_ptr, "Unknown X axis value passed to ef_set_axis_influence");
      /* The C function ef_err_bail_out should not return */
      abort();
   }
   if ( *yax != YES && *yax != NO ) {
      FORTRAN(ef_err_bail_out)(id_ptr, "Unknown Y axis value passed to ef_set_axis_influence");
      /* The C function ef_err_bail_out should not return */
      abort();
   }
   if ( *zax != YES && *zax != NO ) {
      FORTRAN(ef_err_bail_out)(id_ptr, "Unknown Z axis value passed to ef_set_axis_influence");
      /* The C function ef_err_bail_out should not return */
      abort();
   }
   if ( *tax != YES && *tax != NO ) {
      FORTRAN(ef_err_bail_out)(id_ptr, "Unknown T axis value passed to ef_set_axis_influence");
      /* The C function ef_err_bail_out should not return */
      abort();
   }
   if ( *eax != YES && *eax != NO ) {
      FORTRAN(ef_err_bail_out)(id_ptr, "Unknown E axis value passed to ef_set_axis_influence");
      /* The C function ef_err_bail_out should not return */
      abort();
   }
   if ( *fax != YES && *fax != NO ) {
      FORTRAN(ef_err_bail_out)(id_ptr, "Unknown F axis value passed to ef_set_axis_influence");
      /* The C function ef_err_bail_out should not return */
      abort();
   }

   ef_ptr->internals_ptr->axis_implied_from[*arg-1][0] = *xax;
   ef_ptr->internals_ptr->axis_implied_from[*arg-1][1] = *yax;
   ef_ptr->internals_ptr->axis_implied_from[*arg-1][2] = *zax;
   ef_ptr->internals_ptr->axis_implied_from[*arg-1][3] = *tax;
   ef_ptr->internals_ptr->axis_implied_from[*arg-1][4] = *eax;
   ef_ptr->internals_ptr->axis_implied_from[*arg-1][5] = *fax;
}

