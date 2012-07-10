#include <stdlib.h>
#include "EF_Util.h"

void FORTRAN(ef_set_axis_inheritance_6d)(int *id_ptr,
                                         int *xax, int *yax, int *zax,
                                         int *tax, int *eax, int *fax)
{
   ExternalFunction *ef_ptr;

   ef_ptr = ef_ptr_from_id_ptr(id_ptr);
   if ( ef_ptr == NULL ) {
      abort();
   }

   if ( *xax != CUSTOM && *xax != IMPLIED_BY_ARGS && *xax != NORMAL && *xax != ABSTRACT ) {
      FORTRAN(ef_err_bail_out)(id_ptr, "Unknown X axis value passed to ef_set_axis_inheritance");
      /* The C function ef_err_bail_out should not return */
      abort();
   }
   if ( *yax != CUSTOM && *yax != IMPLIED_BY_ARGS && *yax != NORMAL && *yax != ABSTRACT ) {
      FORTRAN(ef_err_bail_out)(id_ptr, "Unknown Y axis value passed to ef_set_axis_inheritance");
      /* The C function ef_err_bail_out should not return */
      abort();
   }
   if ( *zax != CUSTOM && *zax != IMPLIED_BY_ARGS && *zax != NORMAL && *zax != ABSTRACT ) {
      FORTRAN(ef_err_bail_out)(id_ptr, "Unknown Z axis value passed to ef_set_axis_inheritance");
      /* The C function ef_err_bail_out should not return */
      abort();
   }
   if ( *tax != CUSTOM && *tax != IMPLIED_BY_ARGS && *tax != NORMAL && *tax != ABSTRACT ) {
      FORTRAN(ef_err_bail_out)(id_ptr, "Unknown T axis value passed to ef_set_axis_inheritance");
      /* The C function ef_err_bail_out should not return */
      abort();
   }
   if ( *eax != CUSTOM && *eax != IMPLIED_BY_ARGS && *eax != NORMAL && *eax != ABSTRACT ) {
      FORTRAN(ef_err_bail_out)(id_ptr, "Unknown E axis value passed to ef_set_axis_inheritance");
      /* The C function ef_err_bail_out should not return */
      abort();
   }
   if ( *fax != CUSTOM && *fax != IMPLIED_BY_ARGS && *fax != NORMAL && *fax != ABSTRACT ) {
      FORTRAN(ef_err_bail_out)(id_ptr, "Unknown F axis value passed to ef_set_axis_inheritance");
      /* The C function ef_err_bail_out should not return */
      abort();
   }

   ef_ptr->internals_ptr->axis_will_be[0] = *xax;
   ef_ptr->internals_ptr->axis_will_be[1] = *yax;
   ef_ptr->internals_ptr->axis_will_be[2] = *zax;
   ef_ptr->internals_ptr->axis_will_be[3] = *tax;
   ef_ptr->internals_ptr->axis_will_be[4] = *eax;
   ef_ptr->internals_ptr->axis_will_be[5] = *fax;
}

