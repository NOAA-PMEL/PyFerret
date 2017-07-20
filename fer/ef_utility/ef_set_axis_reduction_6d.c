/* Make sure Python.h is included first */
#include <Python.h>
#include <stdlib.h>
#include "ferret.h"
#include "EF_Util.h"

void FORTRAN(ef_set_axis_reduction_6d)(int *id_ptr, int *xax, int *yax, int *zax,
                                                    int *tax, int *eax, int *fax)
{
   ExternalFunction *ef_ptr;

   ef_ptr = ef_ptr_from_id_ptr(id_ptr);
   if ( ef_ptr == NULL ) {
      abort(); 
   }

   if ( *xax != RETAINED && *xax != REDUCED ) {
      FORTRAN(ef_err_bail_out)(id_ptr, "Unknown X axis value passed to ef_set_axis_reduction");
      /* The C function ef_err_bail_out should not return */
      abort();
   }
   if ( *yax != RETAINED && *yax != REDUCED ) {
      FORTRAN(ef_err_bail_out)(id_ptr, "Unknown Y axis value passed to ef_set_axis_reduction");
      /* The C function ef_err_bail_out should not return */
      abort();
   }
   if ( *zax != RETAINED && *zax != REDUCED ) {
      FORTRAN(ef_err_bail_out)(id_ptr, "Unknown Z axis value passed to ef_set_axis_reduction");
      /* The C function ef_err_bail_out should not return */
      abort();
   }
   if ( *tax != RETAINED && *tax != REDUCED ) {
      FORTRAN(ef_err_bail_out)(id_ptr, "Unknown T axis value passed to ef_set_axis_reduction");
      /* The C function ef_err_bail_out should not return */
      abort();
   }
   if ( *eax != RETAINED && *eax != REDUCED ) {
      FORTRAN(ef_err_bail_out)(id_ptr, "Unknown E axis value passed to ef_set_axis_reduction");
      /* The C function ef_err_bail_out should not return */
      abort();
   }
   if ( *fax != RETAINED && *fax != REDUCED ) {
      FORTRAN(ef_err_bail_out)(id_ptr, "Unknown F axis value passed to ef_set_axis_reduction");
      /* The C function ef_err_bail_out should not return */
      abort();
   }

   ef_ptr->internals_ptr->axis_reduction[0] = *xax;
   ef_ptr->internals_ptr->axis_reduction[1] = *yax;
   ef_ptr->internals_ptr->axis_reduction[2] = *zax;
   ef_ptr->internals_ptr->axis_reduction[3] = *tax;
   ef_ptr->internals_ptr->axis_reduction[4] = *eax;
   ef_ptr->internals_ptr->axis_reduction[5] = *fax;
}

