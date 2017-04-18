/* Make sure Python.h is included first */
#include <Python.h>
#include "EF_Util.h"

/*
 *sh 1/17 -- with dynamic memory management in Ferret this routine now
    does nothing.  It could be removed, and the FORTRAN routine 
    ef_get_one_val_sub could be renamed as ef_get_one_val
*/
void FORTRAN(ef_get_one_val)(int *id_ptr, int *arg_ptr, DFTYPE *val_ptr)
{
  FORTRAN(ef_get_one_val_sub)(id_ptr, arg_ptr, val_ptr);
}

