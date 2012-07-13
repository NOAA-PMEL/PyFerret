
#ifdef MAC_SSIZE
typedef long ssize_t;
#endif
#include <stdlib.h>
#include <string.h>
#include "EF_Util.h"

void FORTRAN(ef_set_freq_axis_sub)(int *id_ptr, int *axis_ptr, int *npts,
                                   DFTYPE *box, char *text, int *modulo_ptr)
{
  ExternalFunction *ef_ptr;
  double lo, hi, del, nfreq, yquist;

  ef_ptr = ef_ptr_from_id_ptr(id_ptr);
  if ( ef_ptr == NULL ) {
     abort();
  }

  nfreq = *npts/2;	
  yquist = 0.5* (1./ *box);		/* Nyquist frequency */
  lo = yquist/ nfreq;
  hi = yquist;
  del = lo;

  strcpy(ef_ptr->internals_ptr->axis[*axis_ptr-1].unit, text);
  ef_ptr->internals_ptr->axis[*axis_ptr-1].ww_lo = lo;
  ef_ptr->internals_ptr->axis[*axis_ptr-1].ww_hi = hi;
  ef_ptr->internals_ptr->axis[*axis_ptr-1].ww_del = del;
  ef_ptr->internals_ptr->axis[*axis_ptr-1].modulo = *modulo_ptr;
}  

