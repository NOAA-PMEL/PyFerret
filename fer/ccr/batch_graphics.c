/* batch_graphics.c

* contains entries
*     void set_batch_graphics()    ! sets program state
* and
*     int its_batch_graphics       ! queries program state

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 3/5/97

* compile with
*    cc -g -c batch_graphics.c
*  or
*    cc    -c batch_graphics.c

*/


#include <assert.h>
#include <stdio.h>

#ifdef NO_ENTRY_NAME_UNDERSCORES
#define FORTRAN(a) a
#else
#define FORTRAN(a) a##_
#endif

/* local static variable to contain the state */
static int its_batch=0;

/* set_batch_graphics */
void FORTRAN(set_batch_graphics)(char *outfile)
{
  int length;
  assert(outfile);
  length = strlen(outfile);
  FORTRAN(save_metafile_name)(outfile, &length);
  its_batch = -1;
  return;
}

/* its_batch_graphics */
int FORTRAN(its_batch_graphics)()
{
   return (its_batch);
}


      


