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


/* local static variable to contain the state */
static int its_batch=0;

/* set_batch_graphics */
#ifdef NO_ENTRY_NAME_UNDERSCORES
void set_batch_graphics( )
#else
void set_batch_graphics_( )
#endif
{
  its_batch = -1;
  return;
}

/* its_batch_graphics */
#ifdef NO_ENTRY_NAME_UNDERSCORES
int its_batch_graphics( )
#else
int its_batch_graphics_( )
#endif
{
   return (its_batch);
}


      


