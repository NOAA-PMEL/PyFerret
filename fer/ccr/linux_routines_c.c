#include <stdio.h>
/* Routine needed only for linux.  nag F90 didn't have a perror routine, so 
   the c version of it is called.

   kob 3/97

  to compile:
        gcc -c [-g] linux_routines_c.c

	*/
         
void linux_perror_(char *string)
{
  perror(string);
}
