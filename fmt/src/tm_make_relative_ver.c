#include <stdio.h>
#include <string.h>

/* 

  this routine takes the relative verson number passed to it (eg, .~-3~) and
    calls high_ver_name (passing filename and path) to get the proper
    version number for the file. (eq, ~12~)
    It then returns this value.

 version 0.0 -kob- 10/17/91

*/

/* had to add ifdef check for trailing underscore in routine name
   for aix port *kob* 10/94 */

#ifdef NO_ENTRY_NAME_UNDERSCORES
char *tm_make_relative_ver(curr_ver, fname,path,real_ver)
#else
char *tm_make_relative_ver_(curr_ver, fname,path,real_ver)
#endif
char *curr_ver,*fname, *path;
int *real_ver;

/*

 calling arguments :    
            curr_ver --> contains the relative version num. (eg. .~-3~)
	    real_ver --> will contain and pass back proper version num. (eq. ~12~)
	    fname -----> filename; needed for routine high_ver_name
	    path ------> path to file, also needed for routine high_ver_name

*/

{
  int i,j,int_ver, high_ver, ver_len;
  char *temp_ver, *malloc();

/* allocate temporary memory */
  temp_ver = malloc(20);

/* get just the numeric part of the string, ignoring all else */
  for (i=0,j=0; i<=strlen(curr_ver); i++)
    {
      if (*(curr_ver+i) != '.' && *(curr_ver+i) != '-' && *(curr_ver+i) != '~')
	{
	  *(temp_ver+j) = *(curr_ver+i);
	  ++j;
	}
    }

/* convert the string to an integer */ 
  sscanf (temp_ver, "%d", real_ver);

/* get the new version number by subtracting the relative version number -1
     from the highest version number          */
  *real_ver -= 1;
  high_ver = high_ver_name (fname,path);
  *real_ver = high_ver - *real_ver;

  

/* convert that from integer to character string 
  sprintf (temp_ver, "%d", int_ver);

 surround the new version number with tilda's 
  strcat (real_ver, "~");
  strcat (real_ver, temp_ver);
  strcat (real_ver, "~");
  
 append a null to the string 
  ver_len = strlen(real_ver);
  *(real_ver+ver_len) = '\0';  */

/* return proper version extension */
/*  return real_ver; 
 */

}












