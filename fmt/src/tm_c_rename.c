
/* tm_c_rename.c */
/* make a Unix system call to rename a file */

/*
* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* originally written for DECstation under Ultrix operating system
* V1.0 5/6/91
*/

/* *kob* had to add ifdef for sake of AIX  10/94 */

#ifdef NO_ENTRY_NAME_UNDERSCORES
void tm_c_rename( oldname, newname, status )
#else
void tm_c_rename_( oldname, newname, status )
#endif


   char *oldname, *newname;
   int *status;

/* Unix system call to rename file */

{
   *status = rename ( oldname, newname );

   return;
}
