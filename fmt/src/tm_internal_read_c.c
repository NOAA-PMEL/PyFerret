/* *kob* 10/95

extract floating pt. value from a fortran character string

modified from tm_break_fmt_date_c.c
*/

#ifdef _NO_PROTO
#  ifdef NO_ENTRY_NAME_UNDERSCORES
int tm_internal_read_c(cstring,
#  else
int tm_internal_read_c_(cstring,
#  endif
			value,
			status)
char *cstring;
float *value;
int *status;

#else
#  ifdef NO_ENTRY_NAME_UNDERSCORES
int tm_internal_read_c(char *cstring,
#  else
int tm_internal_read_c_(char *cstring,
#  endif
			float *value,
			int   *status)

#endif
{

  int n;

/* perform the conversion (e.g.) 1992-10-8 15:15:42.5  */
  n = sscanf(cstring,"%f",value);

  return(0);
}
