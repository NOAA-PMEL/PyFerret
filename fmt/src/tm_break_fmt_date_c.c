/* *sh* 2/13/95

 *sh* 1/97 - modified to accept 4-field date used by LLNL: "1992-10-8 0"

break a date of the form yyyy-mm-dd  hh:mm:ss into its component pieces

all fields are integer except seconds which is float

hh:mm:ss are optional (defaulting to 00:00:00) or seconds, alone may be omitted

  compile with
       cc -c -g tm_break_fmt_date_c.c
  on non-ANSI compilers also use:
        -D_NO_PROTO

 *kob* 5/22/95 - need to add an ifdef check for NO_ENTRY_NAME_UNDERSCORES for
                 those machines (like hp) that don't need the underscore 
		 appended at the end of a routine call....

*/

#ifdef _NO_PROTO
#  ifdef NO_ENTRY_NAME_UNDERSCORES
int tm_break_fmt_date_c(date,
#  else
int tm_break_fmt_date_c_(date,
#  endif
			year,
			month,
			day,
			hour,
			minute,
			second)
char *date;
int *year, *month, *day, *hour, *minute;
float *second;

#else
#  ifdef NO_ENTRY_NAME_UNDERSCORES
int tm_break_fmt_date_c(char *date,
#  else
int tm_break_fmt_date_c_(char *date,
#  endif
			int *year,
			int *month,
			int *day,
			int *hour,
			int *minute,
			float *second)

#endif
{

  int n;

/* perform the conversion (e.g.) 1992-10-8 15:15:42.5  */
  n = sscanf(date,"%d-%d-%d %d:%d:%f",year,month,day,hour,minute,second);

  if ( n == 3 ) {
    *hour = 0;
    *minute = 0;
    *second = 0.0;
  } else if ( n == 4 ) {
    *minute = 0;
    *second = 0.0;
  } else if ( n == 5 ) {
    *second = 0.0;
  } else if ( n != 6 ) {
    return(1);
  }

  return(0);
}
