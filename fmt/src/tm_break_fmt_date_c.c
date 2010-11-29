/*
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*
*  Access and use of this software shall impose the following
*  obligations and understandings on the user. The user is granted the
*  right, without any fee or cost, to use, copy, modify, alter, enhance
*  and distribute this software, and any derivative works thereof, and
*  its supporting documentation for any purpose whatsoever, provided
*  that this entire notice appears in all copies of the software,
*  derivative works and supporting documentation.  Further, the user
*  agrees to credit NOAA/PMEL/TMAP in any publications that result from
*  the use of this software or in any product that includes this
*  software. The names TMAP, NOAA and/or PMEL, however, may not be used
*  in any advertising or publicity to endorse or promote any products
*  or commercial entity unless specific written permission is obtained
*  from NOAA/PMEL/TMAP. The user also understands that NOAA/PMEL/TMAP
*  is not obligated to provide the user with any support, consulting,
*  training or assistance of any kind with regard to the use, operation
*  and performance of this software nor to provide the user with any
*  updates, revisions, new versions or "bug fixes".
*
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP "AS IS" AND ANY EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE.  
*
*/



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
