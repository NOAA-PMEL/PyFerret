/* this routine extracted unmodified from the EPS library file fil_time.c
   Replaces the earlier port of this routine under the name tm_ep_time_convrt_
   which was FORTRAN-accessible.  This routine user, instead a FORTRAN jacket
   *sh* 1/94
*/

#define JULGREG   2299161

void ep_time_to_mdyhms(time, mon, day, yr, hour, min, sec)
     long *time;
     int *mon, *day, *yr, *hour, *min;
     float *sec;
{
/*
 * convert eps time format to mdy hms
 */
  long ja, jalpha, jb, jc, jd, je;

  while(time[1] >= 86400000) { /* increament days if ms larger then one day */
    time[0]++;
    time[1] -= 86400000;
  }

  if(time[0] >= JULGREG) {
    jalpha=((double) (time[0]-1867216)-0.25)/36524.25;
    ja=time[0]+1+jalpha-(long)(0.25*jalpha);
  } else
    ja=time[0];

  jb=ja+1524;
  jc=6680.0+((double)(jb-2439870)-122.1)/365.25;
  jd=365*jc+(0.25*jc);
  je=(jb-jd)/30.6001;
  *day=jb-jd-(int)(30.6001*je);
  *mon=je-1;
  if(*mon > 12) *mon -= 12;
  *yr=jc-4715;
  if(*mon > 2) --(*yr);

  if(*yr <=0) --(*yr);

  ja = time[1]/1000;
  *hour = ja/3600;
  *min = (ja - (*hour)*3600)/60;
  *sec = (float)(time[1] - ((*hour)*3600 + (*min)*60)*1000)/1000.0;
}

/* convert from eptime to mdyhms */
/* FORTRAN-callable jacket extracted from file jackets.c
   Note:  The entry point name has been changed from mdyhmstoeptime_
   to tm_ep_time_convrt_ and the calling argument time[2] has been
   changed to the *int args epjday and epmsec.  Also, the entry has 
   re-written in prototyping format

   calling arguments:
      epjday (input) - integer
      epmsec (input) - integer
      mon, day, yr, hr, min (output) - integer
      sec (output) - REAL*4

   *sh* 1/94
*/
#ifdef NO_ENTRY_NAME_UNDERSCORES
void tm_ep_time_convrt(epjday,
#else
void tm_ep_time_convrt_(epjday,
#endif
			epmsec,
			mon,
			day,
			yr,
			hour,
			min,
			sec)

/* prototypes not allowed on TMAP SUN cc compiler.  Need ANSI ?? */
int *epjday, *epmsec, *mon, *day, *yr, *hour, *min;
float *sec;

{
/*  this block added by *sh* 1/94 */
  long time[2];
  time[0] = (long)*epjday;
  time[1] = (long)*epmsec;

  (void) ep_time_to_mdyhms(time, mon, day, yr, hour, min, sec);
}



