      SUBROUTINE TM_YMD_TIME_CONVRT( yymmdd,
     .                               mon, day, yr, hr, min, sec)

* decode a double precision yyyymmddhhmmss into its component parts

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 2/95  -- specifically for accessing CDC files

* argument definitions:
*	yymmdd	 - double precision input
*       mon, day, yr, hr, min, sec  - pieces of time !!

* argument declarations
      INTEGER mon, day, yr, hr, min, sec
      REAL*8  yymmdd

* internal variable declarations
      INTEGER itmp, ymd, hms

* split off the hhmmss fields
      ymd = yymmdd/1000000		!yyyymmdd
      hms = yymmdd - ymd*1000000.D0	!hhmmss

      itmp = hms/100
      sec  = hms - 100*itmp
      hr   = itmp/100
      min  = itmp - 100*hr

      itmp = ymd/100
      day  = ymd - 100*itmp
      yr   = itmp/100
      mon  = itmp - 100*yr

* CDC uses a climatological month like "100000000" for January
* here we shift the day=0 to day=15, arbitrarily
      IF (day .EQ. 0) day = 15

* to avoid the damn nit-picky stuff about negative times we will encode
* the climatological dates as year 1
      IF ( yr .EQ. 0 ) yr = 1


      RETURN	
      END


