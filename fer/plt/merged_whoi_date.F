	CHARACTER*14 FUNCTION MERGED_WHOI_DATE( cx_list, ncx )

* create a string of the form YYMMDDHHMMSSCC (internal ordering) based
* on the earlies time to be plotted from the given contexts

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V420: 11/27/95 - 4-digit YEARS!!

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xcontext.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
#endif

* calling argument declarations
	INTEGER	ncx, cx_list(ncx)

* internal variable declarations
	INTEGER 	yr100, yr, mon, day, hr, mn, sec,
     .			status, i, grid, cx
	CHARACTER*20	TM_SECS_TO_DATE, date
	REAL*8		TSTEP_TO_SECS, bcsecs

* internal parameter declarations
	REAL*8		yrofsecs
	PARAMETER     ( yrofsecs = 60.*60.*24.*365. )

* determne the earliest date to be plotted
	DO 100 i = 1, ncx
	   cx = cx_list(i)
	   grid = cx_grid(cx)
	   IF ( i .EQ. 1 ) THEN
	      bcsecs = TSTEP_TO_SECS( grid, cx_lo_ww(t_dim,cx) )
	   ELSE
	      bcsecs = MIN( bcsecs,
     .		       TSTEP_TO_SECS( grid, cx_lo_ww(t_dim,cx) ) )
	   ENDIF
 100	CONTINUE

* arbitrarily subtract a year -- needed cuz PLOT* rounds down the "tmin" value
	bcsecs = MAX( 0.0D0, bcsecs - yrofsecs )

* Convert to date
	date = TM_SECS_TO_DATE(bcsecs)

* break the date string into component integer values
	CALL TM_BREAK_DATE( date, yr, mon, day, hr, mn, sec, status )
	yr100 = yr/100		! 1980 --> 19
	yr = yr - (yr100*100)	! 1982 --> 82

* assemble the pieces in WHOI format
	WRITE (MERGED_WHOI_DATE,3000) yr,mon,day,hr,mn,sec,yr100
 3000	FORMAT (7I2.2)

	RETURN
	END
