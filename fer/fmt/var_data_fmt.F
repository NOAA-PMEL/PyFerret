	SUBROUTINE VAR_DATA_FMT( nleft, nright, rqst_digits, min_width,
     .				 width, good_fmt, bad_fmt )

* prepare a format suitable to write out gridded data in a minimum width that
* provides the requested precision.  May use E or F format, depending on size
* "min_width" prevents the field from being less than xxx columns

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* V200:   7/9/90 - extracted from LIST_CMND_DATA

#ifdef unix
	include 'ferret_cmn/ferret.parm'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
#endif

* calling argument declarations:
	INTEGER nleft, nright, rqst_digits, min_width,
     .		width
	CHARACTER*(*) good_fmt, bad_fmt

* local variable declarations:
	INTEGER	F_size, E_size

* use F or E format for good data - whichever is smaller
	F_size = nleft + nright + 3	! _.     and extra spc --> 3
	E_size = rqst_digits + 7		! -.E-xx and extra spc --> 7
	IF ( F_size .LT. E_size ) THEN
	   width = MAX( F_size, min_width )
	   WRITE ( good_fmt, 3000 ) 'F', width, nright
	ELSE
	   width  = MAX( E_size, min_width )
	   WRITE ( good_fmt, 3000 ) '1PE', width, rqst_digits-1
	ENDIF

* string format for bad data
	WRITE ( bad_fmt, 3002 ) width
* . . . to create ...FORMAT (Fxx.yy) and FORMAT (Axx)

 3000	FORMAT ( '(',A,I2,'.',I2,')' )	
 3002	FORMAT ( '(A',I2,')' )	

	RETURN
	END
