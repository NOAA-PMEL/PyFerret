	SUBROUTINE TM_BREAK_FMT_DATE (form, date, year, month, day,
     .					 hour, minute, second, status)
*
* Will take date strings of various formats and break them into their
* various components.

* *SH* 2/13/95

* ARGUMENT DEFINITIONS
*  form  - code for the date string format
*  date  - date to process
*  day, month, year, hour, minute, second - obvious
*  status - returned status
	CHARACTER*(*)	date
	INTEGER	form, day, month, year, hour, minute, second, status
*
* INCLUDE FILES
#ifdef unix
	include 'tmap_format/tmap_errors.parm'
#include "tmap_format/tmap_dset.parm"
	include 'tmap_format/gt_lib.parm'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
#endif

* internal variable declarations:
	INTEGER slen
	PARAMETER ( slen = 32)
	INTEGER TM_LENSTR1, TM_BREAK_FMT_DATE_C, dlen
	REAL fsecond
#ifdef sun
	BYTE       dhol(slen)     !c-type Hollerith string buffers
#else
	INTEGER*1  dhol(slen)     !c-type Hollerith string buffers
#endif

	dlen = TM_LENSTR1( date)

	IF ( form .EQ. pdate_vax ) THEN
	    CALL TM_BREAK_DATE( date, year, month, day,
     .			hour, minute, second, status)
	    RETURN

	ELSEIF ( form .EQ. pdate_numslashnum ) THEN
* ... note:  this is a quick and dirty kludge parser.  Use the udunits package
*	     for a more robust approach
* ... the format looks like (e.g.) 1992-10-8 15:15:42.5 

	   CALL TM_FTOC_STRNG( date(:dlen), dhol, slen)
	   status = TM_BREAK_FMT_DATE_C(dhol, year, month, day,
     .				        hour, minute, fsecond)
	   IF ( status .NE. 0 ) GOTO 5100
	   second = NINT(fsecond)	! information loss ... 
	ELSE
	   STOP 'Unknown date format code'
	ENDIF

* successful translation
	status = merr_ok
	RETURN

* error exits
 5000	RETURN
 5100	CALL TM_ERRMSG (merr_badtimedef, status, 'TM_BREAL_FMT_DATE',
     .			unspecified_int4, no_stepfile,
     .			'syntax error in "'//date(:dlen),
     .			no_errstring, *5000)

	END
