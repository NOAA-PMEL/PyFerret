	SUBROUTINE TM_MAKE_FMT_DATE (form, date, year, month, day,
     .				     hour, minute, second, dlen)
*
* Will take the components of a date and convert it to a string of the
* requested date format

* *sh* 2/13/95

* ARGUMENT DEFINITIONS
*  form  - code for the date string format
*  date  - date to process
*  day, month, year, hour, minute, second - obvious
*  status - returned status
	CHARACTER*(*)	date
	INTEGER	form, day, month, year, hour, minute, second, dlen
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
	CHARACTER*20    tempstr
	 CHARACTER*3    cmon(12)

	DATA cmon /'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
     .             'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'/


	IF ( form .EQ. pdate_vax ) THEN
* model date: 01-DEC-1982 12:52:00
	   WRITE (date,1000) day, cmon(month), year, hour,
     .  			minute, second
1000       FORMAT(I2.2,'-',A3,'-',I4.4,' ',2(I2.2,':'),I2.2)
	   dlen = 20
	ELSEIF ( form .EQ. pdate_numslashnum ) THEN
* model date: 1982-12-01 12:52:00
	   WRITE (date,1100) year, month, day, hour,
     .  			minute, second
1100       FORMAT(I4.4,'-',I2.2,'-',I2.2,' ',2(I2.2,':'),I2.2)
	   dlen = 19
	ELSE
	   STOP 'Unknown date format code'
	ENDIF

	RETURN
	END
