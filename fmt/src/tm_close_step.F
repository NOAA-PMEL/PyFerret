	SUBROUTINE TM_CLOSE_STEP ( f_num, status )

* close a file of time step data
* if called with a negative file number this subroutine returns without action
* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.00 - 11/20/85
* revision 0.10 - 02/04/85 - added freeing of logical unit after close
* revision 0.20 - 02/15/89 - renamed TM_CLOSE_STEP, renamed sf_num to f_num
*
* arguments:
*	f_num	- number (in /XSTEP_FILES/) of file to close
*	status	- error status on completion

#ifdef unix
	include 'tmap_format/tmap_errors.parm'
#include "tmap_format/tmap_dset.parm"
	include 'tmap_format/tmap_dims.parm'
	include 'tmap_format/xstep_files.cmn_text'
	external xstep_files_data
#else
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:XSTEP_FILES.CMN'
#endif

	INTEGER		f_num,status,lunit

* initialize
	status = merr_ok			! innocent until proven guilty

* interpret a negative or zero f_num as a no-op
	IF (f_num .LE. 0) RETURN

* normal situation - close the file as requested and free the logical unit
* number
	lunit = sf_lunit(f_num)
	CLOSE (UNIT = lunit, ERR = 5000)
	CALL LIB_FREE_LUN (lunit)
	sf_lunit(f_num) = file_not_open	! signals "closed"
 2000	RETURN

* report ERR= error
 5000	CALL TM_ERRMSG ( merr_erreq, status, 'TM_CLOSE_STEP',
     .			 sf_setnum(f_num), f_num,
     .			 no_errstring, no_errstring , *2000)
	END
