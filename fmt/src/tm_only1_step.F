	SUBROUTINE TM_ONLY1_STEP ( ds_num, f_num, status )

* close all step files in this data set except the one specified
* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.00 - 11/22/85
* revision 0.10 - 02/15/89 - renamed to TM_ONLY1_STEP and changed GT_CLOSE_STEP
*			     to TM_CLOSE_STEP, renamed sf_num to f_num
* arguments:
*	ds_num	- data set number (see COMMON /XSTEP_FILES/)
*	f_num	- file number within COMMON/XSTEP_FILES to leave open

#ifdef unix
	include 'tmap_format/tmap_errors.parm'		! error code parameter defs
#include "tmap_format/tmap_dset.parm"
	include 'tmap_format/tmap_dims.parm'
	include 'tmap_format/xstep_files.cmn_text'
	external xstep_files_data
#else
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'		! error code PARAMETER defs
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:XSTEP_FILES.CMN'
#endif

	INTEGER		ds_num,f_num,status,ierr,i

* initialize error status
	status = merr_ok	! innocent until proven guilty

* loop through all step files
	DO 100 i = 1,maxstepfiles

	  IF ((sf_setnum(i) .EQ. ds_num) .AND.		! correct data set ?
     .	      (sf_lunit (i) .NE. file_not_open) .AND.	! open ?
     .	      (i  .NE. f_num))			 THEN	! leave requested 1 open

	    CALL TM_CLOSE_STEP (i,ierr)
	    IF (ierr .NE. merr_ok) status = ierr
	  ENDIF

 100	CONTINUE

	RETURN
	END
