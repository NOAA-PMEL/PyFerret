	REAL*8 FUNCTION TM_DATE_TO_STEP ( dset_num, date, status )
* This function will return a REAL*8 number giving the time step for a given
* date for a particular data set. This is done by figuring the offset from
* 01-JAN-0000 00:00:00 in seconds, and then calculating the timestep.
*
* written 06/17/87 by Mark Verschell for PMEL/TMAP
*
* revision 0.00	- 06/17/87 - initial incarnation
* revision 0.01 - 07/08/88 - Returns status code, handles errors from
*			     TM_BREAK_DATE
*
* Arguments:
*	dset_num - data set number
*	date	 - date of interest
*	status   - returning status
*
	CHARACTER*20	date
	INTEGER		dset_num, status
*
* Inclusions
*
#ifdef unix
	include 'tmap_format/tmap_errors.parm'
	include 'tmap_format/tmap_dims.parm'
#include "tmap_format/tmap_dset.parm"
	include 'tmap_format/xdset_info.cmn_text'
	external xdset_info_data
#else
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'
#endif

*
* Local definitions
*
	INTEGER		year, month, day, hour, minute, second, istat,
     .			t0year, t0mon, t0day, t0hour, t0min, t0sec
	REAL*8		date_secs, start_secs, offset_secs
	REAL*8		TM_SECS_FROM_BC

* Break up the date string to it's various components
	CALL TM_BREAK_DATE ( date, year, month, day, hour, minute, second,
     .	                     istat )
	IF ( istat .NE. merr_ok ) GOTO 9000

* Find the offset from 01-JAN-0000 00:00:00 for this date
	date_secs = TM_SECS_FROM_BC (year, month, day,
     .				     hour, minute, second)

* First find the offset from 01-JAN-0000 00:00:00 from the descriptor file
	CALL TM_BREAK_DATE (ds_t0time(dset_num), t0year, t0mon,
     .			    t0day, t0hour, t0min, t0sec, status)
	IF ( istat .NE. merr_ok ) GOTO 9100
	start_secs = TM_SECS_FROM_BC (t0year, t0mon, t0day,
     .				      t0hour, t0min, t0sec)

* Find the number of seconds since start of data set for this date
	offset_secs = date_secs - start_secs

* Divide by time step increment to get # of time step
	TM_DATE_TO_STEP = offset_secs/ds_time_unit(dset_num)

* Finished
	status = merr_ok
	GOTO 9990

* Errors
 9000	CALL TM_ERRMSG ( istat, status, 'TM_DATE_TO_STEP',
     .	                 dset_num, no_stepfile,
     .			 'DD-MMM-YYYY HH:MM:SS', date, *9990 )
 9100	CALL TM_ERRMSG ( istat, status, 'TM_DATE_TO_STEP',
     .	                 dset_num, no_stepfile,
     .			 'DD-MMM-YYYY HH:MM:SS',ds_t0time(dset_num),
     .	                 *9990 )

 9990	RETURN
	END
