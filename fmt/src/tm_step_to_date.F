	SUBROUTINE TM_STEP_TO_DATE ( dset_num, tstep, prec,
     .				     date, length, status )
* This function will return a string giving the date for a particular timestep
* for a particular data set. This is done by figuring the offset from
* 01-JAN-0000 00:00:00 in seconds, and then calculating the date.
*
* written 06/15/87 by Mark Verschell for PMEL/TMAP
*
* revision 0.00	- 06/15/87 - initial incarnation
* revision 0.20	- 06/26/87 - changed to subroutine and added length, status
* revision 0.21 - 07/15/88 - handles errors from TM_BREAK_DATE
*
* Arguments:
*	dset_num - data set number
*	tstep    - time step
*	prec	 - precision (1 to 6 for seconds to years)
*	date	 - returned string containing date in ASCII form
*	length	 - length of date
*	status	 - status at return time
*
	CHARACTER*(*)	date
	INTEGER		dset_num, prec, length, status
	REAL*8		tstep
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
	CHARACTER*20	TM_SECS_TO_DATE
	INTEGER		t0year, t0mon, t0day, t0hour, t0min, t0sec, istat
	REAL*8		start_secs, offset_secs, this_secs, prec_fact(6)
	REAL*8		TM_SECS_FROM_BC

	DATA	prec_fact	/  1.,  60.,  3600.,  86400.,
     .				   86400., 86400./

* First find the offset from 01-JAN-0000 00:00:00 from the descriptor file
	CALL TM_BREAK_DATE (ds_t0time(dset_num), t0year, t0mon,
     .			    t0day, t0hour, t0min, t0sec, istat)
	IF (istat .NE. merr_ok) GOTO 9000
	start_secs = TM_SECS_FROM_BC (	t0year, t0mon, t0day,
     .					t0hour, t0min, t0sec)

* Find the number of seconds since start of data set for this time step
	offset_secs = tstep*ds_time_unit(dset_num)

* Total seconds to this time step since 01-JAN-0000 00:00:00
	this_secs = start_secs + offset_secs

* Adjust precision to desired amount by finding nearest multiple of precision
* base (e.g. if precision = day; find closest integer divisible by 86400, which
* is number of seconds in a day). If prec > 4 then find closest day and then
* modify string
	this_secs = DNINT(this_secs/prec_fact(prec)) * prec_fact(prec)

* Convert to date
	date = TM_SECS_TO_DATE(this_secs)

* Now if prec > 4 modify string
	IF (prec .EQ. 5) date(1:2) = '00'
	IF (prec .EQ. 6) date(1:6) = '00-   '

* Now truncate string to desired length
	IF (LEN(date) .GT. length) date(length+1:LEN(date)) = ' '

* Finished
	status = merr_ok
	GOTO 9990

* Errors
 9000	CALL TM_ERRMSG ( istat, status, 'TM_STEP_TO_DATE',
     .	                 dset_num, no_stepfile,
     .			 'DD-MMM-YYYY HH:MM:SS',ds_t0time(dset_num),
     .	                 *9990 )

 9990	RETURN
	END
