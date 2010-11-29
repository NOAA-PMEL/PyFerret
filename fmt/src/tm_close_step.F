	SUBROUTINE TM_CLOSE_STEP ( f_num, status )

*
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*
*  Access and use of this software shall impose the following
*  obligations and understandings on the user. The user is granted the
*  right, without any fee or cost, to use, copy, modify, alter, enhance
*  and distribute this software, and any derivative works thereof, and
*  its supporting documentation for any purpose whatsoever, provided
*  that this entire notice appears in all copies of the software,
*  derivative works and supporting documentation.  Further, the user
*  agrees to credit NOAA/PMEL/TMAP in any publications that result from
*  the use of this software or in any product that includes this
*  software. The names TMAP, NOAA and/or PMEL, however, may not be used
*  in any advertising or publicity to endorse or promote any products
*  or commercial entity unless specific written permission is obtained
*  from NOAA/PMEL/TMAP. The user also understands that NOAA/PMEL/TMAP
*  is not obligated to provide the user with any support, consulting,
*  training or assistance of any kind with regard to the use, operation
*  and performance of this software nor to provide the user with any
*  updates, revisions, new versions or "bug fixes".
*
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP "AS IS" AND ANY EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE. 
*
*
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
	include 'tmap_errors.parm'
#include "tmap_dset.parm"
	include 'tmap_dims.parm'
	include 'xstep_files.cmn_text'
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
