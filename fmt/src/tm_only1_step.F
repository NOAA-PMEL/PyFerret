	SUBROUTINE TM_ONLY1_STEP ( ds_num, f_num, status )

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
	include 'tmap_errors.parm'		! error code parameter defs
#include "tmap_dset.parm"
	include 'tmap_dims.parm'
	include 'xstep_files.cmn_text'
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
