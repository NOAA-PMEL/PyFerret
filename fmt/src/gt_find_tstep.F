	SUBROUTINE GT_FIND_TSTEP ( dset_num,time_step,varpos,
     .				   sf_num,varstrt,status )
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
* Locates and opens (if not already open) the correct data file containing
* the specified timestep and variable. Calls GT_FIND_REC to locate the
* starting record of the specified variable.
* 
* programmer - mark verschell
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.00 - 11/15/85 - (steve hankin)
* revision 1.00 - 08/27/86 - revised to work with new .DES, .SET files(mav)
* revision 2.00 - 11/18/86 - revised to optimize searching by first locating
*			     correct step then checking for open or not (mav)
* revision 3.00 - 02/12/87 - added variable to calling arguments due to 
*			     addition of auxiliary (or seperate) data sets,
*			     also this program is now called by GET_GRID
*			     routines, rather than seperately (mav)
* revision 3.01 - 11/01/88 - removed variable location to TM_READ, this routine
*			     passed varpos only now
*
* arguments:
*	dset_num	- number (in COMMON/DSET_INFO/) of dataset
*	time_step	- desired time step value
*	varpos		- location in ds_ arrays of the correct variable
*	sf_num	 	- (output) the desired file number (in /XSTEP_FILES/)
*	varstrt		- (output) record number of the first variable record
*	status		- returned error code

	INTEGER		dset_num,sf_num,varstrt,varpos,status
	REAL*8		time_step

#ifdef unix
#include "tmap_format/tmap_dset.parm"
	include 'tmap_format/tmap_dims.parm'
	include 'tmap_format/tmap_errors.parm'
	include 'tmap_format/xstep_files.cmn_text'
	external xstep_files_data
	include 'tmap_format/xdset_info.cmn_text'
	external xdset_info_data
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
	INCLUDE 'TMAP_FORMAT:XSTEP_FILES.CMN'
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'
#endif

	LOGICAL	 	missing
	INTEGER		missing_num,sfcnt,ivar
	CHARACTER*13	TM_STRING

* initialize
200	missing = .FALSE.

* find correct file
	DO 300 sfcnt = 1,maxstepfiles
	  IF (sf_setnum(sfcnt) .NE. dset_num) GOTO 300	! right dataset?
	  IF ((sf_1step(sfcnt).GT. time_step) .OR.	! contains timestep
     .	      (sf_lastep(sfcnt) .LT. time_step)) GOTO 300
	  IF (sf_aux_set_num(sfcnt) .NE.		! stepfile contains var?
     .	       ds_aux_set_num(varpos)) GOTO 300
	  GOTO 400
300	CONTINUE
	GOTO 500

* found the correct one!
* check to see if file open already
400	IF (sf_lunit(sfcnt)  .EQ. file_not_open) THEN
* limits correct, file not open, check this data file for irregularities
!	  IF (.NOT. sf_regvars(sfcnt) .OR. .NOT. sf_regtimes(sfcnt))
!     .	     CALL TM_ERRMSG (merr_notsupport, status, 'GT_FIND_TSTEP',
!     .			     dset_num, sfcnt,
!     .			     'Irregular variable or timesteps',
!     .			     no_errstring, *2000)
* no irregularities so far so let's try to open it
	  CALL GT_OPEN_STEP (sfcnt,status)
* Is it mounted?
	  IF (status .NE. merr_OK) GOTO 2000
	ENDIF

* stepfile mounted and normal so locate timestep
	CALL GT_FIND_REC (sfcnt,time_step,varpos,varstrt,status)
	IF ((status .NE. merr_ok) .OR. (varstrt .EQ. 0)) RETURN
* got it
	sf_num = sfcnt
	status = merr_ok
	RETURN

* outta luck - no time step after all that
500	CALL TM_ERRMSG (merr_nostep, status, 'GT_FIND_TSTEP',
     .			dset_num, no_stepfile,
     .			'STEP='//TM_STRING(time_step),
     .			no_errstring, *2000)
2000	RETURN
	END
