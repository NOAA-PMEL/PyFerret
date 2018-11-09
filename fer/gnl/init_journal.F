

	SUBROUTINE INIT_JOURNAL( status )

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
* open the FERRET journal file

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V300:  5/93 - extracted from initialize.F
* V312:  5/94 - insert "PROCLAIM" output from here instead of from INITIALIZE
* V320: 12/29/94 - use WARN to avoid direct tty output (for GUI)
* V66   *kms* 5/10 LIB_GET_LUN is a subroutine, not a function
*       *acm* 3/12 cleanup ifdefs and unnecessary include files

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include 'tmap_errors.parm'  ! error codes
	include	'ferret.parm'
	include	'errmsg.parm'
	include 'xprog_state.cmn'

* calling argument declarations:
	INTEGER status

* local variable declarations:
	INTEGER TM_LENSTR1, jnl_len

* get logical unit to open the journal file
	CALL LIB_GET_LUN( jrnl_lun )

*  rename last journal file by this name to old version
        CALL TM_PARK_LAST_VERSION( journal_file, status )
        IF ( status .NE. merr_ok ) GOTO 9000
	OPEN (	FILE=journal_file,
     .		UNIT=jrnl_lun,
     .		STATUS='NEW',
     .		ERR = 9000 )

* put introductory remarks into journal file
	CALL PROCLAIM ( jrnl_lun, '! ', .FALSE. )

* successful completion
	status = ferr_ok   ! status not really used
	RETURN

* errors
* note: error not reported to ERRMSG since this is called from INITIALIZE
 9000	jrnl_lun = unspecified_int4	! flag that there is no journal file
	jnl_len = TM_LENSTR1(journal_file)
	CALL WARN('Unable to create journal file '
     .			//journal_file(:jnl_len) )
	status = ferr_invalid_command  

	END

