	SUBROUTINE SPLIT_LIST( mode, out_lun, next_line, line_len )

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
* depending on the state of mode_gui either write this string to the
* out_lun unit number or send it to the GUI

* Note that line_len=0 means this routine will determine the line length

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V320:  11/9/94
*	 12/29/94 - added "mode" argument
* V400:	  6/20/95 - bug fix: LIST/FILE output sent to "default_list_lun"
*			should go to disk file, not to screen
* V6.6  *kms* 5/10 - monitor redirect_*_flags for stdout/stderr redirection.
*       *acm* 3/12 6D Ferret (common uses nferdims in tmap_dims.parm)
** 10/2022 *acm* Code cleanup: remove references to obsolete gui interface
*                previous first argument was "mode", unused except in the gui 

        IMPLICIT NONE
	include	'tmap_dims.parm'
	include 'ferret.parm'
	include 'xprog_state.cmn'

* calling arguments
	INTEGER	      mode, out_lun, line_len
	CHARACTER*(*) next_line

*internal variable declarations:
        INTEGER slen
	INTEGER	TM_LENSTR1, llen

* determine line length if not supplied
	IF ( line_len .LE. 0 ) THEN
	   llen = TM_LENSTR1(next_line)
	ELSE
	   llen = line_len
	ENDIF

* output to the desired device
	IF ( (out_lun .EQ. ttout_lun) .AND.
     .            (redirect_stdout_flags .NE. redirect_none) ) THEN
           IF ( (redirect_stdout_flags .EQ. redirect_journal_tee) .OR.
     .          (redirect_stdout_flags .EQ. redirect_journal) ) THEN
              IF (mode_journal .AND. jrnl_lun.NE.unspecified_int4) THEN
                 WRITE ( jrnl_lun, '(A,A)' ) '!', next_line(:llen)
              ENDIF
           ENDIF
           IF ( (redirect_stdout_flags .EQ. redirect_file_tee) .OR.
     .          (redirect_stdout_flags .EQ. redirect_file) ) THEN
              WRITE ( redirect_stdout_lun, '(A)' ) next_line(:llen)
           ENDIF
           IF ( (redirect_stdout_flags .EQ. redirect_journal_tee) .OR.
     .          (redirect_stdout_flags .EQ. redirect_file_tee) ) THEN
              WRITE ( out_lun, '(A)' ) next_line(:llen)
           ENDIF
        ELSE IF ( (out_lun .EQ. err_lun) .AND.
     .            (redirect_stderr_flags .NE. redirect_none) ) THEN
           IF ( (redirect_stderr_flags .EQ. redirect_journal_tee) .OR.
     .          (redirect_stderr_flags .EQ. redirect_journal) ) THEN
              IF (mode_journal .AND. jrnl_lun.NE.unspecified_int4) THEN
                 WRITE ( jrnl_lun, '(A,A)' ) '!', next_line(:llen)
              ENDIF
           ENDIF
           IF ( (redirect_stderr_flags .EQ. redirect_file_tee) .OR.
     .          (redirect_stderr_flags .EQ. redirect_file) ) THEN
              WRITE ( redirect_stderr_lun, '(A)' ) next_line(:llen)
           ENDIF
           IF ( (redirect_stderr_flags .EQ. redirect_journal_tee) .OR.
     .          (redirect_stderr_flags .EQ. redirect_file_tee) ) THEN
              WRITE ( out_lun, '(A)' ) next_line(:llen)
           ENDIF
	ELSE
	   WRITE ( out_lun, '(A)' ) next_line(:llen)
	ENDIF

	RETURN
	END
