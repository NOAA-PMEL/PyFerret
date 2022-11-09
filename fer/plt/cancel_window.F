      SUBROUTINE CANCEL_WINDOW( status )

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
* delete a window from the display.  If it was currently active then make the
* main window active

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 6/29/88
* V200:	12/20/89 - changed from UIS to Xwindows
*	 3/13/90 - start PPLUS if not already started, check mode_GKS
* V320:  2/28/95 - set a flag for ferret_dispatch so it can keep GUI informed
*		   of window opens/closes
* V400:  6/20/95 - remove changes from 2/28/95 -- state change on windows
*		will be detected by mismatch between wn_open AND 
*		gui_window_reported
* *js* 12.12.97 Don't allow windows to be canceled when running in batch mode
* *kob* 1.29.97 changed "can\'t" to cannot for compiler reasons
*       *acm* 3/12 cleanup ifdefs and unnecessary include files
*       *acm* 3/12 6D Ferret (common uses nferdims in tmap_dims.parm)
* PyFr *kms*  1/13 Broke out the code actually cancelling a window 
*                  into the subroutine CANCEL_WINDOW_NUM.
* PyFr *kms*  7/13 Do not call START_PPLUS; if it is not started,
*                  there are no windows open.
** 10/2022 *acm* Code cleanup: remove references to obsolete gui interface

        IMPLICIT NONE
        include 'tmap_dims.parm'
	include 'errmsg.parm'
	include 'xprog_state.cmn'
	include 'xplot_state.cmn'
	include	'pltcom_dat.decl'	! with sizing data, etc.
	include	'PLTCOM.DAT'	! with sizing data, etc.
	include	'gkscm1_inc.decl'	! with gks common
	include	'GKSCM1.INC'	! with gks common
	
* calling argument declarations:
	INTEGER		status

* internal variable declarations:
	LOGICAL 	TM_DIGIT
	INTEGER 	n, n_lo, n_hi, lo
	CHARACTER	buffer*24

* local parameter declarations:
	INTEGER		slash_all
	PARAMETER     ( slash_all = 0+1 )

* get window numbers to close
	IF ( qual_given( slash_all ) .GT. 0 ) THEN
* ... /ALL
	   n_lo = 1
	   n_hi = max_windows
	
	ELSEIF ( num_args .EQ. 0 ) THEN
* ... CANCEL WINDOW -- what window ?
	   CALL ERRMSG
     .		(ferr_invalid_command,status,'specify window # or /ALL',*5000)

	ELSE
* ... use number given
	   buffer = cmnd_buff(arg_start(1):arg_end(1)) 
	   IF ( .NOT.TM_DIGIT( buffer ) ) GOTO 5020
	   READ ( buffer, *, ERR=5020 ) n
	   n_lo = n
	   n_hi = n
	ENDIF	   

* valid number ?
 60	IF ( n_lo.LT.1 .OR. n_hi.GT.max_windows ) 
     .     CALL ERRMSG( ferr_out_of_range, status, 
     .                  'legal numbers are 1-9', *5000 )

* loop through all requested windows
	DO 100 n = n_lo, n_hi
           CALL CANCEL_WINDOW_NUM(n)
 100	CONTINUE

* successful completion
 1000	status = ferr_ok
	RETURN

* error exit(s)
 5000	RETURN
! 5010	CALL ERRMSG(ferr_erreq,status,'closing window',*5000)
 5020	CALL ERRMSG(ferr_invalid_command,status,buffer,*5000)
	END


*
* Deactive,if active, and close the specified window.  
*
* Made into a separate routine for calling from 
* CREATE_WINDOW when the graphics engine for a window 
* is modified.
*
        SUBROUTINE CANCEL_WINDOW_NUM(n)

        include 'tmap_dims.parm'
	include 'errmsg.parm'
	include 'xprog_state.cmn'
	include 'xplot_state.cmn'
	include	'pltcom_dat.decl'	! with sizing data, etc.
	include	'PLTCOM.DAT'	! with sizing data, etc.
	include	'gkscm1_inc.decl'	! with gks common
	include	'GKSCM1.INC'	! with gks common
        include 'gkscm2.cmn'

* calling argument declarations:
        INTEGER n, lo

        IF ( .NOT.wn_open( n ) ) RETURN

* currently active ?
        IF ( wn_active(n) ) THEN
*          Save the plot to meta_file, if appropriate
           IF ( save_on_exit .AND. imgname_sent ) THEN
              CALL FGD_SAVE_WINDOW(meta_file)
              save_on_exit = .FALSE.
              imgname_sent = .FALSE.
           ENDIF
* ... clear segments from it
           CALL SET_VIEWPORT( mvp_dflt )
* ... deactivate it
           CALL FGD_GDAWK( n )
           wn_active( n ) = .FALSE.
* ... activate lowest available window, if any
           DO 70 lo = 1, max_windows
              IF ( lo.NE.n  .AND. wn_open(lo) ) THEN
                 CALL FGD_GACWK( lo )
                 wn_active( lo ) = .TRUE.
                 wsid = lo
                 GOTO 80
              ENDIF
 70        CONTINUE
           wsid = 0           ! no active window if we get here
        ENDIF
            
* close the specified window
 80     CALL FGD_GCLWK( n )
        wn_open ( n ) = .FALSE.

        RETURN
        END
