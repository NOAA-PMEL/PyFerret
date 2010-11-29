	SUBROUTINE SHOW_GRID( grid, cx )

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
* describe the 4 axes of a grid and then give coordinate data about those
* axes as indicated in cx

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 9/23/87
* revision 0.1 - 2/18/88 - checks subscript limits on axes
*			   and allows non-geographical, +/- oriented axes
* revision 0.2 - 3/18/88 - decimal places automated for sig. figs.
*			 - time step written in calendar mode
*			 - interruptable; - axis unit labels via AX_TITLE
* revision 1.0 - 8/1/88  - created LINE_FACTS and removed half of code to it
* revision 1.1 - 8/16/88 - eliminated units column, added delta in coord display
* V200:  5/19/89 - reordered lo/hi_ss arguments
*        9/22/89 - change to LINE_FACTS call for abstract axis orientations
* Unix/RISC port 3/18/91 *sh*: can't use function in WRITE list
*                            - corrected use of risc_buff (was right justified)
* V300:  5/28/93 - bug fix: incorrect branch for mnormal or munknown axis
* V320: 12/29/94 - use SPLIT_LIST to redirect tty output for GUI
*	  1/5/95 - changed default precision from 5 to 7  (and for
*		   dates to seconds) 
* kob*   9/97 - replaced call to ISUBSCRIPT w/ call to ISUBSCR_CX in order
*               to have access to context information - needed for negative
*               time step processing
* *kob*  1/98 - add calls to convert cx_lo/hi_ww to tsteps before calling
*               ISUBSCR_CX
* V500 - *sh* 4/99 - extracted guts of code to SHOW_LINE_COORDS to be shared
*		by the SHOW AXIS command

#ifdef unix
	include	'ferret.parm'
	include	'slash.parm'
	include 'errmsg.parm'
	include 'xprog_state.cmn'
	include 'xrisc.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:SLASH.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
#endif

* calling argument declarations
	INTEGER grid, cx

* local variable declarations:
	INTEGER idim

* list and describe the 4 axes of the grid
	risc_buff = ' '
	WRITE ( risc_buff, 3000 )
	CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
	DO 110 idim = 1, 4
	   CALL LINE_FACTS( ttout_lun, idim, grid, idim )
 110	CONTINUE

* list the data on any axis that has been especially requested
*		via (eg.) /X=lo:hi or /X alone
	DO 130 idim = 1, 4
	   IF ( qual_given( slash_show_grid_x0+idim ) .GT. 0
     .	  .OR.  qual_given( slash_show_grid_i0+idim ) .GT. 0 ) THEN
	     CALL SHOW_LINE_COORDS( grid, cx, idim )
	   ENDIF
 130	CONTINUE

* successful completion
 1000	RETURN

 3000	FORMAT (T2,'name',T13,'axis',T31,'# pts',T39,'start',T60,
     .          'end' )

	END

