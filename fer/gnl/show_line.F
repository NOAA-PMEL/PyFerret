	SUBROUTINE SHOW_line( line, brief )

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
* display basic facts about the given line

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 8/1/88
* V200:  9/22/89 - added "orient" for axes without particular orientations
* V320: 12/29/94 - use SPLIT_LIST to redirect tty output for GUI
* V530:  3/01    - indicate the calendar for non-Gregorian calendar types 

#ifdef unix
	include 'tmap_dims.parm'
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include 'xunits.cmn_text'
	external xunits_data
	include	'ferret.parm'
	include 'xprog_state.cmn'
	include 'xtext_info.cmn'
	include 'calendar.decl'
	include 'calendar.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE 'TMAP_FORMAT:XUNITS.CMN'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XTEXT_INFO.CMN'
#endif

* calling argument declarations
	LOGICAL		brief
	INTEGER 	line

* local variable declarations:
	INTEGER		idim, orient
	INTEGER		TM_LENSTR1, TM_GET_CALENDAR_ID, cal_id, len
	CHARACTER	cal_name*32

* determine orientation
	DO 100 idim = 1, 5
	   IF ( line_direction(line) .EQ. axis_orients( idim ) ) THEN
	      orient = idim
	      GOTO 110
	   ENDIF
 100	CONTINUE
	orient = 0
	idim   = 1			! use X slot for non-oriented
 110	IF ( idim .EQ. 5 ) idim = 3	! DU --> UD on Z axis

* create a dummy grid with this axis
* (kludge to use existing code that works with grids)
	grid_line(idim, mgrid_buff) = line

* report basic facts using code from SHOW_GRID
	CALL LINE_FACTS( ttout_lun, idim, mgrid_buff, orient )
	IF ( brief ) RETURN

* report /T0= for formatted time axis
	IF ( line_direction(line) .EQ. 'TI' ) THEN
	   CALL SPLIT_LIST
     .		(pttmode_explct, ttout_lun, 'T0 = '//line_t0(line), 0)

	   cal_name = line_cal_name (line)
	   cal_id = TM_GET_CALENDAR_ID (cal_name)
	   IF (cal_id .GT. gregorian .AND. cal_id .LE. max_calendars) THEN
	      len = TM_LENSTR1(cal_name)
	      CALL SPLIT_LIST (pttmode_explct, 
     .		          ttout_lun, 'CALENDAR = '//cal_name(:len), 0)
	   ENDIF
	ENDIF

	RETURN
	END
