	SUBROUTINE SHOW_LINE_XML( lun, line)

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
* display basic facts about the given line in xml-style format 

* programmer - Ansley Manke, based on show_line.F
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* 6-Aug-2003
* V581 *acm* 6/05 - For fix to bug 1271, add flag line_shift_origin
*            and restore original t0 on output

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
	include 'xrisc.cmn'

* calling argument declarations
	INTEGER 	lun, line

* local variable declarations:
	LOGICAL		TM_ITS_SUBSPAN_MODULO
	INTEGER		idim, orient, slen
	INTEGER		TM_LENSTR1, TM_GET_CALENDAR_ID, cal_id, status
	CHARACTER	TM_FMT*32, buff*80, outbuff*20

 2020	FORMAT ( '<time_origin>', A, '</time_origin>' )
 2030	FORMAT ( '<calendar>', A, '</calendar>' )
 2040	FORMAT ( '<modulo_length>', A, '</modulo_length>' )
 2050	FORMAT ( '</axis>' )

* determine orientation

	DO 100 idim = 1, 5
	   IF ( line_direction(line) .EQ. axis_orients( idim ) ) THEN
	      orient = idim
	      GOTO 110
	   ENDIF
 100	CONTINUE
        IF (line_direction(line) .EQ. 'XX') THEN
           orient = 1
           idim = 1
           GOTO 110
        ENDIF
        IF (line_direction(line) .EQ. 'YY') THEN
           orient = 2
           idim = 2
           GOTO 110
        ENDIF
        IF (line_direction(line) .EQ. 'TT') THEN
           orient = 4
           idim = 4
           GOTO 110
        ENDIF

	orient = 0
	idim   = 1			! use X slot for non-oriented
 110	IF ( idim .EQ. 5 ) idim = 3	! DU --> UD on Z axis

* create a dummy grid with this axis
* (kludge to use existing code that works with grids)
	grid_line(idim, mgrid_buff) = line

* report basic facts using code from SHOW_GRID
	CALL LINE_FACTS_XML( ttout_lun, idim, mgrid_buff, orient )

* report /T0= for formatted time axis
	IF ( line_direction(line) .EQ. 'TI' ) THEN
        
	   buff = line_cal_name (line)
	   cal_id = TM_GET_CALENDAR_ID (buff)

           IF (line_shift_origin(line)) THEN
              buff = '01-JAN-0001 00:00:00'
           ELSE
              buff = line_t0(line)
           ENDIF

           CALL TM_DATE_REFORMAT (buff, 1, cal_id, outbuff, 2, status) 
           slen = TM_LENSTR1(outbuff)
           WRITE ( risc_buff, 2020 ) outbuff(1:slen)
	   CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

* report the calendar of a time axis if "interesting"

	   IF (cal_id .GT. gregorian .AND. cal_id .LE. max_calendars) THEN
	      slen = TM_LENSTR1(buff)
           WRITE ( risc_buff, 2030 ) buff(1:slen)
	   CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
	   ENDIF
	ENDIF

* if line is subspan modulo, report its modulo cycle length
	IF ( line_modulo(line) .AND. TM_ITS_SUBSPAN_MODULO(line) ) THEN
	   buff = TM_FMT( SNGL(line_modulo_len(line)),7,32,slen)
	   WRITE (risc_buff, 2040) buff(1:slen)
	   CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
	ENDIF

* closer tag

        WRITE ( risc_buff, 2050 ) 
	CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

	RETURN
	END



