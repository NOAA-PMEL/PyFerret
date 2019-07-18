	SUBROUTINE RIBBON_PLOTKEY_SETUP (overlay)

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
*  derivative works and supporting documentation.  Further, the user5/4/2005
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
* V7.4+ ACM 7/18  DSG file development
*	Set up the color key for a ribbon or trajectory plot

	include 'tmap_dims.parm'
	include	'ferret.parm'
	include 'errmsg.parm'
	include 'xplot_state.cmn'
	include 'plot_setup.parm'
	include 'xprog_state.cmn'
	include 'xplot_setup.cmn'
	include 'slash.parm'
	include 'parampl5_dat.decl'
	include 'PARAMPL5.DAT'
	include 'cont_inc.decl'	! with nlev
	include 'CONT.INC'

	LOGICAL overlay

	LOGICAL TM_HAS_STRING, do_key
	INTEGER	TM_LENSTR, loc, status
	chARACTER buff*128

* default color key
	do_key =  (qual_given(slash_plot_key)  .GT. 0)
     .		.OR. ( (qual_given(slash_plot_key) .EQ. 0)
     .		        .AND. .NOT.overlay)  ! .AND. .NOT.no_data
	IF (qual_given(slash_plot_nokey) .gt. 0 ) do_key = .FALSE.
	IF ( do_key ) THEN
	   CALL PPLCMD ( from, line, 0, 'SHAKEY 1,1', 1, 1 )

* ... /KEY[=continuous,horizontal,centerlabel] ?
*     /KEY=title is a legal option, but ignored for ribbon plot.

	   loc = qual_given( slash_plot_key )
	   changed_key = .FALSE.

	   IF ( loc .GT. 0) THEN
	      CALL EQUAL_STRING( cmnd_buff(qual_start(loc):qual_end(loc)),
     .		 buff, status )
	      IF ( status .NE. ferr_ok ) GOTO 5000
	      IF (TM_LENSTR(buff) .GT. 0) THEN
		 changed_key = .FALSE.
		 IF ( TM_HAS_STRING(buff, 'CON') ) THEN
		   changed_key = .TRUE.
		   CALL PPLCMD (from, line, 0, 'S_KEY 1', 1, 1 )
	         ENDIF
	         IF ( TM_HAS_STRING(buff, 'CEN') ) THEN
		    changed_key = .TRUE.
		    CALL PPLCMD (from, line, 0, 'S_KEY 2', 1, 1 )
	         ENDIF
	         IF ( TM_HAS_STRING(buff, 'HOR') ) THEN
		    changed_key = .TRUE.
		    CALL PPLCMD (from, line, 0, 'S_KEY 3', 1, 1 )
	         ENDIF
	         IF ( TM_HAS_STRING(buff, 'TIT') ) THEN  
		    changed_key = .TRUE.
		    use_keys = .FALSE.
	         ENDIF
	         IF (.NOT. changed_key) GOTO 5600
	      
	      ENDIF ! TM_LENSTR(buff)

	   ENDIF ! loc .GT. 0 for /KEY=

	ELSE  ! do_key
	   CALL PPLCMD ( from, line, 0, 'SHAKEY 0,1', 1, 1 )
	ENDIF ! do_key

 5000	RETURN
 5600	CALL ERRMSG( ferr_syntax, status, 
     .		cmnd_buff(qual_start(loc):qual_end(loc))//pCR//
     .		'Unrecognized argument to KEY qualifier', *5000 )
	END
