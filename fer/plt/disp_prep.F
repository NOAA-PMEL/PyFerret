	SUBROUTINE DISP_PREP( status )

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
* called even for overlays

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 7/28/87
* revision 1.0 - 6/22/88 - added VECSET,
*			   added logical flags lev_given, len_given
* revision 1.1 -  9/8/88 - added cmnd_shade
* revision 1.2 - 4/26/89 - added direct control of PPLUS variable iautoc to
*			   prevent PPLUS from resetting contour intervals when
*			   /LEVELS_SAME is specified
* V200:  1/17/90 - bug fix: set nlev2 so PLOT+ doesn't forget # of contour levs
*		 (when levs are set automatically and then preserved via /LEV)
*	 3/13/90 - added CHECK_GRAPHICS and status return
*	 4/12/90 - same fix needed for /LEVELS is also needed in /LENGTH
*	 4/16/90 - allow /LEVEL="(lo,hi,delta)" and /LENGTH=val
* V313: *sh* AIX port.  Replace "LEV ()" with "LEV,()" so AIX space-squeezing
*		filter doesn't corrupt this routine
* V400:	 8/24/94 - allow /LEVELS=n to set the recommended number of levels
* V4.1+  7/25/95 - allow automatic centered levels and delta-only levels 

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/command.parm'
	include 'ferret_cmn/plot_setup.parm'
	include 'ferret_cmn/xprog_state.cmn'
	include 'tmap_pplv11inc/parampl5_dat.decl'	! with inclen
	include 'pplv11inc/PARAMPL5.DAT'	! with inclen
	include 'tmap_pplv11inc/cont_inc.decl'	! with iautoc
	include 'pplv11inc/CONT.INC'	! with iautoc
	include 'tmap_pplv11inc/vector_inc.decl'	! with vauto
	include 'pplv11inc/VECTOR.INC'	! with vauto
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:COMMAND.PARM'
	INCLUDE 'FERRET_CMN:PLOT_SETUP.PARM'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'TMAP_PPLV11INC:PARAMPL5_DAT.DECL'	! with inclen
	INCLUDE 'PPLV11INC:PARAMPL5.DAT'	! with inclen
	INCLUDE 'TMAP_PPLV11INC:CONT_INC.DECL/LIST'	! with iautoc
	INCLUDE 'PPLV11INC:CONT.INC/LIST'	! with iautoc
	INCLUDE 'TMAP_PPLV11INC:VECTOR_INC.DECL/LIST'	! with vauto
	INCLUDE 'PPLV11INC:VECTOR.INC/LIST'	! with vauto
#endif

* calling argument declarations:
	INTEGER	status

* internal variable declarations:
	LOGICAL TM_DIGIT, lev_given, len_given
	INTEGER	TM_LENSTR1, lev_loc, len_loc, str_len, ii
	REAl	val
	CHARACTER str_buff*80

* local parameter definitions:
	INTEGER		slash_levels,
     .			slash_length
	PARAMETER     (	slash_levels = 17,
     .			slash_length = 17 )

* is the graphics output device open and ready ??
	CALL CHECK_GRAPHICS( status )
	IF ( status .NE. ferr_ok ) RETURN

* initialize
	set_delta = .FALSE.	! 7/95
	centered  = .FALSE.	! 7/95
	ii = 0

* what qualifiers will effect the initialization of PLOT+
	lev_loc = qual_given( slash_levels )
	len_loc = qual_given( slash_length )
	lev_given    = ( cmnd_num.EQ.cmnd_contour .OR. 
     .                   cmnd_num.EQ.cmnd_shade) .AND. lev_loc.NE.0
	len_given = cmnd_num .EQ. cmnd_vector
     .		.AND. len_loc .NE. 0

* contour levels ?
	IF ( lev_given    ) THEN
	   CALL EQUAL_STRING( 
     .          cmnd_buff(qual_start(lev_loc):qual_end(lev_loc)),
     .			      str_buff, status )
	   IF ( str_buff .NE. ' ' ) THEN
* ... pass the levels string given by the user to PLOT+
	      str_len = TM_LENSTR1( str_buff )
	      IF ( str_buff(1:1) .NE. '('
     .	      .OR. str_buff(str_len:str_len) .NE. ')' ) THEN
* ... is this a /LEVELS=nDC specifier or a mistaken forgetting of parentheses?
*	/LEVELS=n for n levels
*	/LEVELS=nC for n centered levels
*	/LEVELS=xxD for levels specified by delta value
*	/LEVELS=xxDC for centered levels specified by delta value
	         centered =  str_buff(str_len:str_len) .EQ. 'c'
     .          	.OR. str_buff(str_len:str_len) .EQ. 'C'
	         IF ( centered ) THEN
		    str_len = str_len - 1
	            qual_end(lev_loc) = qual_end(lev_loc) - 1
	         ENDIF
* ... if the user entered /LEVELS=C then str_len just became zero
	         IF ( str_len .GT. 0 ) THEN
	            set_delta = str_buff(str_len:str_len) .EQ. 'd'
     .          	   .OR. str_buff(str_len:str_len) .EQ. 'D'
	            IF ( set_delta ) THEN
		       str_len = str_len - 1
	               qual_end(lev_loc) = qual_end(lev_loc) - 1
	            ENDIF
	            IF (str_len .EQ. 0 ) GOTO 5120
	            IF (INDEX(str_buff(1:str_len),',') .GT. 0
     .	           .OR. .NOT.TM_DIGIT(str_buff(1:1))          ) GOTO 5100
	            CALL EQUAL_VAL(
     .	              cmnd_buff(qual_start(lev_loc):qual_end(lev_loc)),
     .		      val, status )
	            IF ( status .NE. ferr_ok ) GOTO 5000
		    IF ( set_delta ) THEN
		       iautoc = 1
	               delta = val
	            ELSE
	               IF (val.LE.0 .OR. val.GT.250) GOTO 5150 
* ... set the /LEVELS=n recommendation specifier
		       iautoc = 1
	               nlev = val   ! dont fully know why this works ...
	            ENDIF
	         ELSE
		    iautoc = 1
	            nlev = 10	! "C" is equivalent to "10C"
	         ENDIF
	      ELSE
	         CALL PPLCMD( from, line, 0,
     .			      'LEV,()'//str_buff(:str_len), 1, 1 )
	      ENDIF
	   ELSEIF ( iautoc .EQ. 1 ) THEN
* ... if levels were previously set automatically by PLOT+ and user has now
*     requested to reuse them --> make PLOT+ think they were set manually
	      iautoc = 0
	      nlev = nlev2 ! reuse # of levels from last plot
	   ENDIF
	ELSE
* ... set up for automatic levels
	   CALL PPLCMD ( from, line, 0, 'LEV,()', 1, 1 )
	ENDIF

* vector length ?
	IF ( len_given ) THEN
	   CALL EQUAL_STRING( 
     .           cmnd_buff(qual_start(len_loc):qual_end(len_loc)),
     .			      str_buff, status )
	   IF ( str_buff .NE. ' ' ) THEN
	      CALL EQUAL_VAL( 
     .             cmnd_buff(qual_start(len_loc):qual_end(len_loc)),
     .			      val, status )	! syntax check
	      IF ( status .NE. ferr_ok ) GOTO 5000
	      IF ( val .LE. 0.0 ) GOTO 5200
	      str_len = TM_LENSTR1( str_buff )
* ... send specified vector value to PLOT+
	      CALL PPLCMD ( from,line,0,'VECSET ,'//str_buff(:str_len),1,1 )
	   ELSE
* ... preserve previous length - possible auto length from PLOT+
	      IF ( vusrln .EQ. 0.0 ) vusrln = 1.0 ! has vec len been init'ed ?
	      vauto = .FALSE.
	   ENDIF
	ELSE
* ... set up for automatic vector lengths
	   CALL PPLCMD ( from, line, 0, 'VECSET', 1, 1 )
	ENDIF

* success
	RETURN

* error exit
 5000	RETURN
 5100	CALL ERRMSG( ferr_syntax, status,
     .		str_buff(:str_len)//' must be in parentheses', *5000 )
 5120	CALL ERRMSG( ferr_syntax, status,
     .		'LEVELS=xxD with xx value missing', *5000 )
 5150	CALL ERRMSG( ferr_invalid_command, status,
     .		'"'//str_buff(:str_len)//'": must be between 1 and 250',
     .		*5000 )
 5200	CALL ERRMSG( ferr_invalid_command, status,
     .		'zero vector length on /LENGTH=', *5000 )
	END
