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
* prepare the plot package (PPLUS) for another variable.  This routine is
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
* v500 - 1/26/99 - *kob* add PPLCMD call for aline reinitialization - needed for
*                   nolabel plot overlays
* V608 - 10/07   - *acm* Give more detail int he error message when the number 
*                  of contour levels given is too large
* V630  *acm* 9/09 Introduction of syntax for variance-based and histogram levels
* V670  *acm* 2/11 Fix bug 1803. Centered levels confused with variance-based centered
*                  levels.
* V68  *acm* 1/12  changes for double-precision ferret, single-precision pplus.
*       *acm* 3/12 6D Ferret (common uses nferdims in tmap_dims.parm)
* V685+ 12/13    - *acm*  bounds checking; dont refer to parts of the string outside its bounds.
* V6.86 *acm* 1/14 In above fix, let there be spaces before the file spec.
* V694   5/15    - *acm*  Ticket 2285: need to initialize iautoc to = 1 so overlay with /LEV=20V works.
* V695  *acm* 9/15 ticket 2311: variance-based levels on constant variables
* V695  *acm* 9/15 (not fully implemented) INT option for levels
* v697 *acm* 12/15 Ticket 2331, add flag changed_key to common, and restore the color key
*                  style after a SHADE/SET; SHADE.
* v698 *acm*  2/16 Initialize the use_hist flag to false
* v698 *acm*  2/16 Allow for /LEV=h, histogram levels without specifying a number.

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'command.parm'
	include 'slash.parm'
	include 'plot_setup.parm'
	include 'xplot_setup.cmn'
	include 'xplot_state.cmn'
	include 'xprog_state.cmn'
	include 'parampl5_dat.decl'	! with inclen
	include 'PARAMPL5.DAT'	! with inclen
	include 'cont_inc.decl'	! with iautoc
	include 'CONT.INC'	! with iautoc
	include 'vector_inc.decl'	! with vauto
	include 'VECTOR.INC'	! with vauto

* calling argument declarations:
	INTEGER	status

* internal variable declarations:
	LOGICAL TM_DIGIT, lev_given, len_given,
     .		is_delta, is_std, is_mean, is_min, is_max, is_hist,
     .          is_centered, is_openlev, is_int, is_cv, get_more,
     .		has_std , has_mean, has_min, has_max, has_int
	INTEGER	TM_LENSTR1,TM_LENSTR, lev_loc, len_loc, str_1, str_len, 
     .          ii, ilen, ier, ist, nc, set_default_levs
	REAL	val
	REAL*4  rbad_val4

* v500 - 1/26/99 *kob* double buffer size for /levels text
* v532 - 5/11/01 *kob* increase buffer size for /levels text to 1028
	CHARACTER str_buff*1028
        CHARACTER*255 labsym_buff
	CHARACTER*1 cbuf

	CHARACTER buffer*40, buff30*30

* is the graphics output device open and ready ??
	CALL CHECK_GRAPHICS( status )
	IF ( status .NE. ferr_ok ) RETURN

* initialize
	centered  = .FALSE.
	openlev = .FALSE.
	use_hist = .FALSE.	
	set_std = .FALSE.
	set_mean = .FALSE.
	set_min = .FALSE.
	set_max = .FALSE.
	set_delta = .FALSE.
	need_histo = .FALSE.
        need_std = .FALSE.

	rbad_val4 = bad_val4
	lev_mean = rbad_val4
	lev_std = rbad_val4
	lev_min = rbad_val4
	lev_max = rbad_val4

	ii = 0

* reinitialize - *kob* 
* forget line specimen definitions that were used in previous plots
        CALL PPLCMD ( from, line, 0, 'ALINE ', 1, 1 )

* if the colorkey was changed by a previous key=center, etc reset it.
        IF (changed_key ) THEN
           CALL PPLCMD (from, line, 0, 'S_KEY 0', 1, 1 )
           changed_key = .FALSE.
        ENDIF

* what qualifiers will effect the initialization of PLOT+

	lev_loc = qual_given( slash_levels )
	IF (cmnd_num.EQ.cmnd_plot) lev_loc = qual_given( slash_plot_levels )
	len_loc = qual_given( slash_length )
	lev_given    = ( cmnd_num.EQ.cmnd_contour .OR. 
     .                   cmnd_num.EQ.cmnd_shade   .OR.
     .                   cmnd_num.EQ.cmnd_polygon .OR. 
     .                   cmnd_num.EQ.cmnd_plot) .AND. lev_loc.NE.0
	len_given = cmnd_num .EQ. cmnd_vector
     .		.AND. len_loc .NE. 0

        contour_fewer = 1
        need_std = .FALSE.

	
        intlev = 0

	IF ( lev_given    ) THEN
           contour_fewer = 0
           CALL EQUAL_STRING(  
     .          cmnd_buff(qual_start(lev_loc):qual_end(lev_loc)), 
     .                        str_buff, status ) 

* Set symbol LEV_TEXT with the users input to the LABEL command.
* (Note that the above call has put str_buff into uppercase.)

        labsym_buff = "LEV_TEXT "//str_buff
        ilen = TM_LENSTR1(labsym_buff)
        ier = 0
        ist = 0
        CALL SETSYM (labsym_buff, ilen, ier, ist)

	IF ( str_buff .NE. ' ' ) THEN
	
	   neginf = .FALSE.
	   posinf = .FALSE.

* ... pass the levels string given by the user to PLOT+
	      str_len = TM_LENSTR1( str_buff )
	      IF ( str_buff(1:1) .NE. '('
     .	      .OR. str_buff(str_len:str_len) .NE. ')' ) THEN
* ... is this a /LEVELS=nDC specifier or a mistaken forgetting of parentheses?
*	/LEVELS=n for n levels
*	/LEVELS=nC for n centered levels
*	/LEVELS=xxD for levels specified by delta value
*	/LEVELS=xxDC for centered levels specified by delta value
*	/LEVELS=nv for n open-ended levels (possibly with specified MEAN and STD)
*	/LEVELS=xxMEAN for open-ended levels centered about mean value
*	/LEVELS=xxSTD for open-ended levels using given STD value
*	/LEVELS=xxMIN for open-ended levels on top, closed at MIN below
*	/LEVELS=xxMAX for open-ended levels on bottom, closed at MAX above

 1000            CONTINUE
	         is_std = .FALSE.
	         is_mean = .FALSE.
	         is_min = .FALSE.
	         is_max = .FALSE.
	         is_int = .FALSE.
	         is_hist = .FALSE.
	         is_delta = .FALSE.
	         is_cv = .FALSE.

		 iautoc = 1

	         val = bad_val4

	         is_centered =  str_buff(str_len:str_len) .EQ. 'C'
	         is_openlev =  str_buff(str_len:str_len) .EQ. 'V'

	         IF ( is_centered ) THEN
		    centered = .TRUE.
		    str_len = str_len - 1
	            qual_end(lev_loc) = qual_end(lev_loc) - 1
		    is_cv = .TRUE.
	         ENDIF
	         IF ( is_openlev ) THEN
		    openlev = .TRUE.
		    str_len = str_len - 1
	            qual_end(lev_loc) = qual_end(lev_loc) - 1
		    is_cv = .TRUE.
	         ENDIF

* set up some logicals -- with bounds checking we cannot refer to, e.g. 
* str_buff(str_len-2:str_len)  if str_len is less than 3...

		 has_std = .FALSE.
		 IF (str_len .GE. 3) THEN
		    has_std = str_buff(str_len-2:str_len) .EQ. 'STD'
		 ENDIF

		 has_mean = .FALSE.
                 IF (str_len .GE. 4) THEN
		    has_mean = str_buff(str_len-3:str_len) .EQ. 'MEAN'
		 ENDIF
		 
		 has_min = .FALSE.
		 IF (str_len .GE. 3) THEN
		    has_min = str_buff(str_len-2:str_len) .EQ. 'MIN'
		 ENDIF

		 has_max = .FALSE.
		 IF (str_len .GE. 3) THEN
		    has_max = str_buff(str_len-2:str_len) .EQ. 'MAX'
		 ENDIF

		 has_int = .FALSE.
		 IF (str_len .GE. 3) THEN
		    has_int = str_buff(str_len-2:str_len) .EQ. 'INT'
		 ENDIF

	         IF ( str_len .GT. 0 ) THEN
	            set_delta = str_buff(str_len:str_len) .EQ. 'D'
	            is_delta = str_buff(str_len:str_len) .EQ. 'D'
                    IF (has_std) THEN
                          set_delta = .FALSE.
			  is_delta = .FALSE.
		          is_std = .TRUE.
			  set_std = .TRUE.
			  openlev = .TRUE.
			  is_cv = .TRUE.
			  str_len = str_len - 3
	                  qual_end(lev_loc) = qual_end(lev_loc) - 3
                    ENDIF 
	            IF (str_len .EQ. 0 ) GOTO 5120

	            IF ( set_delta ) THEN
		       str_len = str_len - 1
	               qual_end(lev_loc) = qual_end(lev_loc) - 1
	            ENDIF
	            IF (str_len .EQ. 0 ) GOTO 5120

                    IF (has_mean) THEN
                       is_mean = .TRUE.
                       set_mean = .TRUE.
                       centered = .TRUE.
                       is_cv = .TRUE.
                       str_len = str_len - 4
                       qual_end(lev_loc) = qual_end(lev_loc) - 4
                    ENDIF 
	            IF (str_len .EQ. 0 ) GOTO 5120

                    IF (has_min) THEN
                       is_min = .TRUE.
                       set_min = .TRUE.
                       str_len = str_len - 3
                       qual_end(lev_loc) = qual_end(lev_loc) - 3
                    ENDIF 
	            IF (str_len .EQ. 0 ) GOTO 5120

                    IF (has_max) THEN
                       is_max = .TRUE.
                       set_max = .TRUE.
                       str_len = str_len - 3
                       qual_end(lev_loc) = qual_end(lev_loc) - 3
                    ENDIF 

                    IF (has_int) THEN
                       is_int = .TRUE.
		       intlev = 1
                       str_len = str_len - 3
                       qual_end(lev_loc) = qual_end(lev_loc) - 3
		    ENDIF

                    IF (str_buff(str_len:str_len) .EQ. 'H') THEN  ! histogram
                       is_hist = .TRUE.
                       str_len = str_len - 1
		       IF (str_len .NE. 0) qual_end(lev_loc) = qual_end(lev_loc) - 1
                    ENDIF 

                    str_1 = qual_start(lev_loc)

		    get_more = .FALSE.
	            IF (INDEX(str_buff(1:str_len),',') .GT. 0) THEN
		       str_1 = str_len
		       DO WHILE (INDEX(str_buff(str_1:str_1),',') .EQ. 0)
		          str_1 = str_1 - 1
	               ENDDO
		       str_1 = str_1 + 1
		       IF (str_1 .LE. str_len) THEN
      	                  READ (str_buff(str_1:str_len), *, err=5000) val
		          qual_end(lev_loc) = 
     .		          qual_end(lev_loc) - (str_len-str_1+1)  ! numb. len
                          get_more = .TRUE.
		       ENDIF
		       qual_end(lev_loc) = qual_end(lev_loc) - 1 ! comma
                       
                       str_1 = str_1 - 2
                       str_len = str_1
                    ELSE

* If its a number read it, otherwise will go back for another letter code.
		       get_more = .FALSE.
		       IF (qual_start(lev_loc) .LE. qual_end(lev_loc)) THEN
		          labsym_buff = 
     .                  cmnd_buff(qual_start(lev_loc):qual_end(lev_loc))
		          cbuf = 
     .                  cmnd_buff(qual_end(lev_loc):qual_end(lev_loc))
           	          IF ( TM_DIGIT(cbuf) ) THEN
     	                     CALL EQUAL_VAL(labsym_buff, val, status )
                             str_1 = 1
			  ELSE
			     get_more = .TRUE.  ! Get another letter-code
			  ENDIF
		       ENDIF
                    ENDIF

	            IF ( status .NE. ferr_ok ) GOTO 5000

 500		    CONTINUE
		    IF ( is_delta ) THEN
		       iautoc = 1
	               delta = val
	            ELSEIF (is_std) THEN
		       openlev = .TRUE.
	               lev_std = val
		       IF (lev_std .LE. 0.) GOTO 5210
	            ELSEIF ( is_mean) THEN
		       lev_mean = val
	            ELSEIF ( is_min) THEN
		       lev_min = val
	            ELSEIF ( is_max) THEN
		       lev_max = val
	            ELSEIF ( is_int) THEN
		       val = 0
		       intlev = 1
	            ELSEIF ( is_hist) THEN
		       lev_hist = val
		       need_histo = .TRUE.
		       IF (val .NE. bad_val4) THEN
		          nlev = val
		          IF (val.LE.0. .OR. val.GT.250.) GOTO 5160
		       ENDIF
		       IF (str_len .EQ. 0) THEN
		          lev_hist = nlev
		          get_more = .FALSE.
			  str_1 = 1
		       ENDIF
	            ELSEIF ( is_cv) THEN
		       IF (val .NE. bad_val4) THEN
		          nlev = val
			  iautoc = 1
		          IF (val.LE.0. .OR. val.GT.250.) GOTO 5150
		       ENDIF
		    ELSE

	               IF (val.LE.0 .OR. val.GT.250) GOTO 5150 
* ... set the /LEVELS=n recommendation specifier
		       iautoc = 1
	               nlev = val   ! levels set in PPLUS plotting routine
	            ENDIF
		    IF  (str_1 .GT. 1) GOTO 1000
		    IF  (get_more) GOTO 1000

	         ELSE
		    iautoc = 1
! "C" is equivalent to "nC" where n is the current mode nlevels setting
	            nlev = mode_arg(pmode_nlevels, 1)       
		    contour_fewer = 1  ! if its a contour line plot
	         ENDIF

* If requested both open-ended values and a min and max, use only
* the min and max, regardless of any Mean and Std settings
	         IF (set_min .AND. set_max) THEN
	            openlev = .FALSE.
	            SET_STD = .FALSE.
	            SET_MEAN = .FALSE.
	            LEV_STD = rbad_val4
	            LEV_MEAN = rbad_val4
	         ENDIF

* If requested centered, but no mean value given, set the mean to zero.
	         IF (centered) THEN
	            IF (LEV_MEAN .EQ. rbad_val4) LEV_MEAN = 0.
	            set_mean = .TRUE.
	         ENDIF

		 IF (is_int) THEN
		    openlev = .FALSE.
		    need_histo = .FALSE.
		    IF (set_min) lev_min = FLOOR(lev_min)
		    IF (set_max) lev_max = CEILING(lev_max)
		 ENDIF

* The processing always gets the data max/min.
* Flag to indicate whether we need to get mean/std from the data.
	         need_std = .FALSE.

		 IF (openlev .AND. 
     .               ( LEV_STD.EQ.rbad_val4 .OR. LEV_MEAN.EQ.rbad_val4) )
     .              need_std = .TRUE.

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

* ... For ticket 2564. Calls to load the palette and colors, PPL SHASET, use
*     the stored setting shd_levels in the call to setup_color_table. In some
*     instances that previous setting causes colors to be computed for non-existent
*     levels and rgb colors outside the range [0:1]. Set shd_levels to correspond to 
*     the nlev for this plot.

	   CALL SET_NSHD_LEVELS (nlev)
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
     .		'"'//str_buff(:str_len)//
     . '": Number of contour levels must be between 1 and 250',
     .		*5000 )
 5160	CALL ERRMSG( ferr_invalid_command, status,
     .		'"'//str_buff(:str_len)//
     . '": Number of histogram levels must be between 1 and 250',
     .		*5000 )
 5200	CALL ERRMSG( ferr_invalid_command, status,
     .		'zero vector length on /LENGTH=', *5000 )
 5210	CALL ERRMSG( ferr_out_of_range, status,
     .		'STD must be a positive number', *5000 )
	END
