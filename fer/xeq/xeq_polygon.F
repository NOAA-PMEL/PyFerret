	SUBROUTINE XEQ_POLYGON( memory )

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
* execute the POLYGON command

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V500:  *sh* and *jd*
*        *jd  fix for v5.01 for fill if /patt or /pal given 6.7.99
*        *sh* 6/99 -- suppress color key for 2-argument version of command
* V510:  *sh* 3/00 -- change to how line specifications are passed

* V533 *acm*6/01 - add arguments is_logh, is_logv to PLOT_SET_UP call, 
*                  set log axes, inverse log axes and log depth axes with
*                  PLOT/VLOG or PLOT/HLOG
* V541 *acm*  3/02  Make PPL AXSET settings persist, but not settings
*                   done with qualifiers /AXES= or /NOAXES.
* v541 *acm*  5/02 add option to control outline on the shade key boxes 
*                   /KEY=continuous


	include 'tmap_dims.parm'
	include	'ferret.parm'
	include 'errmsg.parm'
	include 'plot_setup.parm'
	include 'slash.parm'
	include 'xcontext.cmn'
	include 'xvariables.cmn'
	include 'xprog_state.cmn'
	include 'xplot_state.cmn'
	include 'xplot_setup.cmn'
        include 'switch_inc.decl'
        include 'SWITCH.INC'
        include 'data_inc.decl'
        include 'DATA.INC'
        include 'PPLDAT.INC'
        include 'parampl5_dat.decl'
        include 'PARAMPL5.DAT'
        include 'lines_inc.decl'
        include 'LINES.INC'

	
* calling argument declarations:
	REAL	memory( mem_blk_size, max_mem_blks )

* internal variable declarations:
        LOGICAL         no_range, do_key, spectrum, pattern, use_line
	LOGICAL		overlay, transpz, no_labels, set_up, noaxes, axes
	LOGICAL         line_it, fill_it, is_logh, is_logv
	INTEGER         sp, patt, TM_LENSTR1
	INTEGER		symbol, color, status,
     .			loc, limit, slen, coord_ax, thick,
     .			ax1_blks, ax1_start,
     .			ax2_blks, ax2_start,
     .			s1, s2, it_start(4), it_end(4),
     .			nparm,  num_it, i, iax(4)
	REAL		val, only_val, sym_size
	CHARACTER	LEFINT*4, TM_FMT*12, fname*128, poly_opt*10,
     .                  buff*25

* internal parameter declarations
	INTEGER ncolors
	PARAMETER (ncolors = 6)
        LOGICAL cont_key

* set up the graphics environment and get the data
	CALL START_PPLUS

	CALL DISP_DATA_SET_UP( memory,
     .		ax1_start, ax1_blks, ax2_start, ax2_blks, status )
	IF ( status .NE. ferr_ok ) GOTO 5100

* Initialize the lines buffer in PPL before setting up
	overlay   = qual_given( slash_plot_over_plot ).GT.0
	if (overlay) SOVER = .true.

	if (sover) then
CC      *jd* NLINES initializes SOVER, and resets MIN/MAX. Don't execute. 
CC       => Set values here instead.
	   LINEN=0
	   IBASE=1
	   PBUF=0
	   LNUM=1
	else
	   CALL PPLCMD ( from, line, 0, 'NLINES', 1, 1 )
	endif

* initialize
	no_labels = qual_given( slash_plot_nolabels  ).GT.0
	transpz   = qual_given( slash_plot_transpz   ).GT.0
	set_up	  = qual_given( slash_shad_set_up    ).GT.0
	axes      = qual_given( slash_polygon_axes   ).GT.0
	noaxes    = qual_given( slash_noaxes         ).GT.0
	coord_ax  = qual_given( slash_plot_vs	     )		! integer
	is_logh   = qual_given( slash_is_logh        ).GT.0
	is_logv   = qual_given( slash_is_logv        ).GT.0 

	symbol	  = qual_off	! or ??unspecified_int4??
	sym_size  = 0.08	! default plot mark size
	thick	  = 1		! default thickness
	color	  = unspecified_int4
	use_line = .FALSE.
        sp        = qual_given( slash_cont_spectrum  )

        spectrum  = sp .GT. 0 
        patt      = qual_given( slash_shad_pattern   )
        pattern   = patt .GT. 0


	IF ( .NOT.overlay ) CALL DISP_INIT(no_labels,is_cx,num_uvars_in_cmnd)
	CALL DISP_PREP( status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* decode qualifiers
* ... POLYGON/COLOR=n
	loc = qual_given( slash_polygon_color )
	IF ( loc .GT. 0  ) THEN
	   CALL EQUAL_COLOR( cmnd_buff(qual_start(loc):qual_end(loc)),
     .                           color, status )
           IF ( status .NE. ferr_ok ) GOTO 5000
	   use_line = .TRUE.
	ENDIF
* ... POLYGON/THICKNESS=n
 200	loc = qual_given( slash_polygon_thick )
	IF ( loc .GT. 0  ) THEN
	   use_line = .TRUE.
	   IF ( color .EQ. unspecified_int4 ) color = 1  ! /THICK -> "BLACK"
	   color = MOD(color-1,ncolors)+1  ! eliminate thickness
	   CALL EQUAL_VAL( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			   val, status )	! syntax check
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   IF ( val .NE. unspecified_val4 ) THEN
	      IF ( val.LE.0.0 .OR. val.GT.3 ) GOTO 5400
	      thick = NINT( val )	! user-specified line style
	   ELSE
	      thick = 2  ! default if no value is specified
	   ENDIF
	   color = ncolors*(thick-1) + color
	ENDIF
* ... POLYGON/LINE_STYLE=n
	loc = qual_given( slash_polygon_line )
	IF ( loc .GT. 0  ) THEN
	   use_line = .TRUE.
	   CALL EQUAL_VAL( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			      val, status )	! syntax check
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   IF ( val .NE. unspecified_val4 ) THEN
	      limit = num_fancy_lines 			!was num_plain 1/94
	      IF ( val.LE. 0.0 .OR. val.GT.limit ) GOTO 5200
	      color = NINT( val )	! user-specified line style
	   ELSEIF ( color .EQ. unspecified_int4 ) THEN
	      color = qual_on		! auto selection of line style
	   ENDIF
	ENDIF


* decide on line or fill or both
	line_it   = use_line 

	if (line_it) then
	   fill_it = qual_given(slash_polygon_fill) .gt. 0 
     .               .or. spectrum .or. pattern
	else
	   fill_it   = .true.
	end if

	if (line_it .and. fill_it) then
	   poly_opt = '/LINE/FILL'
	else if (line_it .and. .not. fill_it) then
	   poly_opt = '/LINE'
	else
	   poly_opt = ' '
	end if

	if (overlay) sover = .true.
* set up the axes and title and load the data into PPLUS
	CALL POLYGON_SET_UP   ( memory,
     .				overlay,
     .				transpz,
     .				coord_ax,
     .				no_labels,
     .				no_range,
     .				only_val,
     .				symbol,
     .				color,
     .				use_line,
     .				is_logh, 
     .				is_logv,
     .				is_mr,
     .				is_cx,
     .				num_uvars_in_cmnd,
     .				memory( 1, ax1_start ),
     .				memory( 1, ax2_start ),
     .				status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* PPLUS blows up setting auto contour levels if all data have same value
	IF ( no_range .AND. qual_given( slash_shad_lev_same ) .EQ. 0 )
     .          CALL PPLCMD ( from, line, 0, 
     .          'LEV () ('//TM_FMT(only_val, 3, 12, slen)//')' , 1, 1 )

* default key

	do_key =  (qual_given(slash_polygon_key)  .GT. 0)
     .          .OR. (fill_it
     .		     .AND. .NOT.overlay
     .		     .AND. num_uvars_in_cmnd.GT.2)
	IF (qual_given(slash_polygon_nokey) .gt. 0) do_key = .FALSE.
	IF ( do_key ) THEN
	   CALL PPLCMD ( from, line, 0, 'SHAKEY 1,1', 1, 1 )

* ... /KEY=continouus ?
              loc = qual_given(slash_polygon_key )
              IF ( loc .GT. 0) THEN
	         CALL EQUAL_STR_LC( cmnd_buff(qual_start(loc):qual_end(loc)),
     .                        buff, status )
                 IF ( status .NE. ferr_ok ) GOTO 5000
	         IF ( buff(1:3) .EQ. 'con' ) THEN
                    cont_key = .TRUE.
                    CALL PPLCMD (from, line, 0, 'S_KEY 1', 1, 1 )
                 ENDIF
              ENDIF

	ELSE
	   CALL PPLCMD ( from, line, 0, 'SHAKEY 0,1', 1, 1 )
	ENDIF

* protect colors already on-screen before a color overlay
	IF ( overlay ) CALL PPL_SHASET( 'PROTECT' )

* save old and reset to new color spectrum if requested
	IF ( spectrum ) THEN
	   CALL EQUAL_STR_LC( cmnd_buff(qual_start(sp):qual_end(sp)),
     .             fname, status )
	   IF ( status .NE. ferr_ok ) RETURN
	   IF ( fname .EQ. ' ' ) fname = 'default'
	   CALL PPL_SHASET( 'SAVE' )
	   CALL PPL_SHASET( 'SPECTRUM='//fname(1:TM_LENSTR1(fname)) )
	ENDIF

* save old and set to new pattern if requested (a la spectrum)
	IF ( pattern ) THEN
	   CALL EQUAL_STR_LC( 
     .             cmnd_buff(qual_start(patt):qual_end(patt)),
     .             fname, status )
	   IF ( status .NE. ferr_ok ) RETURN
	   IF ( fname .EQ. ' ' ) fname = 'default'
	   CALL PPL_PATSET( 'SAVE' )
	   CALL PPL_PATSET( 'PATTERN='//fname(1:TM_LENSTR1(fname)) )
	ENDIF

* turn off axes? 
	IF ( noaxes ) THEN 
           DO 300 i = 1, 4
              iax(i) = 0
 300       ENDDO
        ENDIF

* turn on (one or more) axes? (7/11/01)

* ... PLOT/AXES=(TOP,BOTTOM,LEFT,RIGHT)
	IF ( axes ) THEN
           loc = qual_given( slash_polygon_axes )
	   CALL EQUAL_STR_LC( cmnd_buff(qual_start(loc):qual_end(loc)),
     .                        buff, status )
           IF ( status .NE. ferr_ok ) GOTO 5000

	   IF ( buff .NE. ' ' ) THEN

* Get arguments, if any.  Otherwise turn on all axes
              slen = TM_LENSTR1(buff)
              s1 = 1
              s2 = slen
              IF (buff(1:1) .EQ. '(' )THEN
                  s1 = 2
                  s2 = slen-1
                  IF ( buff(slen:slen) .NE. ')' ) GOTO 5440
              ENDIF

              nparm = 4
              CALL parse_comma_list ( buff, s1, s2, nparm, 
     .             num_it, it_start, it_end, status)
              IF (status .EQ. ferr_ok .AND. it_start(1) .GT. it_end(1)) THEN	! /AXES=()
                 DO 400 i = 1, 4
                    iax(i) = 0
 400             ENDDO
              ELSE IF (status .NE. ferr_ok .OR. num_it .LT. nparm) THEN
                 GOTO 5440
              ELSE
                 DO 500 i = 1, nparm
                    READ (buff(it_start(i):it_end(i)), *) iax(i)
                    IF (iax(i) .LT. 0  .OR.  iax(i) .GT. 1) GOTO 5450
 500             CONTINUE
              ENDIF
           ELSE		! turn on all axes
	      DO 600 i = 1, 4
		 iax(i) = 1
 600	      CONTINUE
           ENDIF
              
	ENDIF

*  Turn off or on axes, including time axes.
        IF (axes .OR. noaxes) CALL ax_on_off (iax)

* /set_up only requested ?
	IF ( set_up ) GOTO 1000

* polygon it 
	   IF ( overlay ) THEN
	      CALL PPLCMD ( ' ', ' ', 0,
     .          'FILLPOL/OVER/'//ppl_wait//poly_opt//' ' , 1, 1 )
	   ELSE
	      CALL PPLCMD ( ' ', ' ', 0,
     .		'FILLPOL/'//ppl_wait//poly_opt//' ', 1, 1 )
	   ENDIF

* protect the new colors and restore the original color spectrum
           IF ( spectrum ) THEN
              CALL PPL_SHASET( 'PROTECT' )
              CALL PPL_SHASET( 'SPECTRUM' )
           ENDIF

* restore the original pattern
           IF ( pattern ) THEN
              CALL PPL_PATSET( 'PATTERN' )
           ENDIF

* restore axes? (5/29/97) revised 3/02
	IF ( axes .or. noaxes ) CALL PPL_AXES_RESTORE

* restore default (non-continuous) key style?
        IF (cont_key ) CALL PPLCMD (from, line, 0, 'S_KEY 0', 1, 1 )

* interrupt occurred during plot ?
	IF ( ppl_interrupted ) GOTO 1000

* capture the bit map ?
	IF ( qual_given( slash_plot_frame ) .GT. 0 ) THEN
	   CALL SAVE_FRAME( frame_file, "HDF", status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	ENDIF

*  Save log axis state

	IF (overlay) THEN
	   is_logh = h_logaxis
	   is_logv = v_logaxis
	ELSE
	   h_logaxis= is_logh
	   v_logaxis = is_logv 
	ENDIF

* release the memory used for axis buffers
 1000	CALL FREE_MEMORY( ax2_start, ax2_blks )
 	CALL FREE_MEMORY( ax1_start, ax2_blks )
	CALL UNPROTECT_CMND_DATA			! 1/94
 5100	RETURN

* error exits
 5000	GOTO 1000
 5200	CALL ERRMSG( ferr_out_of_range, status, 
     .		cmnd_buff(qual_start(loc):qual_end(loc))//pCR//
     .		'Legal range is 1 to '//LEFINT(limit,slen), *1000 )
 5400	CALL ERRMSG( ferr_out_of_range, status, 
     .		cmnd_buff(qual_start(loc):qual_end(loc))//pCR//
     .		'Legal range is 1 to 3', *1000 )
 5440	CALL ERRMSG( ferr_syntax, status, 
     .		cmnd_buff(qual_start(loc):qual_end(loc))//pCR//
     .'AXES[=(TOP,BOTTOM,LEFT,RIGHT)] takes 4 arguments or no argument', 
     .		*1000 )
 5450	CALL ERRMSG( ferr_out_of_range, status, 
     .		cmnd_buff(qual_start(loc):qual_end(loc))//pCR//
     .		'AXES arguments must be 0 or 1', *1000 )
	END
