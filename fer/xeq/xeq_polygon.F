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

#ifdef unix
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
        include 'ppldat_inc.decl'
        include 'PPLDAT.INC'
        include 'parampl5_dat.decl'
        include 'PARAMPL5.DAT'
        include 'lines_inc.decl'
        include 'LINES.INC'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:PLOT_SETUP.PARM'
	INCLUDE 'FERRET_CMN:SLASH.PARM'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XPLOT_STATE.CMN'
	INCLUDE 'FERRET_CMN:XPLOT_SETUP.CMN'
        INCLUDE 'tmap_pplv11inc:tmap_SWITCH.INC'
        INCLUDE 'TMAP_PPLV11INC:DATA_INC.DECL'
        INCLUDE 'PPLV11INC:DATA.INC'
        INCLUDE 'TMAP_PPLV11INC:PPLDAT_INC.DECL'
        INCLUDE 'PPLV11INC:PPLDAT.INC'
        INCLUDE 'TMAP_PPLV11INC:PARAMPL5_DAT.DECL'
        INCLUDE 'PPLV11INC:PARAMPL5.DAT'
        INCLUDE 'TMAP_PPLV11INC:LINES_INC.DECL'
        INCLUDE 'PPLV11INC:LINES.INC'
#endif
	
* calling argument declarations:
	REAL	memory( mem_blk_size, max_mem_blks )

* internal variable declarations:
        LOGICAL         no_range, do_key, spectrum, pattern
	LOGICAL		overlay, transpz, no_labels, set_up, noaxes
	LOGICAL         line_it, fill_it
	INTEGER         sp, patt, TM_LENSTR1
	INTEGER		symbol, line_type, status, mr, cx, ipl,
     .			loc, limit, slen, coord_ax,
     .			ax1_blks, ax1_start,
     .			ax2_blks, ax2_start
	REAL		val, only_val
	CHARACTER	LEFINT*4, TM_FMT*12, fname*128, poly_opt*10

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
	noaxes    = qual_given( slash_noaxes         ).GT.0
	coord_ax  = qual_given( slash_plot_vs	     )		! integer
	symbol	  = qual_off	! or ??unspecified_int4??
	line_type = unspecified_int4
        sp        = qual_given( slash_cont_spectrum  )

        spectrum  = sp .GT. 0 
        patt      = qual_given( slash_shad_pattern   )
        pattern   = patt .GT. 0

	line_it   = qual_given(slash_polygon_line) .gt. 0 

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

	IF ( .NOT.overlay ) CALL DISP_INIT(no_labels,is_cx,num_uvars_in_cmnd)
	CALL DISP_PREP( status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* decode qualifiers
* ... POLYGON/SYMBOL=n
	loc = qual_given( slash_plot_symb )
	IF ( loc .GT. 0  ) THEN
	   CALL EQUAL_VAL( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			      val, status )	! syntax check
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   IF ( val .NE. unspecified_val4 ) THEN
	      limit = max_sym 
	      IF ( val.LE. 0.0 .OR. val.GT.limit ) GOTO 5200
	      symbol = NINT( val )	! user-specified symbol
	   ELSE
	      symbol = qual_on		! auto selection of symbols
	   ENDIF
	   line_type = qual_off		! default to no line w/ symbol	      
	ENDIF
* ... POLYGON/LINE_STYLE=n
	loc = qual_given( slash_polygon_line )
	IF ( loc .GT. 0  ) THEN
	   CALL EQUAL_VAL( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			      val, status )	! syntax check
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   IF ( val .NE. unspecified_val4 ) THEN
	      limit = num_fancy_lines 			!was num_plain 1/94
	      IF ( val.LE. 0.0 .OR. val.GT.limit ) GOTO 5200
	      line_type = NINT( val )	! user-specified line style
	   ELSE
	      line_type = qual_on		! auto selection of line style
	   ENDIF
	ENDIF

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
     .				line_type,
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

* turn off axes? (5/29/97)
	IF ( noaxes ) CALL PPL_AXES_OFF(set_up)

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

* restore axes? (5/29/97)
	IF ( noaxes ) CALL PPL_AXES_RESTORE

* interrupt occurred during plot ?
	IF ( ppl_interrupted ) GOTO 1000

* capture the bit map ?
	IF ( qual_given( slash_plot_frame ) .GT. 0 ) THEN
	   CALL SAVE_FRAME( frame_file, "HDF", status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
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
	END
