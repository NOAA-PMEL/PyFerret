	SUBROUTINE AXIS_FORMAT( x_or_y, idim, grid, lo_val, hi_val,
     .				curvilinear, cx_curv,
     .				its_calendar, tstyle )

* set up PPLUS to properly label the axis tics and titles

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 6/23/87
* revision 1.0 - 2/25/88 - added grid to call; allowed non-geographic axes
* revision 2.0 - 4/12/88 - added time axis set-ups and decimal place calcs.
*			   and units on axis labels (via AX_TITLE)
*			   x/y axis labels through PUT_LABEL call
* revision 2.1 -  5/7/88 - increased format sizes
* revision 2.2 -  9/8/88 - added position relative flags to PUT_LABEL call
* revision 2.3 - 9/21/88 - suppress year on time axis for climatological sets
*			 - added control over time axis tics (TXLINT)
*			 - fixed PPLUS (TAXIS3) bug permitting "day"-style
*			   time axes (TXTYPE) - return "tstyle"
* revision 2.31 - 1/9/89 - doc chg: climatological axis in TAXIS_STYLE
* V200:  10/13/89 - corrected FORMAT 3010 to allow dt_min=1/60
* 	   1/4/90 - modified PUT_LABEL call
* Unix/RISC port - 4/91 *sh*: RISC f77 can't handle CHAR fcn in WRITE list
* V313: *sh* AIX port.  Replace "FOR ()" with "FOR,()" so AIX space-squeezing
*		filter doesn't corrupt this routine
* V400: 6/23/95 - *sh* use risc_buff instead of ppl_buff when calling PUT_LABEL
*		because PUT_LABEL uses ppl_buff internally 
* V420: 11/27/95 - 4-digit PLOT+ YEARS!!  also, new arg its_calendar
*	11/28/95 - changes to TAXIS_STYLE call for Y-oriented calendar axes
* 12/96 *kob* - Linux Port - had to move variable assignments off of
*               declaration line.
* V450: 3/97 - support for curvilinear coordinate plots
*		added args curvilinear and cx_curv

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
	include 'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
	include 'tmap_format/xdset_info.cmn_text'
	external xdset_info_data
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/plot_setup.parm'
	include 'ferret_cmn/xplot_setup.cmn'
* special equivalence needed only with risc compiler: bug fix
	include 'ferret_cmn/xrisc.cmn'     ! for f77 bug
        CHARACTER rbuff(4)*14
        EQUIVALENCE ( rbuff, risc_buff )
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:PLOT_SETUP.PARM'
	INCLUDE 'FERRET_CMN:XPLOT_SETUP.CMN'
#endif

* calling argument declarations:
	LOGICAL		its_calendar, curvilinear
	INTEGER		idim, grid, cx_curv
	CHARACTER	x_or_y*1, tstyle*3
	REAL*8		lo_val, hi_val

* parameter definitions:
	INTEGER		pyear_only
	PARAMETER     ( pyear_only = 1 )

* internal variable declarations:
	LOGICAL		GEOG_LABEL, GEOG_VAR, use_nice, special
	INTEGER		TM_LENSTR1, dlen, places, ndec, dummy, itax
	CHARACTER	AX_TITLE*32, WHOI_DATE*14, VAR_TITLE*128,
     .			fmt*4, date*4
	REAL		res_del, dt_min, time_unit
	REAL*8		TSTEP_TO_SECS
	INTEGER		n_res
	PARAMETER 	(n_res = 15)

* initiailize
	tstyle = ' '

* decimal places needed for axis:
* ... sufficient digits to resolve the tic interval differences and
*     still display the largest axis labels
*  res_del	- labels will be accurate enough to resolve this delta value

	res_del = ABS( hi_val - lo_val ) / n_res
	CALL GET_PREC_DIGITS( hi_val, lo_val, res_del, places, ndec )

* format to use
	use_nice = places .LE. 6
	IF ( use_nice ) THEN
	   IF ( ndec .EQ. 0 ) THEN
	      fmt = 'I7'
	   ELSE
	      WRITE ( fmt, 3000 ) ndec
 3000	      FORMAT ('F8.',I1)
	   ENDIF
	ENDIF

* default format to use has no special formatting
	IF ( use_nice ) THEN
	      CALL PPLCMD ( from, line, 0,
     .				x_or_y//'FOR,('//fmt//')', 1, 1 )
	ENDIF

******* special axis formats ********
	special = idim.EQ.t_dim .AND. its_calendar
     .       .OR. idim.NE.t_dim .AND. GEOG_LABEL( idim, grid )
	IF (curvilinear) THEN
	   special = idim.NE.t_dim .AND. GEOG_VAR( idim, cx_curv )
	ENDIF
	IF ( special ) THEN
	   IF ( idim .EQ. t_dim ) THEN
* labels formatted in calendar dates
* .. convert units of time axis from seconds to minutes
	      time_unit = line_tunit( grid_line(t_dim,grid) )
	      dt_min = time_unit / 60.

* ... TAXIS command
	      IF ( x_or_y .EQ. 'Y' ) THEN
	         WRITE ( ppl_buff, 3010 ) dt_min
	      ELSE
	         WRITE ( ppl_buff, 3012 ) dt_min
	      ENDIF
	      CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )
 3010	      FORMAT ('TAXIS/YAXIS ',G13.5,',ON')
 3012	      FORMAT ('TAXIS ',G13.5,',ON')

* ... TIME command (time axis limits)
#ifdef unix
              rbuff(1) = WHOI_DATE(grid, lo_val )      ! f77 bug
              rbuff(2) = WHOI_DATE(grid, hi_val )
              rbuff(3) = WHOI_DATE(grid, 1.0D0  )
	      WRITE ( ppl_buff, 3020 )	'W'//rbuff(1),
     .					'W'//rbuff(2),
     .					'W'//rbuff(3)
#else
	      WRITE ( ppl_buff, 3020 )	'W'//WHOI_DATE(grid, lo_val ),
     .					'W'//WHOI_DATE(grid, hi_val ),
     .					'W'//WHOI_DATE(grid, 1.0D0  )
#endif
 3020	      FORMAT ('TIME ',A15,',',A15,',',A15)
	      CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )

* ... TXTYPE and TXLINT commands (time axis style)
	      CALL TAXIS_STYLE(	x_or_y, TSTEP_TO_SECS( grid, lo_val ),
     .				TSTEP_TO_SECS( grid, hi_val ), tstyle )

	   ELSEIF ( idim .EQ. x_dim .AND. use_nice ) THEN
* longitude format for axis tic labling
	      ppl_buff = 
     .		x_or_y//'FOR,('//fmt(:TM_LENSTR1(fmt))//',''''LONE'''')' 
	      CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )
	   ELSEIF ( idim .EQ. y_dim .AND. use_nice ) THEN
* latitude format for axis tic labling
	      ppl_buff = 
     .		x_or_y//'FOR,('//fmt(:TM_LENSTR1(fmt))//',''''LAT'''')' 
	      CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )
	   ENDIF
	ENDIF

* lat/long or x/y axis label
	IF ( curvilinear ) THEN
	   risc_buff = VAR_TITLE( cx_curv)
	   dlen = TM_LENSTR1(risc_buff)
	ELSE
	   risc_buff = AX_TITLE( idim, grid, dlen )	! risc_buff as of 6/95
	ENDIF
	IF ( x_or_y .EQ. 'X' ) THEN
	   CALL PUT_LABEL( ppl_xlab, dummy, dummy, dummy,
     .			   0.12, risc_buff(:dlen), dummy, dummy )
	ELSE
	   CALL PUT_LABEL( ppl_ylab, dummy, dummy, dummy,
     .			   0.12, risc_buff(:dlen), dummy, dummy )
	ENDIF
	RETURN
      END
