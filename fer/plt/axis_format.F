	SUBROUTINE AXIS_FORMAT( x_or_y, idim, grid, lo_val, hi_val,
     .				del_val, curvilinear, cx_curv,
     .				its_calendar, calendar_both, 
     .				tstyle, no_labels, imposed_lab, tscale, toff )

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
* V500: 6/99 *sh* - save time axis infor to be used by PLOT/VS overlays
* V510 *sh* 3/00 - added no_labels argument so /NOLABELS avoided xlab & ylab
* V533 *sh* 7/01 - avoid using risc_buff -- conflict with PUT_LABEL
* V580 *acm* 12/04 Fix bug 1129; formatting axis for small lon/lat ranges
* V602 *acm*  3/07 Fix bug 1498: do not computes n_res but leave it hard-wired
*                to 15; else the axis labels may not match data values
* V62+ *acm* 6/09 Allow use_nice for more lon/lat axes; allows for long 
*                 strings in the degree-minute-seconds labeling
* V64+ *acm*12/09 Fix bugs (crash!)for plots with very small ranges in lon and lat
*                 See the benchmark test err64_very_small_latlon.jnl, based on a 
*                 report by Jean Newman.
*       *acm* 3/12 6D Ferret: Add E and F dimensions 
*                  A time axis may be a T axis or F axis, so idim is 
*                  now an arg to TSTEP_TO_SECS, TSTEP_TO_DATE
* PyFr  *kms* 7/13 Use hxlab and hylab from PPL with PUT_LABEL for axis titles
* V694  *acm* 1/15 Changes to allow two time axes on plot (F-T)
* V695+ *sh*  3/15 Added argument "imposed_lab" to disambiguate F and T axes
* V695  *acm* 9/15 Do not plot the labell TIME - on vertical axes with HH:MM
*		   labeling; it interferes with the time/date labels
* V7    *acm* 6/16 Ticket 2434: command qualifiers /DEGMINSEC= /HALFSPAC
*                  DEGMINSEC=1 for Deg-Min axis labels, =2 for Deg-Min-Sec. Default = 1
*                  /HALFSPAC adds a half space between the degree sign and E/W or N/S
        INCLUDE 'LABCOM.INC'    ! hxlab, hylab
* needed to position the imposed labels
	include 'axis_inc.decl'		! axis lengths
        include 'AXIS.INC'              ! axis lengths
* special equivalence needed only with risc compiler: bug fix
	include 'xrisc.cmn'     ! for f77 bug
        CHARACTER rbuff(4)*14
        EQUIVALENCE ( rbuff, risc_buff )

* calling argument declarations:
	LOGICAL		its_calendar, calendar_both, curvilinear, no_labels
	INTEGER		idim, grid, cx_curv
	CHARACTER	x_or_y*1, tstyle*3
	CHARACTER*(*)	imposed_lab
	REAL*8		lo_val, hi_val, del_val, tscale, toff

* parameter definitions:
	INTEGER		pyear_only
	PARAMETER     ( pyear_only = 1 )

* internal variable declarations:
	LOGICAL		GEOG_LABEL, GEOG_VAR,ITSA_TRUEMONTH_AXIS,
     .			use_nice, special
	INTEGER		TM_LENSTR1, TM_LENSTR, dlen, places, ndec, dummy
	CHARACTER	AX_TITLE*32, WHOI_DATE*14, VAR_TITLE*128,
     .			INTERNAL_WHOI_DATE*14,
     .			fmt*4, buff*200, tsetting*2
	REAL		res_del, dt_min, time_unit, frac, labelht
	REAL*8		TSTEP_TO_SECS
	INTEGER		n_res
	REAL*4		x1, x2, uminr, umaxr, dint  ! for call to RANGE

	PARAMETER 	(n_res = 15)

* initiailize
	tstyle = ' '
	tscale = 1.0
	toff = 0.0

* decimal places needed for axis:
* ... sufficient digits to resolve the tic interval differences and
*     still display the largest axis labels
*  res_del - labels will be accurate enough to resolve this delta value

	res_del = ABS( hi_val - lo_val ) / n_res
	CALL GET_PREC_DIGITS( hi_val, lo_val, res_del, places, ndec )

* format to use
* frac - Will time precision be accurate enough to resolve this delta value?
	use_nice = places .LE. 7
	IF (idim.EQ.t_dim .OR.  idim.EQ.f_dim) THEN
	   rescale_time = .FALSE.
  	   time_unit = line_tunit( grid_line(idim,grid) )
	   rescale_time = (time_unit.EQ.1 .AND. places.GT.7)
	ENDIF

	IF ( use_nice ) THEN
	   IF ( ndec .EQ. 0 ) THEN
	      fmt = 'I7'
	   ELSE
	      WRITE ( fmt, 3000 ) ndec
 3000	      FORMAT ('F8.',I1)
	   ENDIF
	ENDIF

* default format to use has no special formatting
	IF ( use_nice )  CALL PPLCMD ( from, line, 0,
     .				x_or_y//'FOR,('//fmt//')', 1, 1 )
	IF ( rescale_time ) THEN
	   fmt = 'I7'
	   CALL PPLCMD ( from, line, 0,
     .				x_or_y//'FOR,('//fmt//')', 1, 1 )
	ENDIF

******* special axis formats ********
	special = ((idim.EQ.t_dim .OR.  idim.EQ.f_dim) .AND. its_calendar)
     .       .OR. ((idim.NE.t_dim .AND. idim.NE.f_dim) .AND. 
     .             GEOG_LABEL( idim, grid ))
	IF (curvilinear) THEN
	   special = (idim.NE.t_dim .AND. idim.NE.f_dim) .AND. 
     .             GEOG_VAR( idim, cx_curv )
	ENDIF
	IF ( special ) THEN
	   IF (idim .EQ. x_dim .OR. idim.EQ. y_dim) 
     .        use_nice = places .LE. 8
     
	   IF ( use_nice ) THEN
	      IF ( ndec .EQ. 0 ) THEN
	         fmt = 'I7'
	      ELSE
	         WRITE ( fmt, 3000 ) ndec
	      ENDIF
	   ENDIF
	   IF (idim.EQ.t_dim .OR.  idim.EQ.f_dim) THEN
* labels formatted in calendar dates
* .. convert units of time axis from seconds to minutes
	      time_unit = line_tunit( grid_line(idim,grid) )
	      IF (ITSA_TRUEMONTH_AXIS(grid_line(idim,grid))) time_unit = un_convert(pun_day)  
	      dt_min = time_unit / 60.

* scale and off if time requires a lot of precision

	      IF (rescale_time) THEN
	         dt_min = 1
		 tscale = 1/60.
		 toff = -1. * lo_val
	      ENDIF

* ... TAXIS command
	      tsetting = 'ON'
	      IF (calendar_both) tsetting = 'BO'
	      IF ( x_or_y .EQ. 'Y' ) THEN
	         WRITE ( ppl_buff, 3010 ) dt_min, tsetting
	      ELSE
	         WRITE ( ppl_buff, 3012 ) dt_min, tsetting
	      ENDIF
	      CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )
 3010	      FORMAT ('TAXIS/YAXIS ',G13.5,',',A2)
 3012	      FORMAT ('TAXIS ',G13.5,',',A2)

* ... TIME command (time axis limits)
              rbuff(1) = WHOI_DATE(grid, idim, lo_val )      ! f77 bug
              rbuff(2) = WHOI_DATE(grid, idim, hi_val )
              rbuff(3) = WHOI_DATE(grid, idim, 1.0D0  )

	      IF (rescale_time) THEN 
	         rbuff(3) = rbuff(1)

	         x1 = (lo_val + toff) * tscale
	         x2 = (hi_val + toff) * tscale
	         CALL RANGE( x1, x2, 10, uminr, umaxr, dint )
	         WRITE ( ppl_buff, '(3(E14.7,1X))' ) x1, x2, dint
	         CALL PPLCMD ( from, line, 0, 
     .                          x_or_y//'AXIS '//ppl_buff , 1, 1 )
     	      ENDIF


	      WRITE ( ppl_buff, 3020 )	'W'//rbuff(1),
     .					'W'//rbuff(2),
     .					'W'//rbuff(3)
 3020	      FORMAT ('TIME ',A15,',',A15,',',A15)
	      CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )

* ... save info for PLOT/VS/OVER
	      saved_dt_min = dt_min
	      saved_t1_date =  INTERNAL_WHOI_DATE(grid, idim, 1.0D0 )

* ... This will be used in polygon overlays on a time axis
	      IF (rescale_time) saved_bb_date =  INTERNAL_WHOI_DATE(grid, idim, lo_val ) 

* ... The XAXIS_MIN, XAXIS_MAx symbols already set in axis_ends are not changed; used in LAS,
*     they are translated by LAS using the time axis info from the original dataset
*     and they have enough precision.

* ... TXTYPE and TXLINT commands (time axis style)
	      IF (no_labels) THEN
	         buff = ' '
	      ELSE
	         buff = imposed_lab
	      ENDIF
	      CALL TAXIS_STYLE(	x_or_y, TSTEP_TO_SECS( grid, idim, lo_val ),
     .				TSTEP_TO_SECS( grid, idim, hi_val ),
     .				tstyle, buff )

	   ELSEIF ( idim .EQ. x_dim .AND. use_nice) THEN
* longitude format for axis tic labling
	      ppl_buff = 
     .		x_or_y//'FOR,('//fmt(:TM_LENSTR1(fmt))//',''''LONE'''')' 
	      CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )

	      CALL LON_LAT_FMT (idim, x_or_y)

	   ELSEIF ( idim .EQ. y_dim  .AND. use_nice) THEN
* latitude format for axis tic labling
	      ppl_buff = 
     .		x_or_y//'FOR,('//fmt(:TM_LENSTR1(fmt))//',''''LAT'''')' 
	      CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )

	      CALL LON_LAT_FMT (idim, x_or_y)

	   ENDIF

	ENDIF

* lat/long or x/y axis label
	IF ( no_labels ) RETURN
        IF (curvilinear ) THEN
	   buff = VAR_TITLE( cx_curv)
	   dlen = TM_LENSTR1(buff)
	ELSE
	   IF (its_calendar.AND. TM_LENSTR(imposed_lab) .EQ.0) THEN ! skip label of TIME
	      buff = ' '
	      dlen = 1
	   ELSE
 	      buff = AX_TITLE( idim, grid, dlen )	! risc_buff as of 6/95
	   ENDIF
	ENDIF
	IF ( x_or_y .EQ. 'X' ) THEN
           labelht = hxlab
	   CALL PUT_LABEL( ppl_xlab, dummy, dummy, dummy,
     .                     labelht, buff(:dlen), dummy, dummy )
	ELSE
           labelht = hylab
	   CALL PUT_LABEL( ppl_ylab, dummy, dummy, dummy,
     .                     labelht, buff(:dlen), dummy, dummy )
	ENDIF

	RETURN
      END
