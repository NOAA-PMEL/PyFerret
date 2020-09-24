	SUBROUTINE RIBBON_PLOT_SET_UP (versus, nvars, addgaps, 
     .		dsg_as_traj, idir, thick, overlay, spectrum, status)

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
* programmer - Ansley Manke
* NOAA/PMEL, Seattle, WA - SDIG program
* 4/24/2018

* Set up for making ribbon-style plots, colored by a variable. Extracted from  
* xeq_plot, and with additions for plotting trajectoriesas ribbon plots.

* V74+ 4/18 *acm*  Automatically do a colored trajectory plot for the command 
*                  "PLOT var" if it's a DSG trajectory set. 
* V74+ 4/18 *acm*  Call RIBBON_PLOTKEY_SETUP to handle setting up the color key 
* V7.5+ *acm*  4/18 PLOT/VS plots from DSG data that are not native-trajectory plots
* V751 7/19 *acm*  DSG-enabled Ferret: change to call of CHECK_PLOT_TRAJ

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
	include 'parampl5_dat.decl'
	include 'PARAMPL5.DAT'
	include 'cont_inc.decl'
	include 'CONT.INC'

* calling argument declarations:
	LOGICAL versus, overlay, addgaps, spectrum
	INTEGER nvars, thick, status

* internal variable declarations:
	LOGICAL TM_HAS_STRING,  do_key, its_traj, dsg_as_traj
	INTEGER	MR_DIM_LEN, CX_DIM_LEN, CGRID_SIZE, TM_DSG_NFEATURES,
     .		TM_LENSTR1, TM_LENSTR, STR_SAME,
     .          nvs, loc, cx, grid, slen, nfeatures, idir
	REAL	val
	CHARACTER buff*128, TM_INQ_PATH*2048, sfile*2048

* if ribbon_plot then:
	
	cx = is_cx(1)
	grid = cx_grid(cx)

	IF (versus) THEN
	   nvs = 3
	   IF (addgaps) nvs = 4

* If it's a DSG trajectory dataset, can make the trajectories colored by var
* with just a "PLOT var".

	   IF (grid_is_dsg .AND. versus) THEN
	      IF (ribbon_plot) nvs = 1

	      cx = is_cx(1)
	      grid = cx_grid(cx)
	      nfeatures = TM_DSG_NFEATURES( grid )

	   ENDIF
	  
	   IF (nvars .LT. nvs) GOTO 5100
	   IF (nvars .GT. nvs .AND. .NOT.grid_is_dsg) THEN
	      buff = 'PLOT/RIBBON/VS uses just 3 variables. Ignoring extra variables.'
	      IF (addgaps) buff = 'PLOT/RIBBON/VS/GAPLOC uses 4 variables. Ignoring extra variables.'
	      CALL WARN (buff)
	   ENDIF
	ELSE
	   IF (nvars .LT. 2) GOTO 5100
	   IF (nvars .GT. 2) CALL WARN (
     .	  'PLOT/RIBBON uses just 2 variables. Did you mean PLOT/RIBBON/VS?')
	ENDIF

* If a color was given, that's not valid for Ribbon plots.
	loc = qual_given( slash_plot_color )
	IF ( loc .GT. 0  ) GOTO 5200

* If a thickness is given, set pen number to the thickness, pen=1,2,or 3

	loc = qual_given( slash_plot_thick )
	IF ( loc .GT. 0  ) THEN

	  CALL EQUAL_VAL( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			  val, status )

	  IF ( val .NE. unspecified_val4 ) THEN
	     thick = NINT( val )	! user-specified line style
	  ELSE
	     thick = 2  ! default if no value is specified
	  ENDIF
	ELSE

* When plotting dsg as trajectory or points in XY (e.g. timeseriesProfile with /ALONG=xy)
* use thick unless they've set something else.

	   IF (dsg_as_traj) thick = 2
	ENDIF

* A "gaps" variable specifies the end of line segments to draw for PLOT/VS/RIBBON.
* If it's DSG we handle all that from the grid defs; interferes with /MASKFEATURE

	IF (addgaps .AND. grid_is_dsg ) GOTO 5600
	IF (addgaps .AND. .NOT.versus ) GOTO 5400

* Style: fast for one-color per point, default is to interpolate
* colors between points

* For trajectoryProfile as Trajectory, connect the profile locations with
* a line, using "fast" style

	loc = qual_given( slash_plot_fast  )
	IF ( loc.GT.0 .OR. (dsg_as_traj .AND. idir.EQ.x_dim) ) THEN
	  CALL PPLCMD ( from, line, 0, 'RIBFAST 1', 1, 1 )
	ELSE
	  CALL PPLCMD ( from, line, 0, 'RIBFAST 0', 1, 1 )
	ENDIF

* Missing: how to represent missing color-values. Default is
* thin black line. Can request /missing=dash, /missing=blank, 
* /missing= color-palette (e.g. gray_light). The color palette 
* will typically be a one-color plot, or if a multicolor palette 
* is given its first color will be used.

* (call this BEFORE setting the main color palette for ribbon plot; so that
* the setting for palette type: percent, by_level, or by_value is set for the
* specified palette.)

	loc = qual_given( slash_plot_missing  )
	IF ( loc.GT.0 ) THEN
	   CALL EQUAL_STR_LC( cmnd_buff(qual_start(loc):qual_end(loc)),
     .	     buff, status )
	   IF ( status .NE. ferr_ok ) RETURN

	    IF (STR_SAME (buff(1:4), 'blan') .EQ. 0) THEN
	     CALL PPLCMD ( from, line, 0, 'RIBMISS BLANK', 1, 1 )
	   ELSEIF ( buff .EQ. ' ' ) THEN
	      CALL PPLCMD ( from, line, 0, 'RIBMISS DEFAULT', 1, 1 )
	   ELSE
	      CALL PPLCMD ( from, line, 0, 'RIBMISS SPECTRUM='//buff, 1, 1 )
	   ENDIF
	ELSE
	   CALL PPLCMD ( from, line, 0, 'RIBMISS DEFAULT', 1, 1 )
	ENDIF

* ... PLOT/PALETTE=  for multi-colored "ribbon" line 

* save old and reset to new color spectrum if requested

	loc = qual_given( slash_plot_spectrum  )
	IF ( loc.GT.0 ) THEN
	   CALL EQUAL_STR_LC( cmnd_buff(qual_start(loc):qual_end(loc)),
     .	     buff, status )
	   IF ( status .NE. ferr_ok ) RETURN
	   IF ( buff .EQ. ' ' ) buff = 'default'
	ENDIF

	IF (loc.GT.0 ) THEN
	   slen = TM_LENSTR1(buff)
	   sfile = TM_INQ_PATH (buff(:slen),'FER_PALETTE','.spk',
     .        	.false.,status)
	   IF (status .NE. ferr_ok) GOTO 5500 

	   CALL PPL_SHASET( 'SAVE' )
	   CALL PPL_SHASET( 'SPECTRUM='//buff(1:TM_LENSTR1(buff)) )
	   spectrum = .TRUE.

	ENDIF
	      
* default color key
	CALL RIBBON_PLOTKEY_SETUP (overlay, status)
	IF (status .NE. ferr_ok) GOTO 5000

* successful completion
	status = ferr_ok
	RETURN

* error exits
 5000	RETURN
     
 5100	IF (versus .AND. addgaps .AND. ribbon_plot) THEN
           CALL ERRMSG( ferr_syntax, status, 
     .	   'PLOT/VS/RIBBON/GAPLOC plot needs 4 arguments', *5000 )
        ELSEIF (versus .AND. addgaps) THEN
           CALL ERRMSG( ferr_syntax, status,
     .     'PLOT/VS/GAPLOC implemented only for RIBBON plots', *5000)
        ELSEIF (versus) THEN
           CALL ERRMSG(ferr_syntax, status,
     .     'PLOT/VS/RIBBON plot needs 3 arguments', *5000 )
	ELSE
           CALL ERRMSG( ferr_syntax, status, 
     .	   'RIBBON plot needs 2 arguments', *5000 )
	ENDIF
 5200	CALL ERRMSG( ferr_syntax, status, 
     .		cmnd_buff(qual_start(loc):qual_end(loc))//pCR//
     .	  '/COLOR not valid for PLOT/LEVELS ribbon plot', *5000 )
 5400	CALL ERRMSG( ferr_syntax, status, 
     .	'PLOT/GAPLOC applies only to PLOT/VS/RIBBON', *5000 )
 5500   CALL ERRMSG (ferr_file_not_found, status, 
     .          'unable to open "' //
     .	buff(:slen)//'.spk" to set ribbon-plot colors.', *5000)

 5600	CALL ERRMSG( ferr_syntax, status, 
     .		cmnd_buff(qual_start(loc):qual_end(loc))//pCR//
     .	  '/GAPLOC invalid for Discrete Sampling Geometries Trajectory dataset', *5000 )
	   CALL WARN ('/GAPLOC invalid for Discrete Sampling Geometries Trajectory file')


	END
