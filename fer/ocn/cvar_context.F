	SUBROUTINE CVAR_CONTEXT( def_cx, cx, com, k_eq1, status)

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
* cvar table component pointer, com, with the default context, def_cx

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* V200:  7/25/89 - based on INST_IJKL_CONTEXT
* 	 2/27/90 - incorporate CAIR as a regrid on the main stack
* 		 - and allow unspecified limits to pass through
*Linux Port 1/97 *kob* - Added a preprocessor include for tmap_dset.parm
*			  because it needed a preprocessor.
#ifdef unix
	include	'tmap_format/tmap_dims.parm'
#	include	"tmap_format/tmap_dset.parm"
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/errmsg.parm'
	include	'ferret_cmn/interp_stack.parm'
	include	'ferret_cmn/instance_table.parm'
	include	'ferret_cmn/xcontext.cmn'
	include	'ferret_cmn/xvariables.cmn'
	include	'ferret_cmn/xcalc_vars.cmn'
	include	'ferret_cmn/xtext_info.cmn'
	include	'ferret_cmn/xdset_parms.cmn'
#else
	INCLUDE	'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
	INCLUDE	'FERRET_CMN:INTERP_STACK.PARM'
	INCLUDE	'FERRET_CMN:INSTANCE_TABLE.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XCALC_VARS.CMN'
	INCLUDE	'FERRET_CMN:XTEXT_INFO.CMN'
	INCLUDE	'FERRET_CMN:XDSET_PARMS.CMN'
#endif

* calling argument declarations:
	LOGICAL k_eq1
	INTEGER	def_cx, cx, com, status

* internal variable declarations:
	LOGICAL	HOURLY_DATA, special_var, re_flesh
	INTEGER KNOWN_GRID, CGRID_AXIS, FVAR_NUMBER, ISUBSCRIPT,
     .		l_step, var, l_lo, idim, def_ax, com_ax,
     .		grid, dset, cat, fvar, cairt, cairt_dset
	REAL*8	TM_WORLD, tptinc, t, hours, tmod, t0cairt,
     .          tclim0, tfrombc
	CHARACTER*20	date

* internal parameters:
	REAL*8		secsperhr, hrsperyr
	PARAMETER     ( secsperhr = 3600.D0,
     .			hrsperyr  = 365.D0*24.D0 )

* initialize with context of variable to be computed
	CALL TRANSFER_CONTEXT( def_cx, cx )

* unravel the variable internal number coding
* (this is the fixed var # - 1=TEMP, 2=SALT, etc.)
	var = cv_variable( com )
	special_var = var .GT. pcv_spec_flg
	IF ( special_var ) var = var - pcv_spec_flg

* determine the data set
	IF ( special_var .AND. var .EQ. pairt ) THEN
	   dset = cx_data_set( def_cx )
	   cairt      = dp_clim_air_sst( dset )		! cairt needed for calcs
	   cairt_dset = dp_airt_dset( cairt )		! dset containing cairt
	   IF ( cairt_dset .EQ. unspecified_int4 ) THEN
	      GOTO 5100					! needed dset not SET
	   ELSE
	      dset = cairt_dset
	   ENDIF
	ELSE
	   dset = cx_data_set( cx )
	ENDIF
	cx_data_set( cx ) = dset

* find the variable number as stored in the data set, if relevant
	IF ( cv_category( com ) .EQ. cat_calc_var ) THEN
	   fvar = var
	ELSE
	   fvar = FVAR_NUMBER( dset, fvar_name_code(var) )
	   IF ( fvar .EQ. var_name_nonexist ) GOTO 5100
	ENDIF
	cx_variable( cx ) = fvar

* determine the variable category and defining grid for the data
	cat = cv_category( com )
	cx_category( cx ) = cat
	grid = KNOWN_GRID( dset, cat, fvar )
	cx_grid( cx ) = grid

* SPACIAL AXES: X,Y,Z
* component may be wider (or narrower) than result
	DO 100 idim = 1, 3
	   def_ax = CGRID_AXIS(idim,def_cx)
	   com_ax = CGRID_AXIS(idim,cx)
	   re_flesh = def_ax .NE. com_ax
	   IF ( def_ax.EQ.mnormal .OR. com_ax.EQ.mnormal ) THEN
	      IF ( def_ax .EQ. com_ax ) GOTO 100
	      re_flesh = .TRUE.
	      IF ( def_ax .EQ. mnormal ) THEN
* ... 3D component of 2D var: e.g. TEMP as comp. of QEVA ... needs SST
	         cx_lo_ss(cx,idim) = 1
	         cx_hi_ss(cx,idim) = 1
	      ENDIF
* ... for 2D component of 3D var FLESH_OUT_AXIS will set limits unspecified
* ... e.g. QFLX as comp. of QDFZ
	   ELSEIF ( cx_lo_ss(cx,idim) .EQ. unspecified_int4 ) THEN
* ... allow unspecified regions to pass through still unspecified 2/90
	      re_flesh = .FALSE.
	   ELSE
	      IF ( cv_lo_ds( com,idim ) .NE. 0 ) THEN
	         cx_lo_ss(cx,idim) = cx_lo_ss(cx,idim) + cv_lo_ds( com,idim )
	         re_flesh = .TRUE.
	      ENDIF
	      IF ( cv_hi_ds( com,idim ) .NE. 0 ) THEN
	         cx_hi_ss(cx,idim) = cx_hi_ss(cx,idim) + cv_hi_ds( com,idim )
	         re_flesh = .TRUE.
	      ENDIF
	      IF ( idim.EQ.3 .AND. k_eq1 ) THEN
	         IF ( cx_lo_ss(cx,idim) .NE. 1 ) THEN
	            cx_lo_ss(cx,idim) = 1
	            re_flesh = .TRUE.
	         ENDIF
	      ENDIF
	   ENDIF
	   IF ( re_flesh ) THEN
	      cx_by_ss( idim,cx ) = .TRUE.
	      CALL CONFINE_AXIS( idim, cx, status )
	      IF ( status .NE. ferr_ok ) RETURN
	      CALL FLESH_OUT_AXIS( idim, cx, status )
	      IF ( status .NE. ferr_ok ) RETURN
	   ENDIF
 100	CONTINUE

* TIME AXIS:
* remark on "hourly_needed" below:
* the time axes of the data sets are not in general the same as the underlying
* time axis of the model (model integration is typically hourly, whereas data
* sets will normally be 24 hour or 73 hour snapshots).  Some calculations, such
* as d/dt(U) simply use the best data available - whatever is in the data set.
* Other calculations require the exact 1 hour data to be exact.  If that data
* is not available then the current time step (dt=0) is used as the best est.

* get subscript for time step
	re_flesh = .FALSE.
	IF ( special_var .AND. var .EQ. pairt ) THEN
* ... climatological air temp is just a variable in another file that
*     needs to be regridded from its monthly axis to the time axis of this
*     data set ( MOD(1 yr) )
	   cx_grid(cx) = cvar_grid( pcair, cx_data_set(def_cx) )
	   cx_unstand_grid(cx) = .TRUE.
	   cx_regrid_trans( x_dim, cx ) = prgrd_exact
	   cx_regrid_trans( y_dim, cx ) = prgrd_exact
	   cx_regrid_trans( z_dim, cx ) = prgrd_exact
	   cx_regrid_trans( t_dim, cx ) = prgrd_linear_interp

* ... subscript on same axis as calling variable for all below
	ELSEIF ( cv_lo_d4(com) .GT. pcv_hourly_flg ) THEN
* ... special treatment of "hourly_needed" diagnostic data
	   IF ( HOURLY_DATA( grid ) ) THEN
* ... use 1 hr earlier for exact match of Philander/Cox code
	      cx_lo_ss(cx,t_dim) = cx_lo_ss(cx,t_dim) - 1
	      re_flesh = .TRUE.
	   ENDIF
	ELSEIF ( cx_lo_ss(cx,idim) .EQ. unspecified_int4 ) THEN
* ... allow unspecified regions to pass through still unspecified 2/90
	   re_flesh = .FALSE.
	ELSE
	   IF ( cv_lo_ds( com,t_dim ) .NE. 0 ) THEN
	      cx_lo_ss(cx,t_dim) = cx_lo_ss(cx,t_dim) + cv_lo_d4( com )
	      re_flesh = .TRUE.
	   ENDIF
	   IF ( cv_hi_ds( com,t_dim ) .NE. 0 ) THEN
	      cx_hi_ss(cx,t_dim) = cx_hi_ss(cx,t_dim) + cv_hi_d4( com )
	      re_flesh = .TRUE.
	   ENDIF
	ENDIF
* finalize the time axis
	IF ( re_flesh ) THEN
	   cx_by_ss( t_dim, cx ) = .TRUE.
	   CALL CONFINE_AXIS( t_dim, cx, status )
	   IF ( status .NE. ferr_ok ) RETURN
	   CALL FLESH_OUT_AXIS( t_dim, cx, status )
	   IF ( status .NE. ferr_ok ) RETURN
	ENDIF

* success
	status = ferr_ok
	RETURN

* error exit
 5000	RETURN
 5100	IF ( special_var .AND. var .EQ. pairt ) THEN
	   CALL ERRMSG( ferr_var_not_in_set, status,
     .			'need to SET DATA CLIM_AIRT_'//dp_airt_name(cairt),
     .			*5000 )
	ELSE
	   CALL ERRMSG( ferr_var_not_in_set, status,
     .						 fvar_name_code(var), *5000 )
	ENDIF
	END
