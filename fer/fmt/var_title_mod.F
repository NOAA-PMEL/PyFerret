	CHARACTER*(*) FUNCTION VAR_TITLE_MOD ( cx )

* return title modifications ( including in-plane transformations) for variable

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 4/28/87
* revision 0.1 - 6/25/87 - special remarks added to some derivatives
* revision 0.2 - 8/29/87 - special remarks added only if DT is known
*			   eliminated enclosing parens / "-" between remarks
* revision 0.3 - 9/10/87 - added "using nnn pts" for transforms
*			   and ds_var_modification --> ds_var_titl_mod
* revision 0.31- 10/23/87- "using" --> "by"
* revision 0.4 - 4/4/88  - added remarks about data-set-dependent calculations
* revision 0.4 - 5/5/88  - modified remarks to include QDF and QDFZ
* revision 0.5 - 7/2/88  - remarks about Am and multiple clim. airt fields for
*			   QFLX
* revision 0.6 - 7/25/88 - added regriding
* revision 0.7 - 1/4/89  - added regridding transformations
* V200:  6/2/89 - interchanged order of lo/hi_ss arguments
*		- check in-plane-ness for transforms to display
*	 1/3/89 - mr --> cx
* V230:  1/27/92 - added special string for @WEQ
*        8/18/92 - eliminated string(1:) notation for MAC compatibility
*        8/23/92 - changes to accomodate regridding transforms
* V420:	 9/25/95 - use REGRID_COMMENTS
* V430:  7/96 - pass cx to ALG_TRANS_TITL so it can document 4D transformations
* V455:  9/97 - modify labeling of xforms @CIA,CIB,CDA,CDB to avoid "by n pts"
* V491:  3/98 - crash due to boolean operator precedence error

#ifdef unix
	include	'tmap_format/tmap_dims.parm'
	include	'tmap_format/xdset_info.cmn_text'
	external xdset_info_data
	include	'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/instance_table.parm'
	include	'ferret_cmn/xcontext.cmn'
	include	'ferret_cmn/xalgebra.cmn'
	include	'ferret_cmn/xtext_info.cmn'
	include	'ferret_cmn/xdset_parms.cmn'
#else
	INCLUDE	'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'TMAP_FORMAT:XDSET_INFO.CMN'
	INCLUDE	'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:INSTANCE_TABLE.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XALGEBRA.CMN'
	INCLUDE	'FERRET_CMN:XTEXT_INFO.CMN'
	INCLUDE	'FERRET_CMN:XDSET_PARMS.CMN'
#endif

* calling argument declarations
	INTEGER cx

* internal variable declarations
	LOGICAL	add_on, all_q, vcalc 
	INTEGER	TM_LENSTR, TM_LENSTR1, CX_DIM_LEN,
     .		var, categ, dset, cairt, cairt_dset,
     .		mlen, t_len, maxlen, idim, trans, slen, k_lo
	CHARACTER ALG_TRANS_TITL*20, LEFINT*4, TM_FMT*20,
     .		REGRID_COMMENTS*64,
     .          trans_title*40, buffer*24, separator*1

* initialize
	var     = cx_variable( cx )
	categ   = cx_category( cx )
	dset    = cx_data_set( cx )
	maxlen  = LEN( VAR_TITLE_MOD )

* begin with stored title modification string
	IF     ( categ .EQ. cat_file_var ) THEN
	   var_title_mod = ds_var_titl_mod( var )
	ELSE
	   var_title_mod = ' '
	ENDIF

* length so far
	mlen = TM_LENSTR( var_title_mod )
	add_on = mlen .GT. 0
	mlen = MAX( 1, mlen )		! avoid zero len string errors

* add remark about accuracy of d/dT derivative
	trans = cx_trans( t_dim, cx )
	IF ( trans .EQ. trans_no_transform 
     . .AND. cx_trans_arg(t_dim,cx) .NE. bad_val4
     . .AND. categ .EQ. cat_calc_var
     . .AND. ( var .EQ. pubps  .OR.  var .EQ. pvbps
     .		.OR.  (pdudt.LE. var .AND. var.LE.pdtdt) ) ) THEN
	   IF ( add_on ) THEN
	      var_title_mod = var_title_mod(:mlen) //' - '
	      mlen = MIN( maxlen, mlen + 3 )
	   ELSE
	      add_on = .TRUE.
	   ENDIF
	   IF ( cx_trans_arg(t_dim,cx) .EQ. bad_val4 ) THEN
	      var_title_mod = var_title_mod(:mlen) //'irregular dt'
	      mlen = MIN( maxlen, mlen+12 )
	   ELSE
	      var_title_mod = var_title_mod(:mlen) //
     .			     'dt='//TM_FMT( cx_trans_arg(t_dim,cx), 4, 8, slen )
	      mlen = MIN( maxlen, mlen+3+slen )
	   ENDIF
	ENDIF

* some variables need all the comments about surface heat flux
* namely: QFLX and (QDF or QDFZ) at the surface
	k_lo     = cx_lo_ss( cx, z_dim ) 
	vcalc = categ .EQ. cat_calc_var
	all_q = vcalc .AND. ( var.EQ.pqflx	.OR.
     .		  ( (var.EQ.pqdfz .OR. var.EQ.pqdf) .AND. k_lo.EQ.1 ) )

* add remark about clim. air T vs. const. air/sea temperature difference
	IF ( all_q .OR. 
     .	     ( vcalc .AND.  ( var.EQ.pqsen .OR. var.EQ.pqeva
     .			.OR.  var.EQ.pair  .OR. var.EQ.pcair ) ) ) THEN
	   IF ( add_on ) THEN
	      var_title_mod = var_title_mod(:mlen) //' - '
	      mlen = MIN( maxlen, mlen + 3 )
	   ELSE
	      add_on = .TRUE.
	   ENDIF
	   IF ( .NOT.dp_const_air_sst(dset) .OR. var .EQ. pcair ) THEN
	      cairt = dp_clim_air_sst( dset )		! cairt used in calcs
	      cairt_dset = dp_airt_dset( cairt )	! dset containing cairt
* ... base remark on experiment number to ensure that neither file names,
*     logicals, nor descriptors can point to wrong climatological data
	      var_title_mod = var_title_mod(:mlen) // ds_expnum( cairt_dset )
     .			      // ' air T'	
	      mlen = MIN( maxlen, mlen+10 )
	   ELSE
* ... constant air sst difference
	      var_title_mod = var_title_mod(:mlen) // 'air/sst dT=' //
     .		TM_FMT( dp_delta_air_sst(dset), 4, 8, slen )
	      mlen = MIN( maxlen, mlen+11+slen )
	   ENDIF
	ENDIF

* add remark about minimum wind used
	IF ( all_q .OR. 
     .	     ( vcalc .AND. (var.EQ.pqsen .OR. var.EQ.pqeva) ) ) THEN
	   var_title_mod = var_title_mod(:mlen) // ';min. wind=' //
     .		TM_FMT( dp_min_wind(dset), 4, 8, slen )
	   mlen = MIN( maxlen, mlen+11+slen )
	ENDIF

* add remark about the surface radiation calculation
	IF ( all_q .OR. ( vcalc .AND. var.EQ.pqrad ) ) THEN
	   IF ( add_on ) THEN
	      var_title_mod = var_title_mod(:mlen) //' - '
	      mlen = MIN( maxlen, mlen + 3 )
	   ELSE
	      add_on = .TRUE.
	   ENDIF
	   IF ( dp_phil_qflux(dset) ) THEN
	      var_title_mod = var_title_mod(:mlen) // 'Phil/Sieg Q flux'
	      mlen = MIN( maxlen, mlen+14 )
	   ELSEIF ( dp_ramp1_qflux(dset) ) THEN
	      var_title_mod = var_title_mod(:mlen) // 'single heat ramp'
	      mlen = MIN( maxlen, mlen+16 )
	   ELSEIF ( dp_ramp2_qflux(dset) ) THEN
	      var_title_mod = var_title_mod(:mlen) // 'double heat ramp'
	      mlen = MIN( maxlen, mlen+16 )
	   ENDIF
	ENDIF

* add remark about momentum diffusion coefficient
	IF ( vcalc .AND. (var.EQ.pudfh) .OR. (var.EQ.pvdfh) ) THEN
	   IF ( add_on ) THEN
	      var_title_mod = var_title_mod(:mlen) //' - '
	      mlen = MIN( maxlen, mlen + 3 )
	   ELSE
	      add_on = .TRUE.
	   ENDIF
	      buffer = 	      TM_FMT( dp_Am_factor(dset), 3, 5, slen )
	      var_title_mod = var_title_mod(:mlen)//
     .			      'Am=' // buffer(:slen)
	      mlen = MIN( maxlen, mlen+3+slen )
	ENDIF

* add axis name to pseudo variables
	IF ( categ .EQ. cat_pseudo_var ) THEN
	   idim = MOD( var-1, 4 ) + 1
	   var_title_mod = 'axis '//line_name(grid_line(idim, cx_grid(cx)))
	ENDIF

* add remarks about in-plane transformations
	DO 100 idim = 1, 4
	   trans = cx_trans( idim, cx )
	   IF ( trans .NE. trans_no_transform
     .	  .AND. trans .LT. trans_compress_code 
     .	  .AND. CX_DIM_LEN(idim,cx) .GT. 1 ) THEN
	      IF ( add_on ) THEN
	         var_title_mod = var_title_mod(:mlen) //' - '
	         mlen = MIN( maxlen, mlen + 3 )
	      ELSE
	         add_on = .TRUE.
	      ENDIF
	      trans_title = ALG_TRANS_TITL( trans, cx )
	      t_len = TM_LENSTR1( trans_title )
	      IF ( cx_trans( idim, cx ) .EQ. trans_wgted_eq ) THEN
                 trans_title = trans_title(:t_len)//' of '//
     .                TM_FMT( cx_trans_arg(idim,cx), 4, 12, slen )
                 t_len = t_len + 4 + slen
	      ELSEIF ( cx_trans_arg( idim, cx ) .NE. bad_val4 ) THEN
	         IF ( trans .GE. trans_nrst_dist_abv
     .        .AND.   trans .LE. trans_nrst_indx_blw ) GOTO 95	!9/97 @CIA trap
* ... add "by nnn pts"
	         trans_title = trans_title(:t_len)//' by '//
     .			TM_FMT( cx_trans_arg(idim,cx), 4, 8, slen )
	         t_len = t_len + 4 + slen
	         trans_title = trans_title(:t_len)//' pts'
	         t_len = t_len + 4
	      ENDIF
 95	      var_title_mod = var_title_mod(:mlen) // trans_title(:t_len) //
     .		' on ' // ww_dim_name( idim )
	      mlen = MIN( maxlen, mlen+t_len+5 )
	   ENDIF
 100	CONTINUE

* re-gridded ?
	IF ( cx_unstand_grid(cx) ) THEN
	   IF ( add_on ) THEN
	      var_title_mod = var_title_mod(:mlen) //' - '
	      mlen = MIN( maxlen, mlen + 3 )
	   ELSE
	      add_on = .TRUE.
	   ENDIF
	   var_title_mod =  var_title_mod(:mlen)//'regrid: '//
     .			REGRID_COMMENTS(cx, slen)
	   mlen = MIN( maxlen, mlen + 8 + slen )
	ENDIF

* signal if the string has been truncated for being too long
	IF ( mlen .EQ. maxlen ) VAR_TITLE_MOD(maxlen:maxlen) = '*'

* clean up
	IF ( VAR_TITLE_MOD .EQ. ' ' ) THEN
	   RETURN

* ... complication from avoiding zero length strings above
	ELSEIF ( VAR_TITLE_MOD(1:1) .EQ. ' ' ) THEN
	   VAR_TITLE_MOD = VAR_TITLE_MOD(2:maxlen)
	   mlen = mlen - 1
	ENDIF

	RETURN
	END
