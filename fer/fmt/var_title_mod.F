	CHARACTER*(*) FUNCTION VAR_TITLE_MOD ( cx )

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
* V530:  8/00 *sh* - added support for cat_string
* V533:  6/01 *sh* - added transformation @EVNT
*       *acm* 3/12 cleanup ifdefs and unnecessary include files
*       *acm* 3/12 Add E and F dimensions (use nferdims in tmap_dims.parm)
*       *acm* 4/12 6D Ferret: time axis may be in t or f direction. 
* V690 *sh* suppress comment about regridding for 1pt 'false regridding'
*       used on  curvilinear variables.  It will be documented in ASCII_LIMITS.
* V698  3/16 *acm* Removed Philander-Seigel code including diagnostic 
*                  variables, pre-defined axes and grids, cat_calc_var.
* V751  8/19 *acm* Special label if it was a regrid-to-DSG

        IMPLICIT NONE
	include	'tmap_dims.parm'
	include	'xdset_info.cmn_text'
	external xdset_info_data
	include	'xtm_grid.cmn_text'
	external xgt_grid_data
	include	'ferret.parm'
	include	'instance_table.parm'
	include	'xcontext.cmn'
	include	'xalgebra.cmn'
	include	'xtext_info.cmn'

* calling argument declarations
	INTEGER cx

* internal variable declarations
	LOGICAL	TM_ITSA_DSG, add_on, all_q, vcalc, false_regrid
	INTEGER	TM_LENSTR, TM_LENSTR1, CX_DIM_LEN, TM_DSG_DSET_FROM_GRID,
     .		var, categ, dset, cairt, cairt_dset,
     .		mlen, t_len, maxlen, idim, trans, slen, k_lo
	CHARACTER ALG_TRANS_TITL*20, TM_FMT*48,
     .		REGRID_COMMENTS*64, CD_DSG_FEATURENAME*20,
     .          trans_title*40, buffer*24, dummy*60

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
	IF ( categ .EQ. cat_string ) RETURN

* length so far
	mlen = TM_LENSTR( var_title_mod )
	add_on = mlen .GT. 0
	mlen = MAX( 1, mlen )		! avoid zero len string errors

* add axis name to pseudo variables
	IF ( categ .EQ. cat_pseudo_var ) THEN
	   idim = MOD( var-1, 6 ) + 1
	   var_title_mod = 'axis '//line_name(grid_line(idim, cx_grid(cx)))
	ENDIF

* add remarks about in-plane transformations
	DO 100 idim = 1, nferdims
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
	      ELSEIF ( cx_trans( idim, cx ) .EQ. trans_event_mask ) THEN
                 trans_title = trans_title(:t_len)//' at '//
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
*    ... check for 'false 1pt regrid'
	false_regrid = .FALSE.
	DO idim = 1, nferdims
	   IF (cx_aux_stat(idim,cx) .EQ. paux_stat_false_1pt_regrid)
     .		false_regrid = .TRUE.
	ENDDO
	IF (   cx_unstand_grid(cx)
     .	 .AND. .NOT.false_regrid ) THEN
	   IF ( add_on ) THEN
	      var_title_mod = var_title_mod(:mlen) //' - '
	      mlen = MIN( maxlen, mlen + 3 )
	   ELSE
	      add_on = .TRUE.
	   ENDIF

*    ... special label if it was a regrid-to-DSG
	   IF ( TM_ITSA_DSG(cx_grid(cx)) ) THEN
	      dset = TM_DSG_DSET_FROM_GRID(cx_grid(cx))
	      buffer = CD_DSG_FEATURENAME(dsg_orientation(dset))
	      slen = TM_LENSTR1( buffer )
	      
	      dummy = 'regrid: to '//buffer(:slen)//' DSG grid'
	      var_title_mod =  var_title_mod(:mlen)//'regrid: to '//buffer(:slen)//' DSG grid'
	      slen = slen + 13

  	  ELSE
	      var_title_mod =  var_title_mod(:mlen)//'regrid: '//
     .			REGRID_COMMENTS(cx, slen)
	  ENDIF
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
