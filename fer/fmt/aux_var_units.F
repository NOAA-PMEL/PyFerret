	CHARACTER*(*) FUNCTION AUX_VAR_UNITS( cx, idim )

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
* return units text string for an auxiliary variable

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V698+ *sh* 5/15 - derived from var_units.F
* V698  3/16 *acm* Removed Philander-Seigel code including diagnostic 
*                  variables, pre-defined axes and grids, cat_calc_var.
* V698  3/16 *sh* added logic for cat_ef_time2d  (ACTS_LIKE_FVAR)

	include	'tmap_dims.parm'
	include	'xdset_info.cmn_text'
	external xdset_info_data
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'xtext_info.cmn'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
        include 'xpyvar_info.cmn'

* calling argument declarations
	INTEGER	cx, idim

* internal variable declarations
	LOGICAL UNIT_CHG_AX, ACTS_LIKE_FVAR
	INTEGER	category, variable, status

* initialize
	category = cx_aux_cat( idim, cx )
	variable = cx_aux_var( idim, cx )

	IF (variable .EQ. unspecified_int4) THEN
	   AUX_VAR_UNITS = 'bad units'
	   CALL ERRMSG( ferr_internal, status, 'aux var cx err', *5000 )
	ENDIF

* select units by category of variable
	IF     ( ACTS_LIKE_FVAR(category) ) THEN

	   AUX_VAR_UNITS = ds_var_units( variable )

	ELSEIF ( category .EQ. cat_pystat_var ) THEN

	   AUX_VAR_UNITS = pyvar_units( variable )

	ELSEIF ( category .EQ. cat_user_var ) THEN

	   AUX_VAR_UNITS = uvar_units( variable )

	ELSEIF ( category .EQ. cat_pseudo_var 
     .	   .OR.  category .EQ. cat_constant
     .	   .OR.  category .EQ. cat_string
     .	   .OR.  category .EQ. cat_const_var
     .	   .OR.  category .EQ. cat_counter_var
     .	   .OR.  category .EQ. cat_attrib_val ) THEN

	   AUX_VAR_UNITS = ' '

	ELSE

	   AUX_VAR_UNITS = 'unit_err'

	ENDIF

 5000	RETURN
	END
