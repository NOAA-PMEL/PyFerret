	CHARACTER*(*) FUNCTION VAR_CODE ( category, variable )

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
* return name code for variable

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 6/4/86
* revision 1.0 - 1/6/87 - cat_expr and cat_temp changed
* revision 1.1 - 3/23/87 - added new TMAP library: time axes, file var. names
* revision 2.0 - 3/10/88 - removed cat_expr_var, added cat_named_var,
*			   changed cat_user_var formatting to use HASH_NAME
* V200:  6/20/89 - eliminated "named" variables
*       11/30/89 - process pseudo-variables, too
*	 12/4/89 - moved hashing of expression to SANITARY_VAR_CODE
* V320:  2/22/95 - added support for reverse-axis variables: cat_file_rev
* V500: 12/98 *sh* - added cat_const_var ... of the form "{1,3,5}"
*	 1/99 *sh8 ... and cat_string

#ifdef unix
	include	'tmap_dims.parm'
	include	'xdset_info.cmn_text'
	external xdset_info_data
	include	'ferret.parm'
	include	'xvariables.cmn'
	include	'xcalc_vars.cmn'
	include	'xalgebra.cmn'
#else
	INCLUDE	'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'TMAP_FORMAT:XDSET_INFO.CMN'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XCALC_VARS.CMN'
	INCLUDE	'FERRET_CMN:XALGEBRA.CMN'
#endif

* calling argument declarations:
	INTEGER category, variable

* internal variable declarations
	INTEGER 	uvar, item, start, end

	IF     ( category .EQ. cat_file_var 
     . .OR.	 category .EQ. cat_file_rev ) THEN

	   VAR_CODE = ds_var_code( variable )

	ELSEIF ( category .EQ. cat_calc_var ) THEN

	   VAR_CODE = cvar_name_code( variable )

	ELSEIF ( category .EQ. cat_user_var ) THEN

	   VAR_CODE = uvar_name_code(variable)

	ELSEIF ( category .EQ. cat_dummy_var ) THEN

	   VAR_CODE = 'dumm'

	ELSEIF ( category .EQ. cat_temp_var ) THEN

	   VAR_CODE = 'tpry'

	ELSEIF ( category .EQ. cat_pseudo_var ) THEN

	   VAR_CODE = alg_pvar( variable )

	ELSEIF ( category .EQ. cat_const_var ) THEN

	   uvar = variable/1000
           item = variable - 1000*uvar
           start = uvar_item_start( item, uvar )
           end   = uvar_item_end  ( item, uvar )
	   VAR_CODE = uvar_text(uvar)(start:end)

	ELSEIF ( category .EQ. cat_constant ) THEN

	   VAR_CODE = 'cnst'

	ELSEIF ( category .EQ. cat_string ) THEN

	   VAR_CODE = 'str'

	ELSE

	   VAR_CODE = 'bad*'

	ENDIF

 3000	FORMAT(A1,I3.3)
	RETURN
	END
