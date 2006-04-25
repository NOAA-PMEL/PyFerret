
	CHARACTER*(*) FUNCTION VAR_UNITS( cx )

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
* return units text string for variable

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 4/4/88
* V200:  6/20/89 - eliminated "named" variables
*	  1/3/90 - mr --> cx
*	 1/12/90 - allow unit changes by transforms
*       10/22/92 - added units on uvars
* V420: variable units may be modified by regrid transforms of the variable
* V500: 12/98 *sh* - added cat_const_var ... of the form "{1,3,5}"
* V530  8/00 *sh* - added support for cat_string
* v553 *kob* 9/03- need to add buff, slen for assignment of VAR_UNITS - 
*                  g77 bug would cause crash when trying to prepend chars
*                  onto CHAR function variable 
* v554 *acm* 1/04 - add variable category cat_counter_var
* V600 *acm* 3/06 - add variable category cat_attrib_val

	include	'tmap_dims.parm'
	include	'xdset_info.cmn_text'
	external xdset_info_data
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include	'ferret.parm'
	include	'xtext_info.cmn'
	include	'xcontext.cmn'
	include	'xcalc_vars.cmn'
	include	'xvariables.cmn'

* calling argument declarations
	INTEGER	cx

* internal variable declarations
	LOGICAL UNIT_CHG_AX
	INTEGER	TM_LENSTR1, slen, category, variable, idim, axis, trans
	CHARACTER*2048 buff

* initialize
	category = cx_category( cx )
	variable = cx_variable( cx )

* select units by category of variable
	IF     ( category .EQ. cat_file_var ) THEN

	   VAR_UNITS = ds_var_units( variable )

	ELSEIF ( category .EQ. cat_calc_var ) THEN

	   VAR_UNITS = unit_text( cvar_unit_code(variable) )

	ELSEIF ( category .EQ. cat_user_var ) THEN

	   VAR_UNITS = uvar_units( variable )

	ELSEIF ( category .EQ. cat_pseudo_var 
     .	   .OR.  category .EQ. cat_constant
     .	   .OR.  category .EQ. cat_string
     .	   .OR.  category .EQ. cat_const_var
     .	   .OR.  category .EQ. cat_counter_var
     .	   .OR.  category .EQ. cat_attrib_val ) THEN

	   VAR_UNITS = ' '

	ELSE

	   VAR_UNITS = 'unit_err'

	ENDIF

* certain transformations change the units
	IF ( UNIT_CHG_AX( cx, idim, trans ) ) THEN
	   IF ( trans .EQ. punit_mod_loc ) THEN
	      axis = grid_line( idim, cx_grid(cx) )
	      VAR_UNITS = line_units( axis )
	   ELSEIF ( trans .EQ. punit_mod_variance ) THEN
	      IF ( VAR_UNITS .NE. ' ' ) THEN
	         slen = TM_LENSTR1(VAR_UNITS)
		 buff = VAR_UNITS
	         VAR_UNITS = '('//buff(:slen) // ')^2'
	      ENDIF
	   ELSEIF ( trans .EQ. punit_mod_npoints ) THEN
	      VAR_UNITS = '# of points'
	   ENDIF	      

	ENDIF

	RETURN
	END
