	BLOCK DATA XGRID_CHG_FCNS_DATA

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
* function names and descriptive information for internally-defined grid-
* changing functions

* V450 - 7/79 - minor change to text on ZAXREPLACE
* V491 - 1/98 - changed function TEST2 so that it replaces T axis
* V500: 1/99 added gfcn_arg_type in support of string arguments
*       2/99 changed UNRAVEL to XSEQUENCE, YSEQUENCE, ZSEQUENCE, TSEQUENCE
*       3/99 *kob* fixed bugs in above additions - parameter references
*	incorrect
*	3/99 *sh* added a numerical argument to ECHO to debug stack problem
* V510: added SAMPLE* functions as *internal* GC functions
* V533: *sh* 6/01 - allow some functions to take parg_type_multi
*            7/01 - added function SPAWN  
* V540: *sh* 10/01 - use improved data typing parameters parg_type_*
*       *acm*11/01 - clarify the function description for ZAXREPLACE
*	     12/01 - bug fix in arg type of SPAWN function
* V541: *kob* 6/02 - Add STRLEN, UPCASE,DNCASE,STRINDEX, STRRINDEX
*                        SUBSTRING, STRCAT, STRFLOAT 
*                    internal string handling functions   

* the addition (or major modification) of an internal grid-changing function
* requires changes to these routines:
*	grid_chg_fcns.parm	- parameters to number the functions
*	XGRID_CHG_FCNS_DATA	- description of the function and args
*	DO_INTERNAL_GC_FCN	- code to execute the function
* (opt)	GCF_GET_IMPOSED_AXES	- return the "imposed" axes, if any
* (opt)	GCF_IMPOSE_RESULT_LIMS	- fill in limits of "imposed" axes, if any

	include 'ferret.parm'
	include 'grid_chg_fcns.parm'
	include 'xgrid_chg_fcns.cmn'

* internal variable declarations
	INTEGER		i


* XSEQUENCE(arg) - same values as input grid but unravelled into an X line
	INTEGER		p_xunravel_arg_start, p_xunravel_num_args 
	PARAMETER     ( p_xunravel_arg_start = 1,
     .			p_xunravel_num_args  = 1 )

	DATA gfcn_name         (p_xunravel) /'XSEQUENCE'/
	DATA gfcn_version      (p_xunravel) / 1.0 /
	DATA gfcn_descr        (p_xunravel) / 'unravel grid to a line in X' /
	DATA gfcn_rtn_type     (p_xunravel) /pfcn_rtn_arg1_type /
	DATA gfcn_num_reqd_args(p_xunravel) /p_xunravel_num_args/
	DATA gfcn_has_vari_args(p_xunravel) /.FALSE./
	DATA (gfcn_axis_will_be(i,p_xunravel),i=1,4)/
     .						pgc_axis_is_abstract,
     .						pgc_axis_is_normal,
     .						pgc_axis_is_normal,
     .						pgc_axis_is_normal /
	DATA (gfcn_piecemeal_ok(i,p_xunravel),i=1,4) / .FALSE.,
     .						      .FALSE.,
     .						      .FALSE.,
     .						      .FALSE. /
	DATA gfcn_arg_ptr(p_xunravel) /p_xunravel_arg_start/

* ... description of arguments
	DATA gfcn_arg_name(p_xunravel_arg_start+0)/'VAR'/
	DATA gfcn_arg_type(p_xunravel_arg_start+0)/parg_type_float_or_string/
	DATA gfcn_arg_units(p_xunravel_arg_start+0)/' '/
	DATA gfcn_arg_descr(p_xunravel_arg_start+0)/' '/
	DATA (gfcn_arg_extend_lo (i,p_xunravel_arg_start+0),i=1,4)
     .		  / 0, 0, 0, 0 /
	DATA (gfcn_arg_extend_hi (i,p_xunravel_arg_start+0),i=1,4)
     .		  / 0, 0, 0, 0 /
* ... merge the axes implied by argument 1 ?
	DATA (gfcn_axis_implied_from(i, p_xunravel_arg_start+0),i=1,4)
     .		/.FALSE., .FALSE., .FALSE.,.FALSE. /


* TIMES2(arg) a.k.a. "ECHO" -- display string  --> debugging function
* 1/98 - changed TIMES2 to test custom axis replacement of T axis
	INTEGER		p_times2_arg_start, p_times2_num_args 
	PARAMETER     ( p_times2_arg_start = p_xunravel_arg_start +
     .					     p_xunravel_num_args,
     .			p_times2_num_args  = 2 )


	DATA gfcn_name         (p_times2) /'ECHO'/
	DATA gfcn_version      (p_times2) / 1.0 /
	DATA gfcn_descr        (p_times2)
     .			/ 'echo passed string (test routine)' /
	DATA gfcn_rtn_type     (p_times2) / pfcn_rtn_float /
	DATA gfcn_num_reqd_args(p_times2) /p_times2_num_args/
	DATA gfcn_has_vari_args(p_times2) /.FALSE./

	DATA (gfcn_axis_will_be(i,p_times2),i=1,4)/
     .						pgc_axis_implied_by_args,
     .						pgc_axis_implied_by_args,
     .						pgc_axis_implied_by_args,
     .						pgc_axis_implied_by_args/
	DATA (gfcn_piecemeal_ok(i,p_times2),i=1,4) / .TRUE.,
     .						     .TRUE.,
     .						     .TRUE.,
     .						     .TRUE. /
	DATA gfcn_arg_ptr(p_times2) /p_times2_arg_start/

* ... description of arguments
	DATA gfcn_arg_name(p_times2_arg_start+0)/'STR'/
	DATA gfcn_arg_type(p_times2_arg_start+0)/parg_type_string/
	DATA gfcn_arg_units(p_times2_arg_start+0)/' '/
	DATA gfcn_arg_descr(p_times2_arg_start+0)/'input text'/
	DATA (gfcn_arg_extend_lo (i,p_times2_arg_start+0),i=1,4)
     .		  / 0, 0, 0, 0 /
	DATA (gfcn_arg_extend_hi (i,p_times2_arg_start+0),i=1,4)
     .		  / 0, 0, 0, 0 /
* ... merge the axes implied by argument 1 ?
	DATA (gfcn_axis_implied_from(i, p_times2_arg_start+0),i=1,4)
     .		/.FALSE., .FALSE., .FALSE.,.FALSE. /
* ... argument 2
	DATA gfcn_arg_name(p_times2_arg_start+1)/'NUM'/
	DATA gfcn_arg_type(p_times2_arg_start+1)/parg_type_float/
	DATA gfcn_arg_units(p_times2_arg_start+1)/'anything'/
	DATA gfcn_arg_descr(p_times2_arg_start+1)/'numerical variable'/
	DATA (gfcn_arg_extend_lo (i,p_times2_arg_start+1),i=1,4)
     .		  / 0, 0, 0, 0 /
	DATA (gfcn_arg_extend_hi (i,p_times2_arg_start+1),i=1,4)
     .		  / 0, 0, 0, 0 /
* ... merge the axes implied by argument 2 ?
	DATA (gfcn_axis_implied_from(i, p_times2_arg_start+1),i=1,4)
     .		/.TRUE., .TRUE., .TRUE.,.TRUE. /


* RESHAPE(data,grid_and_start) - same values as input grid but reshaped
*   into the grid of the 2nd arg ... starting at start point (1,1,1,1)
	INTEGER		p_reshape_arg_start, p_reshape_num_args 
	PARAMETER     ( p_reshape_arg_start = p_times2_arg_start +
     .					      p_times2_num_args,
     .			p_reshape_num_args  = 2 )

	DATA gfcn_name         (p_reshape) /'RESHAPE'/
	DATA gfcn_version      (p_reshape) / 1.0 /
	DATA gfcn_descr        (p_reshape) / 'reshape A to grid of B' /
	DATA gfcn_rtn_type     (p_reshape) /pfcn_rtn_arg1_type /
	DATA gfcn_num_reqd_args(p_reshape) /p_reshape_num_args/
	DATA gfcn_has_vari_args(p_reshape) /.FALSE./
	DATA (gfcn_axis_will_be(i,p_reshape),i=1,4)/
     .						pgc_axis_supplied_herein,
     .						pgc_axis_supplied_herein,
     .						pgc_axis_supplied_herein,
     .						pgc_axis_supplied_herein /
	DATA (gfcn_piecemeal_ok(i,p_reshape),i=1,4) / .FALSE.,
     .						      .FALSE.,
     .						      .FALSE.,
     .						      .FALSE. /
	DATA gfcn_arg_ptr(p_reshape) /p_reshape_arg_start/

* ... description of arguments
	DATA gfcn_arg_name(p_reshape_arg_start+0)/'A'/
	DATA gfcn_arg_type(p_reshape_arg_start+0)/parg_type_float_or_string/
	DATA gfcn_arg_units(p_reshape_arg_start+0)/' '/
	DATA gfcn_arg_descr(p_reshape_arg_start+0)/'data to be reshaped'/
	DATA gfcn_arg_name(p_reshape_arg_start+1)/'B'/
	DATA gfcn_arg_type(p_reshape_arg_start+1)/parg_type_float/
	DATA gfcn_arg_units(p_reshape_arg_start+1)/' '/
	DATA gfcn_arg_descr(p_reshape_arg_start+1)/'destination grid'/
	DATA (gfcn_arg_extend_lo (i,p_reshape_arg_start+0),i=1,4)
     .		  / 0, 0, 0, 0 /
	DATA (gfcn_arg_extend_hi (i,p_reshape_arg_start+0),i=1,4)
     .		  / 0, 0, 0, 0 /
* ... merge the axes implied by argument 1 ?
*  Note: set to TRUE despite the fact that all of the
*     resultant axes are "pgc_axis_supplied_herein". This is to cover the
*     special case in which an axis of the result (imposed) **happens to**
*     match the axis of the argument. In this case TRUE means that context
*     limits do get passed upstream.
	DATA (gfcn_axis_implied_from(i, p_reshape_arg_start+0),i=1,4)
     .		/.TRUE., .TRUE., .TRUE.,.TRUE. /
* ... merge the axes implied by argument 2 ?
	DATA (gfcn_axis_implied_from(i, p_reshape_arg_start+1),i=1,4)
     .		/.FALSE., .FALSE., .FALSE.,.FALSE. /

* ZAXREPLACE(var, depthoflayer, zaxis)
*   interpolate the sigma layer onto the Z (depth) axis provided guided by
*   the depthoflayer field provided
	INTEGER		p_zlayer_arg_start, p_zlayer_num_args 
	PARAMETER     ( p_zlayer_arg_start = p_reshape_arg_start +
     .					     p_reshape_num_args,
     .			p_zlayer_num_args  = 3 )

	DATA gfcn_name         (p_zlayer) /'ZAXREPLACE'/
	DATA gfcn_version      (p_zlayer) / 1.0 /
	DATA gfcn_descr        (p_zlayer)(1:51) /
     .	         'regrid V onto Z axis of ZAX based upon Z values in '/
	DATA gfcn_descr        (p_zlayer)(52:124) /
     .           'ZVALS using linear interpolation'/

	DATA gfcn_rtn_type     (p_zlayer) / pfcn_rtn_float /
	DATA gfcn_num_reqd_args(p_zlayer) /p_zlayer_num_args/
	DATA gfcn_has_vari_args(p_zlayer) /.FALSE./
	DATA (gfcn_axis_will_be(i,p_zlayer),i=1,4)/
     .						pgc_axis_implied_by_args,
     .						pgc_axis_implied_by_args,
     .						pgc_axis_implied_by_args,
     .						pgc_axis_implied_by_args/
	DATA (gfcn_piecemeal_ok(i,p_zlayer),i=1,4) /  .FALSE.,
     .						      .FALSE.,
     .						      .FALSE.,
     .						      .TRUE. /
	DATA gfcn_arg_ptr(p_zlayer) /p_zlayer_arg_start/

* ... description of arguments
	DATA gfcn_arg_name(p_zlayer_arg_start+0)/'V'/
	DATA gfcn_arg_type(p_zlayer_arg_start+0)/parg_type_float/
	DATA gfcn_arg_units(p_zlayer_arg_start+0)/' '/
	DATA gfcn_arg_descr(p_zlayer_arg_start+0)/
     .		'variable on native Z axis'/
	DATA gfcn_arg_name(p_zlayer_arg_start+1)/'ZVALS'/
	DATA gfcn_arg_type(p_zlayer_arg_start+1)/parg_type_float/
	DATA gfcn_arg_units(p_zlayer_arg_start+1)/' '/
	DATA gfcn_arg_descr(p_zlayer_arg_start+1)/
     .	'Z-value field corresponding to data points of V'/
	DATA gfcn_arg_name(p_zlayer_arg_start+2)/'ZAX'/
	DATA gfcn_arg_type(p_zlayer_arg_start+2)/parg_type_float/
	DATA gfcn_arg_units(p_zlayer_arg_start+2)/' '/
	DATA gfcn_arg_descr(p_zlayer_arg_start+2)/
     .		'variable with desired Z (depth) axis points'/
	DATA (gfcn_arg_extend_lo (i,p_zlayer_arg_start+0),i=1,4)
     .		  / 0, 0, 0, 0 /
	DATA (gfcn_arg_extend_hi (i,p_zlayer_arg_start+0),i=1,4)
     .		  / 0, 0, 0, 0 /
* ... merge the axes implied by argument 1 ?
	DATA (gfcn_axis_implied_from(i, p_zlayer_arg_start+0),i=1,4)
     .		/.TRUE., .TRUE., .FALSE.,.TRUE. /
* ... argument 2 
	DATA (gfcn_axis_implied_from(i, p_zlayer_arg_start+1),i=1,4)
     .		/.TRUE., .TRUE., .FALSE.,.TRUE. /
* ... argument 3
	DATA (gfcn_axis_implied_from(i, p_zlayer_arg_start+2),i=1,4)
     .		/.FALSE., .FALSE., .TRUE.,.FALSE. /


* YSEQUENCE(arg) - same values as input grid but unravelled into a Y line
	INTEGER		p_yunravel_arg_start, p_yunravel_num_args
	PARAMETER     ( p_yunravel_arg_start = p_zlayer_arg_start +
     .					       p_zlayer_num_args,
     .			p_yunravel_num_args  = 1 )
	DATA gfcn_name         (p_yunravel) /'YSEQUENCE'/
	DATA gfcn_version      (p_yunravel) / 1.0 /
	DATA gfcn_descr        (p_yunravel) / 'unravel grid to a line in Y' /
	DATA gfcn_rtn_type     (p_yunravel) /pfcn_rtn_arg1_type /
	DATA gfcn_num_reqd_args(p_yunravel) /p_yunravel_num_args/
	DATA gfcn_has_vari_args(p_yunravel) /.FALSE./
	DATA (gfcn_axis_will_be(i,p_yunravel),i=1,4)/
     .						pgc_axis_is_normal,
     .						pgc_axis_is_abstract,
     .						pgc_axis_is_normal,
     .						pgc_axis_is_normal /
	DATA (gfcn_piecemeal_ok(i,p_yunravel),i=1,4) / .FALSE.,
     .						      .FALSE.,
     .						      .FALSE.,
     .						      .FALSE. /
	DATA gfcn_arg_ptr(p_yunravel) /p_yunravel_arg_start/

* ... description of arguments
	DATA gfcn_arg_name(p_yunravel_arg_start+0)/'VAR'/
	DATA gfcn_arg_type(p_yunravel_arg_start+0)/parg_type_float_or_string/
	DATA gfcn_arg_units(p_yunravel_arg_start+0)/' '/
	DATA gfcn_arg_descr(p_yunravel_arg_start+0)/' '/
	DATA (gfcn_arg_extend_lo (i,p_yunravel_arg_start+0),i=1,4)
     .		  / 0, 0, 0, 0 /
	DATA (gfcn_arg_extend_hi (i,p_yunravel_arg_start+0),i=1,4)
     .		  / 0, 0, 0, 0 /
* ... merge the axes implied by argument 1 ?
	DATA (gfcn_axis_implied_from(i, p_yunravel_arg_start+0),i=1,4)
     .		/.FALSE., .FALSE., .FALSE.,.FALSE. /


* ZSEQUENCE(arg) - same values as input grid but unravelled into a Y line
	INTEGER		p_zunravel_arg_start, p_zunravel_num_args
	PARAMETER     ( p_zunravel_arg_start = p_yunravel_arg_start +
     .					       p_yunravel_num_args,
     .			p_zunravel_num_args  = 1 )
	DATA gfcn_name         (p_zunravel) /'ZSEQUENCE'/
	DATA gfcn_version      (p_zunravel) / 1.0 /
	DATA gfcn_descr        (p_zunravel) / 'unravel grid to a line in Z' /
	DATA gfcn_rtn_type     (p_zunravel) /pfcn_rtn_arg1_type /
	DATA gfcn_num_reqd_args(p_zunravel) /p_zunravel_num_args/
	DATA gfcn_has_vari_args(p_zunravel) /.FALSE./
	DATA (gfcn_axis_will_be(i,p_zunravel),i=1,4)/
     .						pgc_axis_is_normal,
     .						pgc_axis_is_normal,
     .						pgc_axis_is_abstract,
     .						pgc_axis_is_normal /
	DATA (gfcn_piecemeal_ok(i,p_zunravel),i=1,4) / .FALSE.,
     .						      .FALSE.,
     .						      .FALSE.,
     .						      .FALSE. /
	DATA gfcn_arg_ptr(p_zunravel) /p_zunravel_arg_start/

* ... description of arguments
	DATA gfcn_arg_name(p_zunravel_arg_start+0)/'VAR'/
	DATA gfcn_arg_type(p_zunravel_arg_start+0)/parg_type_float_or_string/
	DATA gfcn_arg_units(p_zunravel_arg_start+0)/' '/
	DATA gfcn_arg_descr(p_zunravel_arg_start+0)/' '/
	DATA (gfcn_arg_extend_lo (i,p_zunravel_arg_start+0),i=1,4)
     .		  / 0, 0, 0, 0 /
	DATA (gfcn_arg_extend_hi (i,p_zunravel_arg_start+0),i=1,4)
     .		  / 0, 0, 0, 0 /
* ... merge the axes implied by argument 1 ?
	DATA (gfcn_axis_implied_from(i, p_zunravel_arg_start+0),i=1,4)
     .		/.FALSE., .FALSE., .FALSE.,.FALSE. /


* TSEQUENCE(arg) - same values as input grid but unravelled into a Y line
	INTEGER		p_tunravel_arg_start, p_tunravel_num_args
	PARAMETER     ( p_tunravel_arg_start = p_zunravel_arg_start +
     .					       p_zunravel_num_args,
     .			p_tunravel_num_args  = 1 )
	DATA gfcn_name         (p_tunravel) /'TSEQUENCE'/
	DATA gfcn_version      (p_tunravel) / 1.0 /
	DATA gfcn_descr        (p_tunravel) / 'unravel grid to a line in T' /
	DATA gfcn_rtn_type     (p_tunravel) /pfcn_rtn_arg1_type /
	DATA gfcn_num_reqd_args(p_tunravel) /p_tunravel_num_args/
	DATA gfcn_has_vari_args(p_tunravel) /.FALSE./
	DATA (gfcn_axis_will_be(i,p_tunravel),i=1,4)/
     .						pgc_axis_is_normal,
     .						pgc_axis_is_normal,
     .						pgc_axis_is_normal,
     .						pgc_axis_is_abstract/
	DATA (gfcn_piecemeal_ok(i,p_tunravel),i=1,4) / .FALSE.,
     .						      .FALSE.,
     .						      .FALSE.,
     .						      .FALSE. /
	DATA gfcn_arg_ptr(p_tunravel) /p_tunravel_arg_start/

* ... description of arguments
	DATA gfcn_arg_name(p_tunravel_arg_start+0)/'VAR'/
	DATA gfcn_arg_type(p_tunravel_arg_start+0)/parg_type_float_or_string/
	DATA gfcn_arg_units(p_tunravel_arg_start+0)/' '/
	DATA gfcn_arg_descr(p_tunravel_arg_start+0)/' '/
	DATA (gfcn_arg_extend_lo (i,p_tunravel_arg_start+0),i=1,4)
     .		  / 0, 0, 0, 0 /
	DATA (gfcn_arg_extend_hi (i,p_tunravel_arg_start+0),i=1,4)
     .		  / 0, 0, 0, 0 /
* ... merge the axes implied by argument 1 ?
	DATA (gfcn_axis_implied_from(i, p_tunravel_arg_start+0),i=1,4)
     .		/.FALSE., .FALSE., .FALSE.,.FALSE. /


* SAMPLEI - sample argument 2 at the indices indicated in arg 2
	INTEGER		p_samplei_arg_start, p_samplei_num_args 
	PARAMETER     ( p_samplei_arg_start = p_tunravel_arg_start +
     .					      p_tunravel_num_args,
     .			p_samplei_num_args  = 2 )

	DATA gfcn_name         (p_samplei) /'SAMPLEI'/
	DATA gfcn_version      (p_samplei) / 1.0 /
	DATA gfcn_descr        (p_samplei)
     .		/ 'sample a field at a list of X indices' /
	DATA gfcn_rtn_type     (p_samplei) /pfcn_rtn_arg1_type /
	DATA gfcn_num_reqd_args(p_samplei) /p_samplei_num_args/
	DATA gfcn_has_vari_args(p_samplei) /.FALSE./
	DATA (gfcn_axis_will_be(i,p_samplei),i=1,4)/
     .					pgc_axis_is_abstract,
     .					pgc_axis_implied_by_args,
     .					pgc_axis_implied_by_args,
     .					pgc_axis_implied_by_args /
	DATA (gfcn_piecemeal_ok(i,p_samplei),i=1,4) / .FALSE.,
     .						      .FALSE.,
     .						      .FALSE.,
     .						      .FALSE. /
	DATA gfcn_arg_ptr(p_samplei) /p_samplei_arg_start/

* ... description of arguments
	DATA gfcn_arg_name(p_samplei_arg_start+0)/'TO_BE_SAMPLED'/
	DATA gfcn_arg_type(p_samplei_arg_start+0)/parg_type_float_or_string/
	DATA gfcn_arg_units(p_samplei_arg_start+0)/' '/
	DATA gfcn_arg_descr(p_samplei_arg_start+0)
     .			/'data to sample at list of X indices supplied '/
	DATA (gfcn_arg_extend_lo (i,p_samplei_arg_start+0),i=1,4)
     .		  / 0, 0, 0, 0 /
	DATA (gfcn_arg_extend_hi (i,p_samplei_arg_start+0),i=1,4)
     .		  / 0, 0, 0, 0 /
	DATA gfcn_arg_name(p_samplei_arg_start+1)/'X_INDICES'/
	DATA gfcn_arg_type(p_samplei_arg_start+1)/parg_type_float/
	DATA gfcn_arg_units(p_samplei_arg_start+1)/' '/
	DATA gfcn_arg_descr(p_samplei_arg_start+1)
     .			/'list of X indices at which to sample'/
	DATA (gfcn_arg_extend_lo (i,p_samplei_arg_start+1),i=1,4)
     .		  / 0, 0, 0, 0 /
	DATA (gfcn_arg_extend_hi (i,p_samplei_arg_start+1),i=1,4)
     .		  / 0, 0, 0, 0 /
* ... merge the axes implied by argument 1 ?
	DATA (gfcn_axis_implied_from(i, p_samplei_arg_start+0),i=1,4)
     .		/.FALSE., .TRUE., .TRUE.,.TRUE. /
* ... by argument 2?
	DATA (gfcn_axis_implied_from(i, p_samplei_arg_start+1),i=1,4)
     .		/.FALSE., .FALSE., .FALSE.,.FALSE. /


* SAMPLEJ - sample argument 2 at the indices indicated in arg 2
	INTEGER		p_samplej_arg_start, p_samplej_num_args 
	PARAMETER     ( p_samplej_arg_start = p_samplei_arg_start +
     .					      p_samplei_num_args,
     .			p_samplej_num_args  = 2 )

	DATA gfcn_name         (p_samplej) /'SAMPLEJ'/
	DATA gfcn_version      (p_samplej) / 1.0 /
	DATA gfcn_descr        (p_samplej)
     .		/ 'sample a field at a list of Y indices' /
	DATA gfcn_rtn_type     (p_samplej) /pfcn_rtn_arg1_type /
	DATA gfcn_num_reqd_args(p_samplej) /p_samplej_num_args/
	DATA gfcn_has_vari_args(p_samplej) /.FALSE./
	DATA (gfcn_axis_will_be(i,p_samplej),i=1,4)/
     .					pgc_axis_implied_by_args,
     .					pgc_axis_is_abstract,
     .					pgc_axis_implied_by_args,
     .					pgc_axis_implied_by_args /
	DATA (gfcn_piecemeal_ok(i,p_samplej),i=1,4) / .FALSE.,
     .						      .FALSE.,
     .						      .FALSE.,
     .						      .FALSE. /
	DATA gfcn_arg_ptr(p_samplej) /p_samplej_arg_start/

* ... description of arguments
	DATA gfcn_arg_name(p_samplej_arg_start+0)/'TO_BE_SAMPLED'/
	DATA gfcn_arg_type(p_samplej_arg_start+0)/parg_type_float_or_string/
	DATA gfcn_arg_units(p_samplej_arg_start+0)/' '/
	DATA gfcn_arg_descr(p_samplej_arg_start+0)
     .			/'data to sample at list of Y indices supplied '/
	DATA (gfcn_arg_extend_lo (i,p_samplej_arg_start+0),i=1,4)
     .		  / 0, 0, 0, 0 /
	DATA (gfcn_arg_extend_hi (i,p_samplej_arg_start+0),i=1,4)
     .		  / 0, 0, 0, 0 /
	DATA gfcn_arg_name(p_samplej_arg_start+1)/'Y_INDICES'/
	DATA gfcn_arg_type(p_samplej_arg_start+1)/parg_type_float/
	DATA gfcn_arg_units(p_samplej_arg_start+1)/' '/
	DATA gfcn_arg_descr(p_samplej_arg_start+1)
     .			/'list of Y indices at which to sample'/
	DATA (gfcn_arg_extend_lo (i,p_samplej_arg_start+1),i=1,4)
     .		  / 0, 0, 0, 0 /
	DATA (gfcn_arg_extend_hi (i,p_samplej_arg_start+1),i=1,4)
     .		  / 0, 0, 0, 0 /
* ... merge the axes implied by argument 1 ?
	DATA (gfcn_axis_implied_from(i, p_samplej_arg_start+0),i=1,4)
     .		/.TRUE., .FALSE., .TRUE.,.TRUE. /
* ... by argument 2?
	DATA (gfcn_axis_implied_from(i, p_samplej_arg_start+1),i=1,4)
     .		/.FALSE., .FALSE., .FALSE.,.FALSE. /


* SAMPLEK - sample argument 2 at the indices indicated in arg 2
	INTEGER		p_samplek_arg_start, p_samplek_num_args 
	PARAMETER     ( p_samplek_arg_start = p_samplej_arg_start +
     .					      p_samplej_num_args,
     .			p_samplek_num_args  = 2 )

	DATA gfcn_name         (p_samplek) /'SAMPLEK'/
	DATA gfcn_version      (p_samplek) / 1.0 /
	DATA gfcn_descr        (p_samplek)
     .		/ 'sample a field at a list of Z indices' /
	DATA gfcn_rtn_type     (p_samplek) /pfcn_rtn_arg1_type /
	DATA gfcn_num_reqd_args(p_samplek) /p_samplek_num_args/
	DATA gfcn_has_vari_args(p_samplek) /.FALSE./
	DATA (gfcn_axis_will_be(i,p_samplek),i=1,4)/
     .					pgc_axis_implied_by_args,
     .					pgc_axis_implied_by_args,
     .					pgc_axis_is_abstract,
     .					pgc_axis_implied_by_args /
	DATA (gfcn_piecemeal_ok(i,p_samplek),i=1,4) / .FALSE.,
     .						      .FALSE.,
     .						      .FALSE.,
     .						      .FALSE. /
	DATA gfcn_arg_ptr(p_samplek) /p_samplek_arg_start/

* ... description of arguments
	DATA gfcn_arg_name(p_samplek_arg_start+0)/'TO_BE_SAMPLED'/
	DATA gfcn_arg_type(p_samplek_arg_start+0)/parg_type_float_or_string/
	DATA gfcn_arg_units(p_samplek_arg_start+0)/' '/
	DATA gfcn_arg_descr(p_samplek_arg_start+0)
     .			/'data to sample at list of Z indices supplied '/
	DATA (gfcn_arg_extend_lo (i,p_samplek_arg_start+0),i=1,4)
     .		  / 0, 0, 0, 0 /
	DATA (gfcn_arg_extend_hi (i,p_samplek_arg_start+0),i=1,4)
     .		  / 0, 0, 0, 0 /
	DATA gfcn_arg_name(p_samplek_arg_start+1)/'Z_INDICES'/
	DATA gfcn_arg_type(p_samplek_arg_start+1)/parg_type_float/
	DATA gfcn_arg_units(p_samplek_arg_start+1)/' '/
	DATA gfcn_arg_descr(p_samplek_arg_start+1)
     .			/'list of Z indices at which to sample'/
	DATA (gfcn_arg_extend_lo (i,p_samplek_arg_start+1),i=1,4)
     .		  / 0, 0, 0, 0 /
	DATA (gfcn_arg_extend_hi (i,p_samplek_arg_start+1),i=1,4)
     .		  / 0, 0, 0, 0 /
* ... merge the axes implied by argument 1 ?
	DATA (gfcn_axis_implied_from(i, p_samplek_arg_start+0),i=1,4)
     .		/ .TRUE., .TRUE., .FALSE., .TRUE. /
* ... by argument 2?
	DATA (gfcn_axis_implied_from(i, p_samplek_arg_start+1),i=1,4)
     .		/.FALSE., .FALSE., .FALSE.,.FALSE. /


* SAMPLEL - sample argument 2 at the indices indicated in arg 2
	INTEGER		p_samplel_arg_start, p_samplel_num_args 
	PARAMETER     ( p_samplel_arg_start = p_samplek_arg_start +
     .					      p_samplek_num_args, 
     .			p_samplel_num_args  = 2 )

	DATA gfcn_name         (p_samplel) /'SAMPLEL'/
	DATA gfcn_version      (p_samplel) / 1.0 /
	DATA gfcn_descr        (p_samplel)
     .		/ 'sample a field at a list of T indices' /
	DATA gfcn_rtn_type     (p_samplel) /pfcn_rtn_arg1_type /
	DATA gfcn_num_reqd_args(p_samplel) /p_samplel_num_args/
	DATA gfcn_has_vari_args(p_samplel) /.FALSE./
	DATA (gfcn_axis_will_be(i,p_samplel),i=1,4)/
     .					pgc_axis_implied_by_args,
     .					pgc_axis_implied_by_args,
     .					pgc_axis_implied_by_args,
     .					pgc_axis_is_abstract/
	DATA (gfcn_piecemeal_ok(i,p_samplel),i=1,4) / .FALSE.,
     .						      .FALSE.,
     .						      .FALSE.,
     .						      .FALSE. /
	DATA gfcn_arg_ptr(p_samplel) /p_samplel_arg_start/

* ... description of arguments
	DATA gfcn_arg_name(p_samplel_arg_start+0)/'TO_BE_SAMPLED'/
	DATA gfcn_arg_type(p_samplel_arg_start+0)/parg_type_float_or_string/
	DATA gfcn_arg_units(p_samplel_arg_start+0)/' '/
	DATA gfcn_arg_descr(p_samplel_arg_start+0)
     .			/'data to sample at list of T indices supplied '/
	DATA (gfcn_arg_extend_lo (i,p_samplel_arg_start+0),i=1,4)
     .		  / 0, 0, 0, 0 /
	DATA (gfcn_arg_extend_hi (i,p_samplel_arg_start+0),i=1,4)
     .		  / 0, 0, 0, 0 /
	DATA gfcn_arg_name(p_samplel_arg_start+1)/'T_INDICES'/
	DATA gfcn_arg_type(p_samplel_arg_start+1)/parg_type_float/
	DATA gfcn_arg_units(p_samplel_arg_start+1)/' '/
	DATA gfcn_arg_descr(p_samplel_arg_start+1)
     .			/'list of T indices at which to sample'/
	DATA (gfcn_arg_extend_lo (i,p_samplel_arg_start+1),i=1,4)
     .		  / 0, 0, 0, 0 /
	DATA (gfcn_arg_extend_hi (i,p_samplel_arg_start+1),i=1,4)
     .		  / 0, 0, 0, 0 /
* ... merge the axes implied by argument 1 ?
	DATA (gfcn_axis_implied_from(i, p_samplel_arg_start+0),i=1,4)
     .		/.TRUE., .TRUE., .TRUE., .FALSE. /
* ... by argument 2?
	DATA (gfcn_axis_implied_from(i, p_samplel_arg_start+1),i=1,4)
     .		/.FALSE., .FALSE., .FALSE.,.FALSE. /


* SPAWN("unix cmnd") - execute a Unix system command and return strings
	INTEGER		p_spawn_arg_start, p_spawn_num_args
	PARAMETER     ( p_spawn_arg_start = p_samplel_arg_start +
     .					    p_samplel_num_args,
     .			p_spawn_num_args  = 1 )
	DATA gfcn_name         (p_spawn) /'SPAWN'/
	DATA gfcn_version      (p_spawn) / 1.0 /
	DATA gfcn_descr        (p_spawn) / 'execute a system command' /
	DATA gfcn_rtn_type     (p_spawn) /pfcn_rtn_string /
	DATA gfcn_num_reqd_args(p_spawn) /p_spawn_num_args/
	DATA gfcn_has_vari_args(p_spawn) /.FALSE./
	DATA (gfcn_axis_will_be(i,p_spawn),i=1,4)/
     .						pgc_axis_is_abstract,
     .						pgc_axis_is_normal,
     .						pgc_axis_is_normal,
     .						pgc_axis_is_normal/
	DATA (gfcn_piecemeal_ok(i,p_spawn),i=1,4) / .FALSE.,
     .						    .FALSE.,
     .						    .FALSE.,
     .						    .FALSE. /
	DATA gfcn_arg_ptr(p_spawn) /p_spawn_arg_start/

* ... description of arguments
	DATA gfcn_arg_name(p_spawn_arg_start+0)/'STR'/
	DATA gfcn_arg_type(p_spawn_arg_start+0)/parg_type_string/
	DATA gfcn_arg_units(p_spawn_arg_start+0)/' '/
	DATA gfcn_arg_descr(p_spawn_arg_start+0)/'Unix command string'/
	DATA (gfcn_arg_extend_lo (i,p_spawn_arg_start+0),i=1,4)
     .		  / 0, 0, 0, 0 /
	DATA (gfcn_arg_extend_hi (i,p_spawn_arg_start+0),i=1,4)
     .		  / 0, 0, 0, 0 /
* ... merge the axes implied by argument 1 ?
	DATA (gfcn_axis_implied_from(i, p_spawn_arg_start+0),i=1,4)
     .		/.FALSE., .FALSE., .FALSE.,.FALSE. /

* STRCMP(string1, string2) - compare 2 string *arrays* as in C function
	INTEGER		p_strcmp_arg_start, p_strcmp_num_args
	PARAMETER     ( p_strcmp_arg_start = p_spawn_arg_start +
     .					    p_spawn_num_args,
     .			p_strcmp_num_args  = 2 )
	DATA gfcn_name         (p_strcmp) /'STRCMP'/
	DATA gfcn_version      (p_strcmp) / 1.0 /
	DATA gfcn_descr        (p_strcmp) /
     .	'compare strings: +, 0, or - for str1 >, =, or < str2 rspctvly' /
	DATA gfcn_rtn_type     (p_strcmp) /pfcn_rtn_float /
	DATA gfcn_num_reqd_args(p_strcmp) /p_strcmp_num_args/
	DATA gfcn_has_vari_args(p_strcmp) /.FALSE./
	DATA (gfcn_axis_will_be(i,p_strcmp),i=1,4)/
     .					pgc_axis_implied_by_args,
     .					pgc_axis_implied_by_args,
     .					pgc_axis_implied_by_args,
     .					pgc_axis_implied_by_args/
	DATA (gfcn_piecemeal_ok(i,p_strcmp),i=1,4) / .FALSE.,
     .						    .FALSE.,
     .						    .FALSE.,
     .						    .FALSE. /
	DATA gfcn_arg_ptr(p_strcmp) /p_strcmp_arg_start/

* ... description of argument 1
	DATA gfcn_arg_name(p_strcmp_arg_start+0)/'STR1'/
	DATA gfcn_arg_type(p_strcmp_arg_start+0)/parg_type_string/
	DATA gfcn_arg_units(p_strcmp_arg_start+0)/' '/
	DATA gfcn_arg_descr(p_strcmp_arg_start+0)/'string array 1'/
	DATA (gfcn_arg_extend_lo (i,p_strcmp_arg_start+0),i=1,4)
     .		  / 0, 0, 0, 0 /
	DATA (gfcn_arg_extend_hi (i,p_strcmp_arg_start+0),i=1,4)
     .		  / 0, 0, 0, 0 /
* ... merge the axes implied by argument 1 ?
	DATA (gfcn_axis_implied_from(i, p_strcmp_arg_start+0),i=1,4)
     .		/.TRUE., .TRUE., .TRUE.,.TRUE. /

* ... description of argument 2
	DATA gfcn_arg_name(p_strcmp_arg_start+1)/'STR2'/
	DATA gfcn_arg_type(p_strcmp_arg_start+1)/parg_type_string/
	DATA gfcn_arg_units(p_strcmp_arg_start+1)/' '/
	DATA gfcn_arg_descr(p_strcmp_arg_start+1)/'string array 2'/
	DATA (gfcn_arg_extend_lo (i,p_strcmp_arg_start+1),i=1,4)
     .		  / 0, 0, 0, 0 /
	DATA (gfcn_arg_extend_hi (i,p_strcmp_arg_start+1),i=1,4)
     .		  / 0, 0, 0, 0 /
* ... merge the axes implied by argument 1 ?
	DATA (gfcn_axis_implied_from(i, p_strcmp_arg_start+1),i=1,4)
     .		/.TRUE., .TRUE., .TRUE.,.TRUE. /


* STRLEN(string1) - return the length of the string
	INTEGER		p_strlen_arg_start, p_strlen_num_args
	PARAMETER     ( p_strlen_arg_start = p_strcmp_arg_start +
     .					    p_strcmp_num_args,
     .			p_strlen_num_args  = 1 )
	DATA gfcn_name         (p_strlen) /'STRLEN'/
	DATA gfcn_version      (p_strlen) / 1.0 /
	DATA gfcn_descr        (p_strlen) /
     .	'determine string length' /
	DATA gfcn_rtn_type     (p_strlen) /pfcn_rtn_float /
	DATA gfcn_num_reqd_args(p_strlen) /p_strlen_num_args/
	DATA gfcn_has_vari_args(p_strlen) /.FALSE./
	DATA (gfcn_axis_will_be(i,p_strlen),i=1,4)/
     .					pgc_axis_implied_by_args,
     .					pgc_axis_implied_by_args,
     .					pgc_axis_implied_by_args,
     .					pgc_axis_implied_by_args/
	DATA (gfcn_piecemeal_ok(i,p_strlen),i=1,4) / .FALSE.,
     .						    .FALSE.,
     .						    .FALSE.,
     .						    .FALSE. /
	DATA gfcn_arg_ptr(p_strlen) /p_strlen_arg_start/

* ... description of argument 1
	DATA gfcn_arg_name(p_strlen_arg_start+0)/'STR'/
	DATA gfcn_arg_type(p_strlen_arg_start+0)/parg_type_string/
	DATA gfcn_arg_units(p_strlen_arg_start+0)/' '/
	DATA gfcn_arg_descr(p_strlen_arg_start+0)/'string array 1'/
	DATA (gfcn_arg_extend_lo (i,p_strlen_arg_start+0),i=1,4)
     .		  / 0, 0, 0, 0 /
	DATA (gfcn_arg_extend_hi (i,p_strlen_arg_start+0),i=1,4)
     .		  / 0, 0, 0, 0 /
* ... merge the axes implied by argument 1 ?
	DATA (gfcn_axis_implied_from(i, p_strlen_arg_start+0),i=1,4)
     .		/.TRUE., .TRUE., .TRUE.,.TRUE. /


* UPCASE(string1) - return string in all upper case
	INTEGER		p_upcase_arg_start, p_upcase_num_args
	PARAMETER     ( p_upcase_arg_start = p_strlen_arg_start +
     .					    p_strlen_num_args,
     .			p_upcase_num_args  = 1 )
	DATA gfcn_name         (p_upcase) /'UPCASE'/
	DATA gfcn_version      (p_upcase) / 1.0 /
	DATA gfcn_descr        (p_upcase) /
     .	'upper case entire string' /
	DATA gfcn_rtn_type     (p_upcase) /pfcn_rtn_string /
	DATA gfcn_num_reqd_args(p_upcase) /p_upcase_num_args/
	DATA gfcn_has_vari_args(p_upcase) /.FALSE./
	DATA (gfcn_axis_will_be(i,p_upcase),i=1,4)/
     .					pgc_axis_implied_by_args,
     .					pgc_axis_implied_by_args,
     .					pgc_axis_implied_by_args,
     .					pgc_axis_implied_by_args/
	DATA (gfcn_piecemeal_ok(i,p_upcase),i=1,4) / .FALSE.,
     .						    .FALSE.,
     .						    .FALSE.,
     .						    .FALSE. /
	DATA gfcn_arg_ptr(p_upcase) /p_upcase_arg_start/

* ... description of argument 1
	DATA gfcn_arg_name(p_upcase_arg_start+0)/'STR'/
	DATA gfcn_arg_type(p_upcase_arg_start+0)/parg_type_string/
	DATA gfcn_arg_units(p_upcase_arg_start+0)/' '/
	DATA gfcn_arg_descr(p_upcase_arg_start+0)/'string array 1'/
	DATA (gfcn_arg_extend_lo (i,p_upcase_arg_start+0),i=1,4)
     .		  / 0, 0, 0, 0 /
	DATA (gfcn_arg_extend_hi (i,p_upcase_arg_start+0),i=1,4)
     .		  / 0, 0, 0, 0 /
* ... merge the axes implied by argument 1 ?
	DATA (gfcn_axis_implied_from(i, p_upcase_arg_start+0),i=1,4)
     .		/.TRUE., .TRUE., .TRUE.,.TRUE. /



* STRINDEX(string1, substring) - find star char position of substring
	INTEGER		p_strindex_arg_start, p_strindex_num_args
	PARAMETER     ( p_strindex_arg_start = p_upcase_arg_start +
     .					    p_upcase_num_args,
     .			p_strindex_num_args  = 2 )
	DATA gfcn_name         (p_strindex) /'STRINDEX'/
	DATA gfcn_version      (p_strindex) / 1.0 /
	DATA gfcn_descr        (p_strindex) /
     .	'Return start character position of substring in string' /
	DATA gfcn_rtn_type     (p_strindex) /pfcn_rtn_float /
	DATA gfcn_num_reqd_args(p_strindex) /p_strindex_num_args/
	DATA gfcn_has_vari_args(p_strindex) /.FALSE./
	DATA (gfcn_axis_will_be(i,p_strindex),i=1,4)/
     .					pgc_axis_implied_by_args,
     .					pgc_axis_implied_by_args,
     .					pgc_axis_implied_by_args,
     .					pgc_axis_implied_by_args/
	DATA (gfcn_piecemeal_ok(i,p_strindex),i=1,4) / .FALSE.,
     .						    .FALSE.,
     .						    .FALSE.,
     .						    .FALSE. /
	DATA gfcn_arg_ptr(p_strindex) /p_strindex_arg_start/

* ... description of argument 1
	DATA gfcn_arg_name(p_strindex_arg_start+0)/'STR1'/
	DATA gfcn_arg_type(p_strindex_arg_start+0)/parg_type_string/
	DATA gfcn_arg_units(p_strindex_arg_start+0)/' '/
	DATA gfcn_arg_descr(p_strindex_arg_start+0)/'string array 1'/
	DATA (gfcn_arg_extend_lo (i,p_strindex_arg_start+0),i=1,4)
     .		  / 0, 0, 0, 0 /
	DATA (gfcn_arg_extend_hi (i,p_strindex_arg_start+0),i=1,4)
     .		  / 0, 0, 0, 0 /
* ... merge the axes implied by argument 1 ?
	DATA (gfcn_axis_implied_from(i, p_strindex_arg_start+0),i=1,4)
     .		/.TRUE., .TRUE., .TRUE.,.TRUE. /

* ... description of argument 2
	DATA gfcn_arg_name(p_strindex_arg_start+1)/'SUBSTR'/
	DATA gfcn_arg_type(p_strindex_arg_start+1)/parg_type_string/
	DATA gfcn_arg_units(p_strindex_arg_start+1)/' '/
	DATA gfcn_arg_descr(p_strindex_arg_start+1)/'string array 2'/
	DATA (gfcn_arg_extend_lo (i,p_strindex_arg_start+1),i=1,4)
     .		  / 0, 0, 0, 0 /
	DATA (gfcn_arg_extend_hi (i,p_strindex_arg_start+1),i=1,4)
     .		  / 0, 0, 0, 0 /
* ... merge the axes implied by argument 1 ?
	DATA (gfcn_axis_implied_from(i, p_strindex_arg_start+1),i=1,4)
     .		/.TRUE., .TRUE., .TRUE.,.TRUE. /



* STRRINDEX(string1, substring) - find last char position of substring
	INTEGER		p_strrindex_arg_start, p_strrindex_num_args
	PARAMETER     ( p_strrindex_arg_start = p_strindex_arg_start +
     .					    p_strindex_num_args,
     .			p_strrindex_num_args  = 2 )
	DATA gfcn_name         (p_strrindex) /'STRRINDEX'/
	DATA gfcn_version      (p_strrindex) / 1.0 /
	DATA gfcn_descr        (p_strrindex) /
     .	'Return last character position of substring in string' /
	DATA gfcn_rtn_type     (p_strrindex) /pfcn_rtn_float /
	DATA gfcn_num_reqd_args(p_strrindex) /p_strrindex_num_args/
	DATA gfcn_has_vari_args(p_strrindex) /.FALSE./
	DATA (gfcn_axis_will_be(i,p_strrindex),i=1,4)/
     .					pgc_axis_implied_by_args,
     .					pgc_axis_implied_by_args,
     .					pgc_axis_implied_by_args,
     .					pgc_axis_implied_by_args/
	DATA (gfcn_piecemeal_ok(i,p_strrindex),i=1,4) / .FALSE.,
     .						    .FALSE.,
     .						    .FALSE.,
     .						    .FALSE. /
	DATA gfcn_arg_ptr(p_strrindex) /p_strrindex_arg_start/

* ... description of argument 1
	DATA gfcn_arg_name(p_strrindex_arg_start+0)/'STR1'/
	DATA gfcn_arg_type(p_strrindex_arg_start+0)/parg_type_string/
	DATA gfcn_arg_units(p_strrindex_arg_start+0)/' '/
	DATA gfcn_arg_descr(p_strrindex_arg_start+0)/'string array 1'/
	DATA (gfcn_arg_extend_lo (i,p_strrindex_arg_start+0),i=1,4)
     .		  / 0, 0, 0, 0 /
	DATA (gfcn_arg_extend_hi (i,p_strrindex_arg_start+0),i=1,4)
     .		  / 0, 0, 0, 0 /
* ... merge the axes implied by argument 1 ?
	DATA (gfcn_axis_implied_from(i, p_strrindex_arg_start+0),i=1,4)
     .		/.TRUE., .TRUE., .TRUE.,.TRUE. /

* ... description of argument 2
	DATA gfcn_arg_name(p_strrindex_arg_start+1)/'SUBSTR'/
	DATA gfcn_arg_type(p_strrindex_arg_start+1)/parg_type_string/
	DATA gfcn_arg_units(p_strrindex_arg_start+1)/' '/
	DATA gfcn_arg_descr(p_strrindex_arg_start+1)/'string array 2'/
	DATA (gfcn_arg_extend_lo (i,p_strrindex_arg_start+1),i=1,4)
     .		  / 0, 0, 0, 0 /
	DATA (gfcn_arg_extend_hi (i,p_strrindex_arg_start+1),i=1,4)
     .		  / 0, 0, 0, 0 /
* ... merge the axes implied by argument 1 ?
	DATA (gfcn_axis_implied_from(i, p_strrindex_arg_start+1),i=1,4)
     .		/.TRUE., .TRUE., .TRUE.,.TRUE. /



* DNCASE(string1) - return string in all lower case
	INTEGER		p_dncase_arg_start, p_dncase_num_args
	PARAMETER     ( p_dncase_arg_start = p_strrindex_arg_start +
     .					    p_strrindex_num_args,
     .			p_dncase_num_args  = 1 )
	DATA gfcn_name         (p_dncase) /'DNCASE'/
	DATA gfcn_version      (p_dncase) / 1.0 /
	DATA gfcn_descr        (p_dncase) /
     .	'make entire string lower case' /
	DATA gfcn_rtn_type     (p_dncase) /pfcn_rtn_string /
	DATA gfcn_num_reqd_args(p_dncase) /p_dncase_num_args/
	DATA gfcn_has_vari_args(p_dncase) /.FALSE./
	DATA (gfcn_axis_will_be(i,p_dncase),i=1,4)/
     .					pgc_axis_implied_by_args,
     .					pgc_axis_implied_by_args,
     .					pgc_axis_implied_by_args,
     .					pgc_axis_implied_by_args/
	DATA (gfcn_piecemeal_ok(i,p_dncase),i=1,4) / .FALSE.,
     .						    .FALSE.,
     .						    .FALSE.,
     .						    .FALSE. /
	DATA gfcn_arg_ptr(p_dncase) /p_dncase_arg_start/

* ... description of argument 1
	DATA gfcn_arg_name(p_dncase_arg_start+0)/'STR'/
	DATA gfcn_arg_type(p_dncase_arg_start+0)/parg_type_string/
	DATA gfcn_arg_units(p_dncase_arg_start+0)/' '/
	DATA gfcn_arg_descr(p_dncase_arg_start+0)/'string array 1'/
	DATA (gfcn_arg_extend_lo (i,p_dncase_arg_start+0),i=1,4)
     .		  / 0, 0, 0, 0 /
	DATA (gfcn_arg_extend_hi (i,p_dncase_arg_start+0),i=1,4)
     .		  / 0, 0, 0, 0 /
* ... merge the axes implied by argument 1 ?
	DATA (gfcn_axis_implied_from(i, p_dncase_arg_start+0),i=1,4)
     .		/.TRUE., .TRUE., .TRUE.,.TRUE. /


* STRCAT(string1, string2) - compare 2 string *arrays* as in C function
	INTEGER		p_strcat_arg_start, p_strcat_num_args
	PARAMETER     ( p_strcat_arg_start = p_dncase_arg_start +
     .					    p_dncase_num_args,
     .			p_strcat_num_args  = 2 )
	DATA gfcn_name         (p_strcat) /'STRCAT'/
	DATA gfcn_version      (p_strcat) / 1.0 /
	DATA gfcn_descr        (p_strcat) /
     .	'concatenate two strings' /
	DATA gfcn_rtn_type     (p_strcat) /pfcn_rtn_string /
	DATA gfcn_num_reqd_args(p_strcat) /p_strcat_num_args/
	DATA gfcn_has_vari_args(p_strcat) /.FALSE./
	DATA (gfcn_axis_will_be(i,p_strcat),i=1,4)/
     .					pgc_axis_implied_by_args,
     .					pgc_axis_implied_by_args,
     .					pgc_axis_implied_by_args,
     .					pgc_axis_implied_by_args/
	DATA (gfcn_piecemeal_ok(i,p_strcat),i=1,4) / .FALSE.,
     .						    .FALSE.,
     .						    .FALSE.,
     .						    .FALSE. /
	DATA gfcn_arg_ptr(p_strcat) /p_strcat_arg_start/

* ... description of argument 1
	DATA gfcn_arg_name(p_strcat_arg_start+0)/'STR1'/
	DATA gfcn_arg_type(p_strcat_arg_start+0)/parg_type_string/
	DATA gfcn_arg_units(p_strcat_arg_start+0)/' '/
	DATA gfcn_arg_descr(p_strcat_arg_start+0)/'string array 1'/
	DATA (gfcn_arg_extend_lo (i,p_strcat_arg_start+0),i=1,4)
     .		  / 0, 0, 0, 0 /
	DATA (gfcn_arg_extend_hi (i,p_strcat_arg_start+0),i=1,4)
     .		  / 0, 0, 0, 0 /
* ... merge the axes implied by argument 1 ?
	DATA (gfcn_axis_implied_from(i, p_strcat_arg_start+0),i=1,4)
     .		/.TRUE., .TRUE., .TRUE.,.TRUE. /

* ... description of argument 2
	DATA gfcn_arg_name(p_strcat_arg_start+1)/'STR2'/
	DATA gfcn_arg_type(p_strcat_arg_start+1)/parg_type_string/
	DATA gfcn_arg_units(p_strcat_arg_start+1)/' '/
	DATA gfcn_arg_descr(p_strcat_arg_start+1)/'string array 2'/
	DATA (gfcn_arg_extend_lo (i,p_strcat_arg_start+1),i=1,4)
     .		  / 0, 0, 0, 0 /
	DATA (gfcn_arg_extend_hi (i,p_strcat_arg_start+1),i=1,4)
     .		  / 0, 0, 0, 0 /
* ... merge the axes implied by argument 1 ?
	DATA (gfcn_axis_implied_from(i, p_strcat_arg_start+1),i=1,4)
     .		/.TRUE., .TRUE., .TRUE.,.TRUE. /



* SUBSTR(string1, offset, lenght) - clips a substring from a string
	INTEGER		p_substring_arg_start, p_substring_num_args
	PARAMETER     ( p_substring_arg_start = p_strcat_arg_start +
     .					    p_strcat_num_args,
     .			p_substring_num_args  = 3 )
	DATA gfcn_name         (p_substring) /'SUBSTRING'/
	DATA gfcn_version      (p_substring) / 1.0 /
	DATA gfcn_descr        (p_substring) /
     .	'concatenate two strings' /
	DATA gfcn_rtn_type     (p_substring) /pfcn_rtn_string /
	DATA gfcn_num_reqd_args(p_substring) /p_substring_num_args/
	DATA gfcn_has_vari_args(p_substring) /.FALSE./
	DATA (gfcn_axis_will_be(i,p_substring),i=1,4)/
     .					pgc_axis_implied_by_args,
     .					pgc_axis_implied_by_args,
     .					pgc_axis_implied_by_args,
     .					pgc_axis_implied_by_args/
	DATA (gfcn_piecemeal_ok(i,p_substring),i=1,4) / .FALSE.,
     .						    .FALSE.,
     .						    .FALSE.,
     .						    .FALSE. /
	DATA gfcn_arg_ptr(p_substring) /p_substring_arg_start/

* ... description of argument 1
	DATA gfcn_arg_name(p_substring_arg_start+0)/'STR'/
	DATA gfcn_arg_type(p_substring_arg_start+0)/parg_type_string/
	DATA gfcn_arg_units(p_substring_arg_start+0)/' '/
	DATA gfcn_arg_descr(p_substring_arg_start+0)/'string array'/
	DATA (gfcn_arg_extend_lo (i,p_substring_arg_start+0),i=1,4)
     .		  / 0, 0, 0, 0 /
	DATA (gfcn_arg_extend_hi (i,p_substring_arg_start+0),i=1,4)
     .		  / 0, 0, 0, 0 /
* ... merge the axes implied by argument 1 ?
	DATA (gfcn_axis_implied_from(i, p_substring_arg_start+0),i=1,4)
     .		/.TRUE., .TRUE., .TRUE.,.TRUE. /

* ... description of argument 2
	DATA gfcn_arg_name(p_substring_arg_start+1)/'OFFSET'/
	DATA gfcn_arg_type(p_substring_arg_start+1)/parg_type_float/
	DATA gfcn_arg_units(p_substring_arg_start+1)/' '/
	DATA gfcn_arg_descr(p_substring_arg_start+1)/'start position'/
	DATA (gfcn_arg_extend_lo (i,p_substring_arg_start+1),i=1,4)
     .		  / 0, 0, 0, 0 /
	DATA (gfcn_arg_extend_hi (i,p_substring_arg_start+1),i=1,4)
     .		  / 0, 0, 0, 0 /
* ... merge the axes implied by argument 1 ?
	DATA (gfcn_axis_implied_from(i, p_substring_arg_start+1),i=1,4)
     .		/.TRUE., .TRUE., .TRUE.,.TRUE. /


* ... description of argument 3
	DATA gfcn_arg_name(p_substring_arg_start+2)/'LENGTH'/
	DATA gfcn_arg_type(p_substring_arg_start+2)/parg_type_float/
	DATA gfcn_arg_units(p_substring_arg_start+2)/' '/
	DATA gfcn_arg_descr(p_substring_arg_start+2)/'number of chars'/
	DATA (gfcn_arg_extend_lo (i,p_substring_arg_start+2),i=1,4)
     .		  / 0, 0, 0, 0 /
	DATA (gfcn_arg_extend_hi (i,p_substring_arg_start+2),i=1,4)
     .		  / 0, 0, 0, 0 /
* ... merge the axes implied by argument 1 ?
	DATA (gfcn_axis_implied_from(i, p_substring_arg_start+2),i=1,4)
     .		/.TRUE., .TRUE., .TRUE.,.TRUE. /


* STRFLOAT(string1) - return float value of given string
	INTEGER		p_strfloat_arg_start, p_strfloat_num_args
	PARAMETER     ( p_strfloat_arg_start = p_substring_arg_start +
     .					    p_substring_num_args,
     .			p_strfloat_num_args  = 1 )
	DATA gfcn_name         (p_strfloat) /'STRFLOAT'/
	DATA gfcn_version      (p_strfloat) / 1.0 /
	DATA gfcn_descr        (p_strfloat) /
     .	'Return float value from character string' /
	DATA gfcn_rtn_type     (p_strfloat) /pfcn_rtn_float /
	DATA gfcn_num_reqd_args(p_strfloat) /p_strfloat_num_args/
	DATA gfcn_has_vari_args(p_strfloat) /.FALSE./
	DATA (gfcn_axis_will_be(i,p_strfloat),i=1,4)/
     .					pgc_axis_implied_by_args,
     .					pgc_axis_implied_by_args,
     .					pgc_axis_implied_by_args,
     .					pgc_axis_implied_by_args/
	DATA (gfcn_piecemeal_ok(i,p_strfloat),i=1,4) / .FALSE.,
     .						    .FALSE.,
     .						    .FALSE.,
     .						    .FALSE. /
	DATA gfcn_arg_ptr(p_strfloat) /p_strfloat_arg_start/

* ... description of argument 1
	DATA gfcn_arg_name(p_strfloat_arg_start+0)/'STR'/
	DATA gfcn_arg_type(p_strfloat_arg_start+0)/parg_type_string/
	DATA gfcn_arg_units(p_strfloat_arg_start+0)/' '/
	DATA gfcn_arg_descr(p_strfloat_arg_start+0)/'string array 1'/
	DATA (gfcn_arg_extend_lo (i,p_strfloat_arg_start+0),i=1,4)
     .		  / 0, 0, 0, 0 /
	DATA (gfcn_arg_extend_hi (i,p_strfloat_arg_start+0),i=1,4)
     .		  / 0, 0, 0, 0 /
* ... merge the axes implied by argument 1 ?
	DATA (gfcn_axis_implied_from(i, p_strfloat_arg_start+0),i=1,4)
     .		/.TRUE., .TRUE., .TRUE.,.TRUE. /





* internal parameters helpful for initialization
	DATA gfcn_num_internal/p_strfloat/

* template for next function ...
	INTEGER		p_next, p_next_arg_start, p_next_num_args 
	PARAMETER     ( p_next = p_strfloat + 1,
     .			p_next_arg_start = p_strfloat_arg_start +
     .					   p_strfloat_num_args,
     .			p_next_num_args  = 1 )

	END

