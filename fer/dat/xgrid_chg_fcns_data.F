	BLOCK DATA XGRID_CHG_FCNS_DATA

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
* function names and descriptive information for internally-defined grid-
* changing functions

* V450 - 7/79 - minor change to text on ZAXREPLACE
* V491 - 1/98 - changed function TEST2 so that it replaces T axis
* V500: 1/99 added gfcn_arg_type in support of string arguments
*       2/99 changed UNRAVEL to XSEQUENCE, YSEQUENCE, ZSEQUENCE, TSEQUENCE

* the addition (or major modification) of an internal grid-changing function
* requires changes to these routines:
*	grid_chg_fcns.parm	- parameters to number the functions
*	XGRID_CHG_FCNS_DATA	- description of the function and args
*	DO_INTERNAL_GC_FCN	- code to execute the function
* (opt)	GCF_GET_IMPOSED_AXES	- return the "imposed" axes, if any
* (opt)	GCF_IMPOSE_RESULT_LIMS	- fill in limits of "imposed" axes, if any

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/grid_chg_fcns.parm'
	include 'ferret_cmn/xgrid_chg_fcns.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:GRID_CHG_FCNS.PARM'
	INCLUDE 'FERRET_CMN:XGRID_CHG_FCNS.CMN'
#endif

* internal variable declarations
	INTEGER		i


* XSEQUENCE(arg) - same values as input grid but unravelled into an X line
	INTEGER		p_xunravel_arg_start, p_xunravel_num_args 
	PARAMETER     ( p_xunravel_arg_start = 1,
     .			p_xunravel_num_args  = 1 )

	DATA gfcn_name         (p_xunravel) /'XSEQUENCE'/
	DATA gfcn_version      (p_xunravel) / 1.0 /
	DATA gfcn_descr        (p_xunravel) / 'unravel grid to a line in X' /
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
	DATA gfcn_arg_type(p_xunravel_arg_start+0)/parg_type_float/
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
     .			p_times2_num_args  = 1 )


	DATA gfcn_name         (p_times2) /'ECHO'/
	DATA gfcn_version      (p_times2) / 1.0 /
	DATA gfcn_descr        (p_times2)
     .			/ 'echo passed string (test routine)' /
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
     .		/.TRUE., .TRUE., .TRUE.,.FALSE. /

* RESHAPE(data,grid_and_start) - same values as input grid but reshaped
*   into the grid of the 2nd arg ... starting at start point (1,1,1,1)
	INTEGER		p_reshape_arg_start, p_reshape_num_args 
	PARAMETER     ( p_reshape_arg_start = p_times2_arg_start +
     .					      p_times2_num_args,
     .			p_reshape_num_args  = 2 )

	DATA gfcn_name         (p_reshape) /'RESHAPE'/
	DATA gfcn_version      (p_reshape) / 1.0 /
	DATA gfcn_descr        (p_reshape) / 'reshape A to grid of B' /
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
	DATA gfcn_arg_type(p_reshape_arg_start+0)/parg_type_float/
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
	DATA gfcn_descr        (p_zlayer) /
     .		'interpolate V onto Z axis of ZAX using Z values in ZVALS' /
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
	DATA gfcn_arg_type(p_zlayer_arg_start+3)/parg_type_float/
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
	DATA gfcn_arg_type(p_yunravel_arg_start+0)/parg_type_float/
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
	PARAMETER     ( p_zunravel_arg_start = p_zlayer_arg_start +
     .					       p_zlayer_num_args,
     .			p_zunravel_num_args  = 1 )
	DATA gfcn_name         (p_zunravel) /'ZSEQUENCE'/
	DATA gfcn_version      (p_zunravel) / 1.0 /
	DATA gfcn_descr        (p_zunravel) / 'unravel grid to a line in Z' /
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
	DATA gfcn_arg_type(p_zunravel_arg_start+0)/parg_type_float/
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
	PARAMETER     ( p_tunravel_arg_start = p_zlayer_arg_start +
     .					       p_zlayer_num_args,
     .			p_tunravel_num_args  = 1 )
	DATA gfcn_name         (p_tunravel) /'TSEQUENCE'/
	DATA gfcn_version      (p_tunravel) / 1.0 /
	DATA gfcn_descr        (p_tunravel) / 'unravel grid to a line in T' /
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
	DATA gfcn_arg_type(p_tunravel_arg_start+0)/parg_type_float/
	DATA gfcn_arg_units(p_tunravel_arg_start+0)/' '/
	DATA gfcn_arg_descr(p_tunravel_arg_start+0)/' '/
	DATA (gfcn_arg_extend_lo (i,p_tunravel_arg_start+0),i=1,4)
     .		  / 0, 0, 0, 0 /
	DATA (gfcn_arg_extend_hi (i,p_tunravel_arg_start+0),i=1,4)
     .		  / 0, 0, 0, 0 /
* ... merge the axes implied by argument 1 ?
	DATA (gfcn_axis_implied_from(i, p_tunravel_arg_start+0),i=1,4)
     .		/.FALSE., .FALSE., .FALSE.,.FALSE. /


* internal parameters helpful for initialization
	DATA gfcn_num_internal/p_tunravel/

* template for next function ...
	INTEGER		p_next, p_next_arg_start, p_next_num_args 
	PARAMETER     ( p_next = p_tunravel + 1,
     .			p_next_arg_start = p_tunravel_arg_start +
     .					   p_tunravel_num_args,
     .			p_next_num_args  = 1 )

	END

