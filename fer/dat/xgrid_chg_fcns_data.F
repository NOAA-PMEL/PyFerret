	BLOCK DATA XGRID_CHG_FCNS_DATA

* function names and descriptive information for internally-defined grid-
* changing functions

* V450 - 7/79 - minor change to text on ZAXREPLACE

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

* internal parameters helpful for initialization
	DATA gfcn_num_internal/ 4 /


* UNRAVEL(arg) - same values as input grid but unravelled into a line
	INTEGER		p_unravel_arg_start, p_unravel_num_args 
	PARAMETER     ( p_unravel_arg_start = 1,
     .			p_unravel_num_args  = 1 )

	DATA gfcn_name         (p_unravel) /'UNRAVEL'/
	DATA gfcn_version      (p_unravel) / 1.0 /
	DATA gfcn_descr        (p_unravel) / 'unravel grid to a line in X' /
	DATA gfcn_num_reqd_args(p_unravel) /p_unravel_num_args/
	DATA gfcn_has_vari_args(p_unravel) /.FALSE./
	DATA (gfcn_axis_will_be(i,p_unravel),i=1,4)/
     .						pgc_axis_is_abstract,
     .						pgc_axis_is_normal,
     .						pgc_axis_is_normal,
     .						pgc_axis_is_normal /
	DATA (gfcn_piecemeal_ok(i,p_unravel),i=1,4) / .FALSE.,
     .						      .FALSE.,
     .						      .FALSE.,
     .						      .FALSE. /
	DATA gfcn_arg_ptr(p_unravel) /p_unravel_arg_start/

* ... description of arguments
	DATA gfcn_arg_name(p_unravel_arg_start+0)/'X'/
	DATA gfcn_arg_units(p_unravel_arg_start+0)/' '/
	DATA gfcn_arg_descr(p_unravel_arg_start+0)/' '/
	DATA (gfcn_arg_extend_lo (i,p_unravel_arg_start+0),i=1,4)
     .		  / 0, 0, 0, 0 /
	DATA (gfcn_arg_extend_hi (i,p_unravel_arg_start+0),i=1,4)
     .		  / 0, 0, 0, 0 /
* ... merge the axes implied by argument 1 ?
	DATA (gfcn_axis_implied_from(i, p_unravel_arg_start+0),i=1,4)
     .		/.FALSE., .FALSE., .FALSE.,.FALSE. /


* TIMES2(arg) - multiply input value by 2 --> debugging function
	INTEGER		p_times2_arg_start, p_times2_num_args 
	PARAMETER     ( p_times2_arg_start = p_unravel_arg_start +
     .					     p_unravel_num_args,
     .			p_times2_num_args  = 1 )


	DATA gfcn_name         (p_times2) /'TIMES2'/
	DATA gfcn_version      (p_times2) / 1.0 /
	DATA gfcn_descr        (p_times2) / 'multiply by 2 (test routine)' /
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
	DATA gfcn_arg_name(p_times2_arg_start+0)/'X'/
	DATA gfcn_arg_units(p_times2_arg_start+0)/' '/
	DATA gfcn_arg_descr(p_times2_arg_start+0)/' '/
	DATA (gfcn_arg_extend_lo (i,p_times2_arg_start+0),i=1,4)
     .		  / 0, 0, 0, 0 /
	DATA (gfcn_arg_extend_hi (i,p_times2_arg_start+0),i=1,4)
     .		  / 0, 0, 0, 0 /
* ... merge the axes implied by argument 1 ?
	DATA (gfcn_axis_implied_from(i, p_times2_arg_start+0),i=1,4)
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
	DATA gfcn_arg_units(p_reshape_arg_start+0)/' '/
	DATA gfcn_arg_descr(p_reshape_arg_start+0)/'data to be reshaped'/
	DATA gfcn_arg_name(p_reshape_arg_start+1)/'B'/
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
	DATA gfcn_arg_units(p_zlayer_arg_start+0)/' '/
	DATA gfcn_arg_descr(p_zlayer_arg_start+0)/
     .		'variable on native Z axis'/
	DATA gfcn_arg_name(p_zlayer_arg_start+1)/'ZVALS'/
	DATA gfcn_arg_units(p_zlayer_arg_start+1)/' '/
	DATA gfcn_arg_descr(p_zlayer_arg_start+1)/
     .	'Z-value field corresponding to data points of V'/
	DATA gfcn_arg_name(p_zlayer_arg_start+2)/'ZAX'/
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


* template for next function ...
	INTEGER		p_next, p_next_arg_start, p_next_num_args 
	PARAMETER     ( p_next = p_zlayer + 1,
     .			p_next_arg_start = p_zlayer_arg_start +
     .					   p_zlayer_num_args,
     .			p_next_num_args  = 1 )

	END
