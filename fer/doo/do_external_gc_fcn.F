      SUBROUTINE DO_EXTERNAL_GC_FCN( memory,
     .				fcn, num_com, com_cx,
     .				com1, mr1,
     .				com2, mr2,
     .				com3, mr3,
     .				com4, mr4,
     .				com5, mr5,
     .				com6, mr6,
     .				com7, mr7,
     .				com8, mr8,
     .				com9, mr9,
     .				res, mres,
     .                          status )

* execute a grid changing function

* the addition (or major modification) of an internal grid-changing function
* requires changes to these routines:
*	grid_chg_fcns.parm	- parameters to number the functions
*	XGRID_CHG_FCNS_DATA	- description of the function and args
*	DO_INTERNAL_GC_FCN	- code to execute the function
* (opt)	GCF_GET_IMPOSED_AXES	- return the "imposed" axes, if any
* (opt)	GCF_IMPOSE_RESULT_LIMS	- fill in limits of "imposed" axes, if any

* V450:  1/97 - new, based on operate.F, but for grid changing functions
*        7/97 - complete the ZAXREPLACE function
*       7/25/97 - increase number of args from 4 to 9
*               - add function PPLGRID

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/xvariables.cmn'
	include	'ferret_cmn/xmem_subsc.cmn'
	include 'ferret_cmn/xcontext.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XMEM_SUBSC.CMN'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
#endif

* calling argument declarations:
	INTEGER	fcn, num_com, com_cx(9), status,
     .		mr1, mr2, mr3, mr4, mr5, mr6, mr7, mr8, mr9, mres
        REAL	memory( mem_blk_size, max_mem_blks )
	REAL    com1( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit ),
     .		com2( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,m2lot:m2hit ),
     .		com3( m3lox:m3hix,m3loy:m3hiy,m3loz:m3hiz,m3lot:m3hit ),
     .		com4( m4lox:m4hix,m4loy:m4hiy,m4loz:m4hiz,m4lot:m4hit ),
     .		com5( m5lox:m4hix,m5loy:m5hiy,m5loz:m5hiz,m5lot:m5hit ),
     .		com6( m6lox:m4hix,m6loy:m6hiy,m6loz:m6hiz,m6lot:m6hit ),
     .		com7( m7lox:m4hix,m7loy:m7hiy,m7loz:m7hiz,m7lot:m7hit ),
     .		com8( m8lox:m4hix,m8loy:m8hiy,m8loz:m8hiz,m8lot:m8hit ),
     .		com9( m9lox:m4hix,m9loy:m9hiy,m9loz:m9hiz,m9lot:m9hit ),
     .		 res( m10lox:m10hix,m10loy:m10hiy,
     .					  m10loz:m10hiz,m10lot:m10hit )

* internal variable declarations:
        INTEGER loss_array(9,4), hiss_array(9,4), 
     .       mr_arg_offset_array(10), mr_res_offset
        REAL bad_flag_array(10)

* internal parameter declarations:
	REAL*8          pdays_by_1900
	PARAMETER     ( pdays_by_1900 = 59958230400.0 / (60.*60.*24.) )

        status = ferr_ok

! JC_TODO:  could I use equivalences to make this easier?
       loss_array(1,1) = m1lox
       loss_array(1,2) = m1loy
       loss_array(1,3) = m1loz
       loss_array(1,4) = m1lot 
       hiss_array(1,1) = m1hix
       hiss_array(1,2) = m1hiy
       hiss_array(1,3) = m1hiz
       hiss_array(1,4) = m1hit

       loss_array(2,1) = m2lox
       loss_array(2,2) = m2loy
       loss_array(2,3) = m2loz
       loss_array(2,4) = m2lot 
       hiss_array(2,1) = m2hix
       hiss_array(2,2) = m2hiy
       hiss_array(2,3) = m2hiz
       hiss_array(2,4) = m2hit

       loss_array(3,1) = m3lox
       loss_array(3,2) = m3loy
       loss_array(3,3) = m3loz
       loss_array(3,4) = m3lot 
       hiss_array(3,1) = m3hix
       hiss_array(3,2) = m3hiy
       hiss_array(3,3) = m3hiz
       hiss_array(3,4) = m3hit

       loss_array(4,1) = m4lox
       loss_array(4,2) = m4loy
       loss_array(4,3) = m4loz
       loss_array(4,4) = m4lot 
       hiss_array(4,1) = m4hix
       hiss_array(4,2) = m4hiy
       hiss_array(4,3) = m4hiz
       hiss_array(4,4) = m4hit

       loss_array(5,1) = m5lox
       loss_array(5,2) = m5loy
       loss_array(5,3) = m5loz
       loss_array(5,4) = m5lot 
       hiss_array(5,1) = m5hix
       hiss_array(5,2) = m5hiy
       hiss_array(5,3) = m5hiz
       hiss_array(5,4) = m5hit

       loss_array(6,1) = m6lox
       loss_array(6,2) = m6loy
       loss_array(6,3) = m6loz
       loss_array(6,4) = m6lot 
       hiss_array(6,1) = m6hix
       hiss_array(6,2) = m6hiy
       hiss_array(6,3) = m6hiz
       hiss_array(6,4) = m6hit

       loss_array(7,1) = m7lox
       loss_array(7,2) = m7loy
       loss_array(7,3) = m7loz
       loss_array(7,4) = m7lot 
       hiss_array(7,1) = m7hix
       hiss_array(7,2) = m7hiy
       hiss_array(7,3) = m7hiz
       hiss_array(7,4) = m7hit

       loss_array(8,1) = m8lox
       loss_array(8,2) = m8loy
       loss_array(8,3) = m8loz
       loss_array(8,4) = m8lot 
       hiss_array(8,1) = m8hix
       hiss_array(8,2) = m8hiy
       hiss_array(8,3) = m8hiz
       hiss_array(8,4) = m8hit

       loss_array(9,1) = m9lox
       loss_array(9,2) = m9loy
       loss_array(9,3) = m9loz
       loss_array(9,4) = m9lot 
       hiss_array(9,1) = m9hix
       hiss_array(9,2) = m9hiy
       hiss_array(9,3) = m9hiz
       hiss_array(9,4) = m9hit


       mr_arg_offset_array(1) = ( mr_blk1(mr1) - 1 ) * mem_blk_size
       mr_arg_offset_array(2) = ( mr_blk1(mr2) - 1 ) * mem_blk_size
       mr_arg_offset_array(3) = ( mr_blk1(mr3) - 1 ) * mem_blk_size
       mr_arg_offset_array(4) = ( mr_blk1(mr4) - 1 ) * mem_blk_size
       mr_arg_offset_array(5) = ( mr_blk1(mr5) - 1 ) * mem_blk_size
       mr_arg_offset_array(6) = ( mr_blk1(mr6) - 1 ) * mem_blk_size
       mr_arg_offset_array(7) = ( mr_blk1(mr7) - 1 ) * mem_blk_size
       mr_arg_offset_array(8) = ( mr_blk1(mr8) - 1 ) * mem_blk_size
       mr_arg_offset_array(9) = ( mr_blk1(mr9) - 1 ) * mem_blk_size
       mr_arg_offset_array(num_com+1) = ( mr_blk1(mres) - 1 ) * 
     .      mem_blk_size

       bad_flag_array(1) = mr_bad_data ( mr1 )
       bad_flag_array(2) = mr_bad_data ( mr2 )
       bad_flag_array(3) = mr_bad_data ( mr3 )
       bad_flag_array(4) = mr_bad_data ( mr4 )
       bad_flag_array(5) = mr_bad_data ( mr5 )
       bad_flag_array(6) = mr_bad_data ( mr6 )
       bad_flag_array(7) = mr_bad_data ( mr7 )
       bad_flag_array(8) = mr_bad_data ( mr8 )
       bad_flag_array(9) = mr_bad_data ( mr9 )
       bad_flag_array(num_com+1) = mr_bad_data ( mres )

       mr_res_offset = ( mr_blk1(mres) - 1 ) * mem_blk_size

       call efcn_compute(fcn, num_com, com_cx,
     .      bad_flag_array, mr_arg_offset_array, mr_res_offset, 
     .      memory)

 
* error exits
 5000	RETURN
	END

