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
     .       mr1, mr2, mr3, mr4, mr5, mr6, mr7, mr8, mr9, mres
        REAL memory( mem_blk_size, max_mem_blks )
	REAL com1( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit ),
     .       com2( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,m2lot:m2hit ),
     .       com3( m3lox:m3hix,m3loy:m3hiy,m3loz:m3hiz,m3lot:m3hit ),
     .       com4( m4lox:m4hix,m4loy:m4hiy,m4loz:m4hiz,m4lot:m4hit ),
     .       com5( m5lox:m4hix,m5loy:m5hiy,m5loz:m5hiz,m5lot:m5hit ),
     .       com6( m6lox:m4hix,m6loy:m6hiy,m6loz:m6hiz,m6lot:m6hit ),
     .       com7( m7lox:m4hix,m7loy:m7hiy,m7loz:m7hiz,m7lot:m7hit ),
     .       com8( m8lox:m4hix,m8loy:m8hiy,m8loz:m8hiz,m8lot:m8hit ),
     .       com9( m9lox:m4hix,m9loy:m9hiy,m9loz:m9hiz,m9lot:m9hit ),
     .       res( m10lox:m10hix,m10loy:m10hiy,
     .       m10loz:m10hiz,m10lot:m10hit )

* internal variable declarations:
        INTEGER mr_arg_offset_array(10)
        REAL bad_flag_array(10)

* internal parameter declarations:
	REAL*8          pdays_by_1900
	PARAMETER     ( pdays_by_1900 = 59958230400.0 / 
     .       (60.*60.*24.) )

        status = ferr_ok

       mr_arg_offset_array(1)  = ( mr_blk1(mr1) - 1 ) * mem_blk_size
       mr_arg_offset_array(2)  = ( mr_blk1(mr2) - 1 ) * mem_blk_size
       mr_arg_offset_array(3)  = ( mr_blk1(mr3) - 1 ) * mem_blk_size
       mr_arg_offset_array(4)  = ( mr_blk1(mr4) - 1 ) * mem_blk_size
       mr_arg_offset_array(5)  = ( mr_blk1(mr5) - 1 ) * mem_blk_size
       mr_arg_offset_array(6)  = ( mr_blk1(mr6) - 1 ) * mem_blk_size
       mr_arg_offset_array(7)  = ( mr_blk1(mr7) - 1 ) * mem_blk_size
       mr_arg_offset_array(8)  = ( mr_blk1(mr8) - 1 ) * mem_blk_size
       mr_arg_offset_array(9)  = ( mr_blk1(mr9) - 1 ) * mem_blk_size
* the result should always have index MAX_EF_ARGS+1
       mr_arg_offset_array(10) = ( mr_blk1(mres) - 1 ) * mem_blk_size

       bad_flag_array(1)  = mr_bad_data ( mr1 )
       bad_flag_array(2)  = mr_bad_data ( mr2 )
       bad_flag_array(3)  = mr_bad_data ( mr3 )
       bad_flag_array(4)  = mr_bad_data ( mr4 )
       bad_flag_array(5)  = mr_bad_data ( mr5 )
       bad_flag_array(6)  = mr_bad_data ( mr6 )
       bad_flag_array(7)  = mr_bad_data ( mr7 )
       bad_flag_array(8)  = mr_bad_data ( mr8 )
       bad_flag_array(9)  = mr_bad_data ( mr9 )
* the result should always have index MAX_EF_ARGS+1
       bad_flag_array(10) = mr_bad_data ( mres )


       call efcn_compute(fcn, num_com, com_cx, mres, bad_flag_array, 
     .      mr_arg_offset_array, memory)

 
* error exits
 5000	RETURN
	END

