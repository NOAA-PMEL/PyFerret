	SUBROUTINE IS_COUNTER_VAR( var, memory, *, status )
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
* create a counter array  "variable" in memory

* programmer - Ansley Manke, from is_const_var
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* Ferret v5.54 1/04

	include	'ferret.parm'
	include	'errmsg.parm'
	include	'xvariables.cmn'
	include	'xcontext.cmn'
	include	'xprog_state.cmn'
        include 'xcontrol.cmn'
        include 'xalgebra.cmn'

* calling argument declarations:
	INTEGER	var, status
	REAL	memory( mem_blk_size, max_mem_blks )

* internal variable declarations:
	INTEGER	mr, cx, csp1, idim
	REAL    counter_const

* initialize
	cx = is_cx( isp )
	IF ( status .NE. ferr_ok ) GOTO 5000

* diagnostic output
	IF (mode_diagnostic) CALL DIAGNOSTIC_OUT('counter_v',mr,point_to_mr)

* parse and insert the counter variable

* cs_command has set cs_phase to point to the next value of the loop controller
        
        csp1 = csp_cvar(var)
	counter_const = cs_phase( csp1 ) - cs_delta( csp1 )

* reserve and catalog space for the result
	CALL CREATE_TEMP_MEM_VAR( cx, mr, status )
	IF ( status .NE. ferr_ok ) GOTO 5000
	is_mr( isp ) = mr
        
        idim = 1

	cx_lo_ss    ( cx, idim ) = unspecified_int4
	cx_hi_ss    ( cx, idim ) = unspecified_int4
        cx_by_ss    ( idim, cx ) = .TRUE.

	mr_lo_ss    ( cx, idim ) = unspecified_int4
	mr_hi_ss    ( cx, idim ) = unspecified_int4

	cx_trans    ( idim, cx ) = trans_no_transform
	cx_trans_arg( idim, cx ) = bad_val4
	if (idim .EQ. t_dim) 
     .      cx_calendar(cx) = .FALSE.


* diagnostic output
	IF (mode_diagnostic) CALL DIAGNOSTIC_OUT('constan',mr,point_to_mr)

* insert the constant
	memory( 1, mr_blk1(mr) ) = counter_const

* set context data
	cx_bad_data( cx ) = bad_val4
	cx_data_set( cx ) = pdset_irrelevant
	cx_dset_gvn( cx ) = .FALSE.

* set the bad data flag to an unusual number
	mr_bad_data( mr ) = -2.1E34

* got the requested value
	RETURN 1

* error exit
 5000	RETURN
	END	
