	SUBROUTINE IS_CONSTANT( memory, *, status )

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
* create a constant value "variable" in memory

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 7/28/87
* revision 0.1 - 6/23/88 - set bad data flag to -2.1E34 so -1.E34 is valid
* V200:  6/26/89 - just a single value
*	11/30/89 - added diagnostic output
* V312: 5/94 - array "memory" as a calling argument
* V554  1/04 - *acm* remove VAX common statements.

	include	'ferret.parm'
	include	'errmsg.parm'
	include	'interp_stack.parm'
	include	'xvariables.cmn'
	include	'xcontext.cmn'
	include	'xprog_state.cmn'

* calling argument declarations:
	INTEGER	status
	REAL	memory( mem_blk_size, max_mem_blks )

* internal variable declarations:
	INTEGER	mr, cx

* equivalences
	REAL	cx_value( max_context )
	EQUIVALENCE ( cx_variable, cx_value )

* --- end of introductory code ---
* initialize
	cx = is_cx( isp )

* reserve and catalog space for the result
	CALL CREATE_TEMP_MEM_VAR( cx, mr, status )
	IF ( status .NE. ferr_ok ) GOTO 5000
	is_mr( isp ) = mr

* diagnostic output
	IF (mode_diagnostic) CALL DIAGNOSTIC_OUT('constan',mr,point_to_mr)

* insert the constant
	memory( 1, mr_blk1(mr) ) = cx_value(cx)

* set the bad data flag to an unusual number
	mr_bad_data( mr ) = -2.1E34

	RETURN 1

* error exit
 5000	RETURN
	END	
