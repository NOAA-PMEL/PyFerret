	SUBROUTINE IS_PSEUDO(  *, status )

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
* create a gridded "pseudo-variable" (eg. "X" or "K")

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 3/2/87
* V200:  6/26/89 - 4-D symmetric version
*	10/16/89 - changed adj. array bounds for MAKE_PSEUDO_VAR call
*	11/30/89 - added diagnostic output
* V312: 5/94 - array "memory" as a calling argument
*       *acm* 3/12 6D Ferret (common uses nferdims in tmap_dims.parm)
*       *acm* 4/12 Add new MODE FOURDEE to allow E,F,M,N,etc as user-vars
*       *sh*  2/17 pre-check for 99999999 in order to issue nicer error message
* v74   *acm* 3/18 Introduce parameter abstractax_len replacing 99999999
* Vxx *sh* 5/17 DSG support

        IMPLICIT NONE
	include	'tmap_dims.parm'
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'interp_stack.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xprog_state.cmn'
	include	'xalgebra.cmn'

* calling argument declarations:
	INTEGER	status

* internal variable declarations:
	LOGICAL TM_ITSA_DSG, its_dsg
	INTEGER	CX_DIM_LEN, mr, cx, idim, pvlen, lo, hi

* --- end of introductory code ---
* initialize
	cx = is_cx( isp )

* remove regridding flag 
* for pseudo-vars it just means the grid was explicitly specified - there
* is no regridding)
	cx_unstand_grid( cx ) = .FALSE.

* Get the direction of the pseudo-variable.
	idim = MOD( cx_variable(cx)-1, nferdims ) + 1

* check for DSG
	its_dsg = idim .NE. f_dim .AND. TM_ITSA_DSG(cx_grid(cx))

* pre-check to make sure limits were given
	pvlen = CX_DIM_LEN(idim, cx)
	IF (pvlen .EQ. abstractax_len) GOTO 5100

* reserve and catalog space for the result
	CALL CREATE_TEMP_MEM_VAR( cx, mr, status )
	IF ( status .NE. ferr_ok ) GOTO 5000
	is_mr( isp ) = mr

* diagnostic output
	IF (mode_diagnostic) CALL DIAGNOSTIC_OUT('pseudo ',mr,point_to_mr)

* fill the space with the desired data
	IF (its_dsg) THEN
	   lo = 1
	   hi = mr_size(mr)
	ELSE
	   lo = mr_lo_ss(mr,idim)
	   hi = mr_hi_ss(mr,idim)
	ENDIF

	CALL MAKE_PSEUDO_VAR( cx, idim, memry(mr)%ptr, mr, lo, hi)

	RETURN 1

* error exit
 5000	RETURN

 5100	CALL ERRMSG(ferr_limits,status,
     .         "Limits must be provided for pseudo-variable.", *5000)
     .		
	END	
