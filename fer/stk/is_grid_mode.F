	SUBROUTINE IS_GRID_MODE( status )

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
* set up interp stack to begin determining the grid of an algebraic expression

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  6/28/89
* V420:  1/95 - need to init cx_delta since it is used by dynamic axes
* V491:  12/97 - bug fix: need to initialize 
* V510: *sh* 4/00 - initialize the grid that is allocated to 

* calling argument declarations:
	INTEGER	status

* internal variable declarations:
	INTEGER cx, big_cx, idim

#ifdef unix
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'interp_stack.parm'
#       include "tmap_dset.parm"
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xprog_state.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
	INCLUDE	'FERRET_CMN:INTERP_STACK.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XPROG_STATE.CMN'
#endif

* carve a space on the context stack for a "bland" (1:1,1:1,1:1,1:1) context
	cx = cx_stack_ptr
	cx_grid(cx) = unspecified_int4		! for pseudo_vars in pass 1
	DO 100 idim = 1, 4
	  cx_lo_ss(cx,idim) = 1
	  cx_hi_ss(cx,idim) = 1
	  cx_by_ss(idim,cx) = .TRUE.
	  cx_trans(idim,cx) = trans_no_transform
	  cx_given(idim,cx) = .FALSE.
	  cx_lo_ww(idim,cx) = unspecified_val8
	  cx_hi_ww(idim,cx) = unspecified_val8
	  cx_delta(idim,cx) = unspecified_val8
 100	CONTINUE

* diagnostic alert
	IF ( mode_diagnostic )
     .			CALL DIAGNOSTIC_OUT( 'getgrid', cx, point_to_cx )

* initialize a level on the interpretion stack for the activity
	is_phase( isp ) = 0

* allocate another context for the "merged" grid
	CALL STACK_PTR_UP( cx_stack_ptr, max_context, status )
	IF ( status .NE. ferr_ok ) GOTO 5000
	big_cx = cx_stack_ptr
	is_big_cx( isp ) = big_cx
	CALL ALLO_GRID( cx_grid(big_cx), status )
	IF ( status .NE. ferr_ok ) GOTO 5000
	CALL INIT_GRID( cx_grid(big_cx), char_init16, mnormal )

* set flag indicating that the first component grid has not yet been encountered
	is_sum( isp ) = isalg_transfer
	RETURN

* error exit
 5000	RETURN
	END	

