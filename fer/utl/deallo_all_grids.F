	SUBROUTINE DEALLO_ALL_GRIDS

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
* deallocate all scratch space from the grid COMMON stack
* (9/95) also deallocate any dynamic grids that were in progress
* (used following error conditions)

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  6/22/89
* V230:  7/24/92 - bug fix: loop index (igrid)  wasn't used inside loop
*                           grd_stk_ptr was!
* V420:  9/20/95 - deallocate dynamic grids, too
*Linux Port 1/97 *kob* - Added a preprocessor include for tmap_dset.parm
*			  because it needed a preprocessor.
#ifdef unix
	include 'tmap_dims.parm'
#	include "tmap_dset.parm"
	include	'xtm_grid.cmn_text'
	external xgt_grid_data
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'interp_stack.parm'
	include 'xprog_state.cmn'
	include 'xvariables.cmn'
	include 'xcontext.cmn'
	include 'xfr_grid.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE	'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:INTERP_STACK.PARM'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE 'FERRET_CMN:XFR_GRID.CMN'
#endif

* internal variable declarations
	INTEGER	igrid, i, cx, status


* deallocate the temporary grids from the upper grid stack
* (see documentation in tm_allo_dyn_grid)
* this may occasionally also deallocate associated dynamic axes
	DO 100 igrid = grd_stk_ptr, max_grids-1
	   CALL DEALLO_GRID(status)
	   IF ( status .NE. ferr_ok ) THEN	
	      grd_stk_ptr = max_grids
	      GOTO 150			! internal error!!
	   ENDIF
 100	CONTINUE

* Deallocate any dynamic grid (and associated memory vars) the need for which
* has been terminated by this error condition.
* A record of all of the dynamic grids in progress is implicitly available
* on the interpretation stack.
 150 	DO 200 cx = cx_stack_ptr, cx_stack_ptr_base, -1
	   IF ( cx_has_impl_grid(cx) .AND. cx_grid(cx).GT.max_grids) THEN
	      CALL DEALLO_DYN_GRID(cx_grid(cx))
	      cx_has_impl_grid(cx) = .FALSE.
	   ENDIF
 200	CONTINUE

	RETURN
	END
