	SUBROUTINE SETUP_SAMPLE(  cx_indices, mr_indices,
     .				 ifcn, big_cx, status )

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


* This routine sets up for the SAMPLE* functions
* When the user issues SAMPLEI({1,5,3,3,8},my_var) this routine will
* sort and store the given indices and prepare for isact_sample activities 

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* V510: 8/99 *sh*
*       *acm* 3/12 6D Ferret (common uses nferdims in tmap_dims.parm)
* v767 6/22 *acm* See issue 112: Use the INTEGER*8 routine CGRID_SIZE8

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'interp_stack.parm'
	include	'grid_chg_fcns.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xprog_state.cmn'

* calling argument declarations:
	INTEGER	cx_indices, ifcn, mr_indices, big_cx, status

* ... interpretation stack setup at the time this is called ...
*	stack level		stack contents
*	isp+1	<--	level to request individual sample chunks
*	 isp	<--	current stack level (for sampling )

* internal variable declarations:
	LOGICAL TM_ITSA_DSG
	INTEGER n, dims(nferdims), ndim, hold_isp, cx_holder,
     .		min_ss, max_ss, res_grid,
     .		mr_sorted_indices, mr_index_map, axis, nvalid
	INTEGER*8 CGRID_SIZE8, n8

* --- end of introductory code ---

* initialize
	hold_isp  = isp
	cx_holder = is_cx( hold_isp )
	res_grid  = cx_grid( cx_holder )

	IF (TM_ITSA_DSG( cx_grid(cx_holder) )) GOTO 5020

* Check that index list is 1D.
	CALL GET_CX_DIMS(cx_indices, ndim, dims)
	IF (ndim .GT. 1) GOTO 5010

* create memory managed work arrays for the sorted_indices and index_map
	n8 = CGRID_SIZE8(cx_indices)
	CALL GET_WORK_MR(n8,mr_sorted_indices, status)
	IF ( status .NE. ferr_ok ) GOTO 5000
	CALL GET_WORK_MR(n8,mr_index_map, status)
	IF ( status .NE. ferr_ok ) GOTO 5000
!	mr_hi_ss(mr_index_map,x_dim) = n	! done in GET_WORK_MR

* extract the 1D list of indices to be obtained
        CALL EXTRACT_LINE ( cx_indices,
     .                      memry(mr_indices)%ptr,
     .                      mr_indices,
     .                      memry(mr_sorted_indices)%ptr,
     .                      dims,
     .                      ndim,
     .                      n,
     .                      status      )
        IF ( status .NE. ferr_ok ) GOTO 5000

* which axis is the sampling along?
	axis = ifcn - p_samplei + 1
	is_axis( hold_isp ) = axis

* sort the list of indices and store with a map of the sort
	CALL GRID_SUBSCRIPT_EXTREMES(min_ss, max_ss, res_grid, axis)
	CALL SORT_LIST(n, min_ss, max_ss,  
     .			   mr_bad_data(mr_indices),
     .			   mr_bad_data(mr_sorted_indices),
     .			   memry(mr_sorted_indices)%ptr,
     .			   memry(mr_index_map)%ptr, nvalid )

* pass the number of valid points through the (irrelevant) T axis indices
* of the (one-dimensional) mr_index_map array
	mr_lo_ss( mr_index_map, t_dim ) = nvalid
	mr_hi_ss( mr_index_map, t_dim ) = nvalid

* encode both the sorted indices and the index map into the hold_isp stack
	is_sum( hold_isp )  = mr_sorted_indices + 10000*mr_index_map

* tell the interp stack what activity we are about
	is_act ( hold_isp ) = isact_sample

* tell IS_SAMPLE that we're just beginning
	is_phase( hold_isp ) = 0

* setup the next stack level to grab the individual chunks
	CALL PUSH_INTERP_STACK( hold_isp, status )
	IF ( status .NE. ferr_ok) GOTO 5000
	is_obj( isp ) = hold_isp	! claimed by this level  (?ok?)

* allocate a context that will be modified for each chunk requested
	CALL STACK_PTR_UP( cx_stack_ptr, max_context, status )
        IF ( status .NE. ferr_ok ) RETURN
	CALL TRANSFER_CONTEXT( cx_holder, cx_stack_ptr )
	is_cx( isp ) = cx_stack_ptr

* now drop the isp pointer back down one, so that we will enter IS_SAMPLE
* in the same state that we would had we just grabbed the first chunk
* of sampled data
	isp = hold_isp

* the grid of the result is ABSTRACT on axis "axis". Need to correspondingly
* modify the context of the result
	cx_grid(cx_holder)     = cx_grid(big_cx)
!?	IF (cx_grid(cx_holder) .GT. max_grids)
!?     .			CALL TM_USE_DYN_GRID(cx_grid(cx_holder))
	cx_category(cx_holder) = cat_temp_var
	cx_has_impl_grid(cx_holder) = .FALSE.
	cx_lo_ss(cx_holder,axis) = 1
	cx_hi_ss(cx_holder,axis) = n
	cx_by_ss(axis,cx_holder) = .TRUE.
	CALL FLESH_OUT_AXIS( axis, cx_holder, status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* other axes of the holder are not yet known. Result memory
* space will be allocated after the first "chunk" has been obtained
	is_mr( hold_isp ) = unspecified_int4

* diagnostic mode output: " doing gather VAR_NAME[x=lo:hi@TRN:n,D=#]"
	IF ( mode_diagnostic ) 	CALL DIAG_OP
     .		('doing', isact_class_sample, cx_holder, axis)

* success
	RETURN

* error exits
 5000	RETURN
 5010	CALL ERRMSG( ferr_invalid_command, status,
     .          'list of indices supplied to SAMPLE* must be 1D ',
     .          *5000)
 5020	CALL ERRMSG(ferr_invalid_command, status,
     .   'SAMPLE functions do not apply to data from '//
     .   'Discrete Sampling Geometries (DSG) dataset. '//
     .   'Use constraints or masks.', *5000 )
	END

