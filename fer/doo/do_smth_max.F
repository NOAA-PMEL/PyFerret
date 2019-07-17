	INTEGER FUNCTION DO_SMTH_MAX(	idim, arg,
     .					com, com_mr, com_cx,
     .					res, res_mr, res_cx,
     .					work	)

*
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration''s (NOAA) Pacific Marine Environmental Lab(PMEL),
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
* transform along axis idim by a running max filter, based on median one

* programmer - Ansley Manke
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* V622: 5/2010
*        2/12 *kms* Add E and F dimensions (use nferdims in tmap_dims.parm)

        IMPLICIT NONE
        include 'tmap_dims.parm'
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'interp_stack.parm'
	include	'xprog_state.cmn'
	include 'xcontext.cmn'
	include	'xdset_info.cmn_text'

* calling argument declarations:
	INTEGER	idim, com_mr, com_cx, res_mr, res_cx
	REAL	arg, work(*), com(*), res(*)

* internal variable declarations
	
	LOGICAL	TM_ITSA_DSG
	INTEGER	TM_DSG_DSET_FROM_GRID, TM_DSG_NFEATURES,
     .		wlen, grid, dset, nfeatures, orientation, mr_list(2)

* diagnostic mode output: " doing --> VAR_NAME[x=lo:hi@TRN:n,D=#]"
	IF ( mode_diagnostic ) 	CALL DIAG_OP
     .				( 'doing', isact_class_trans, res_cx, idim )

* initialize
	wlen = arg	! length of smoother
	grid = cx_grid(res_cx)
	
	IF (TM_ITSA_DSG(grid)) THEN 

* DSG ragged array
	   dset = TM_DSG_DSET_FROM_GRID(grid)
	   nfeatures = TM_DSG_NFEATURES(grid)
	   orientation = dsg_orientation(dset)
	   mr_list(1) = com_mr
	   mr_list(2) = res_mr
	   CALL NON_ARRAY_SUBSC( mr_list, 2 )
	   CALL DSG_ARRAY_SUBSC( mr_list, 2, orientation )

* now smooth this along the desired axis of the grid
	   CALL MAXSMOOTHER_DSG( idim, wlen, dset, nfeatures,
     .			com, com_mr, com_cx,
     .			res, res_mr, res_cx, work )

	ELSE

* now smooth this along the desired axis of the grid
	   CALL MAXSMOOTHER( idim, wlen,
     .			com, com_mr, com_cx,
     .			res, res_mr, res_cx, work )

	ENDIF

* success
	DO_SMTH_MAX = ferr_ok
	RETURN
	END

