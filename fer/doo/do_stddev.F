	INTEGER FUNCTION DO_STDDEV (	idim, arg,
     .					com, com_mr, com_cx,
     .					res, res_mr, res_cx,
     .					box	)

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
* determine the variance of the given field along the indicated axis

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V230:  3/2/92
*        2/12 *kms* Add E and F dimensions (use nferdims in tmap_dims.parm)
* V702 *sh* 11/16 mods for trac #2428 - calculation in gathered fragments
*                 pass address of buffer area, where weights are accumulated
* v767 6/22 *acm* See issue 112: Use the INTEGER*8 routine CGRID_SIZE8

        IMPLICIT NONE
        include 'tmap_dims.parm'
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'interp_stack.parm'
	include	'xcontext.cmn'
	include	'xprog_state.cmn'
	include	'xvariables.cmn'
	include	'xdset_info.cmn_text'

* calling argument declarations:
	INTEGER	idim, com_mr, com_cx, res_mr, res_cx
	REAL	arg, com(*), res(*), box(*)

* internal variable declarations:
	LOGICAL TM_ITSA_DSG, stddev
	INTEGER*8 CGRID_SIZE8, size
	INTEGER TM_DSG_DSET_FROM_GRID, TM_DSG_NFEATURES,
     .		grid, dset, nfeatures, maxobs, orientation, mr_list(2)
	REAL	little_val, big_val


	stddev = .TRUE.

* diagnostic mode output: " doing --> VAR_NAME[x=lo:hi@TRN:n,D=#]"
	IF ( mode_diagnostic ) 	CALL DIAG_OP
     .				( 'doing', isact_class_trans, res_cx, idim )

* get size of result array to locate buffer areas just beyond
	size = CGRID_SIZE8(res_cx)
	
	grid = mr_grid( com_mr )

	IF (TM_ITSA_DSG(grid)) THEN 
* DSG ragged array
	   dset = TM_DSG_DSET_FROM_GRID(grid)
	   nfeatures = TM_DSG_NFEATURES(grid)
	   orientation = dsg_orientation(dset)
	   mr_list(1) = com_mr
	   mr_list(2) = res_mr
	   CALL NON_ARRAY_SUBSC( mr_list, 2 )
	   CALL DSG_ARRAY_SUBSC( mr_list, 2, orientation )

* ... get the length of the longest feature
	   CALL DSG_COORD_LIMS(dset, pdsg_row_size, little_val, big_val)
	   maxobs = big_val

	   CALL DO_VAR_DSG( idim, dset, nfeatures, maxobs,
     .			 com, com_mr, com_cx,
     .			 res, res_mr, res_cx,
     .			 box, stddev,
     .			 res(size+1),res(2*size+1) )

	ELSE


	   CALL DO_VAR_SUB( idim,
     .			 cx_lo_ss(com_cx,idim), cx_hi_ss(com_cx,idim),
     .			 com, com_mr, com_cx,
     .			 res, res_mr, res_cx,
     .			 box, stddev,
     .			 res(size+1),res(2*size+1) )

	ENDIF

	DO_STDDEV = ferr_ok

	RETURN
	END
