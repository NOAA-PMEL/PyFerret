	INTEGER FUNCTION DO_MIN( idim, arg,
     .				 com, com_mr, com_cx,
     .				 res, res_mr, res_cx,
     .				 buff	)

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
* find the minimum value of the component grid along axis idim

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  6/1/89
*	10/11/89 - modified array declarations using XMEM_SUBSC.CMN
*        2/12 *kms* Add E and F dimensions (use nferdims in tmap_dims.parm)
* V702 *sh* 11/16 this routine now performs accumulation only.  Initialization
*                 and finalizing called separately from IS_TRANS
* V71  *acm* 12/16 Send extra-buffer space that was allocated for the 
*                 accumulation/finalization, to store counts. To distinguish 
*                 all-missing data with a value of zero, from a zero min.
* v767 6/22 *acm* See issue 112: Use the INTEGER*8 routine CGRID_SIZE8

        IMPLICIT NONE
	include 'ferret.parm'
	include	'errmsg.parm'
	include	'tmap_dims.parm'
	include 'xcontext.cmn'
	include	'xdset_info.cmn_text'

* calling argument declarations:
	INTEGER	idim, com_mr, com_cx, res_mr, res_cx
	REAL	arg, buff(*),
     .          com( * ), res( * )

* internal variable declarations:
	LOGICAL TM_ITSA_DSG
	INTEGER*8 CGRID_SIZE8, size
	INTEGER TM_DSG_DSET_FROM_GRID, TM_DSG_NFEATURES,
     .		grid, dset, nfeatures, maxobs, mr_list(2)
	REAL	little_val, big_val

* initialize
	grid = cx_grid(res_cx)

* Get size of result array to locate buffer area just beyond
* Pass extra-buffer portion of res array to accumulate counts

	size = CGRID_SIZE8(res_cx)

	IF (TM_ITSA_DSG(grid)) THEN 
* DSG ragged array
	   dset = TM_DSG_DSET_FROM_GRID(grid)
	   nfeatures = TM_DSG_NFEATURES(grid)
* ... get the length of the longest feature
	   CALL DSG_COORD_LIMS(dset, pdsg_row_size, little_val, big_val)
	   maxobs = big_val

	   mr_list(1) = com_mr
	   mr_list(2) = res_mr
	   CALL NON_ARRAY_SUBSC( mr_list, 2 )
	   CALL DSG_ARRAY_SUBSC( mr_list, 2, dsg_orientation(dset) )

	   CALL DO_MIN_DSG (idim, arg, dset, nfeatures, maxobs,
     .                      com, com_mr, com_cx,
     .                      res, res_mr, res_cx,
     .                      res(size+1)	)
	ELSE

	   CALL DO_MIN_SUB (idim, arg,
     .                   com, com_mr, com_cx,
     .                   res, res_mr, res_cx,
     .                   res(size+1)	)

	ENDIF

	DO_MIN = ferr_ok

	RETURN
	END
