	SUBROUTINE INIT_CXDSG(dset, cx, nfeatures)

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
* Summarize useful information about the given DSG context in COMMON CXDSG

* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V720 5/17 *sh*
* V73+ *acm* 12/17 INIT_CXDSG needs nfeatures as argument
* V741 6/18 *acm* Setting up to handle modulo longitudes

	include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'xtm_grid.cmn_text'
	include 'xdset_info.cmn_text'
	include 'xdyn_linemem.cmn_text'
	include 'xcontext.cmn'
	include 'xdsg_context.cmn'
	include 'xprog_state.cmn'
	include 'xvariables.cmn'
#include "tmap_dset.parm"

* calling argument declarations:
	INTEGER dset, cx, nfeatures

* internal variable declarations:
        INTEGER i, idim, var, inst_grid, lm, obs_dimlen
	REAL min_ww, max_ww

* initialize
	var = dsg_row_size_var(dset)
	inst_grid = ds_grid_number(var)

	obs_dimlen = dsg_obs_dimlen(dset)

* get the E axis limits as integers
	cxdsg_lo_e = cx_lo_ww(e_dim,cx) + .5   ! +.5 cuz ww lims
	cxdsg_hi_e = cx_hi_ww(e_dim,cx)

* Grid of the timeseries-station or trajectory variable in a timeseriesProfile or trajectorProfile set
	IF (cx_lo_ww(e_dim,cx) .EQ. unspecified_val8 .OR. 
     .      cx_lo_ww(e_dim,cx) .EQ. unspecified_val8) THEN
	   cxdsg_lo_e = 1
	   cxdsg_hi_e = nfeatures
	ENDIF
        
	cxdsg_constrain_e = cxdsg_lo_e.GT.1 .OR. cxdsg_hi_e.LT.nfeatures

* has the user constrained beyond the default WW limits of the dataset?
	cxdsg_no_coord_constraints = .TRUE.
	DO idim = 1, t_dim
	   
	   CALL DSG_COORD_LIMS(dset, idim, min_ww, max_ww)
	   IF (cx_lo_ww(idim,cx) .EQ. unspecified_val8) THEN
	      cx_lo_ww(idim,cx) = min_ww
	      cx_hi_ww(idim,cx) = max_ww
	   ENDIF
	   cxdsg_constrain(idim) = cx_lo_ww(idim,cx) .NE. min_ww
     .			      .OR. cx_hi_ww(idim,cx) .NE. max_ww
	   IF (cxdsg_constrain(idim)) cxdsg_no_coord_constraints = .FALSE.

* apply constraints, but longitude modulo in constraints handled later
	   var = dsg_coord_var(idim,dset)
	   IF (var.NE.int4_init .AND. idim.EQ.x_dim .AND. cx_lo_ww(idim,cx_cmnd).NE.unspecified_val8) THEN 
	      cxdsg_constrain_lo(idim) = MIN(cx_lo_ww(idim,cx), cx_lo_ww(idim,cx_cmnd) )
	      cxdsg_constrain_hi(idim) = MAX(cx_hi_ww(idim,cx), cx_hi_ww(idim,cx_cmnd) )
	   ELSE
	      cxdsg_constrain_lo(idim) = cx_lo_ww(idim,cx)
	      cxdsg_constrain_hi(idim) = cx_hi_ww(idim,cx)
	   ENDIF

	ENDDO

* which coordinates does this DSG provide?
	DO idim = 1, t_dim
	   var = dsg_coord_var(idim,dset)
	   cxdsg_has_coord(idim) =  var .NE. int4_init
	   IF (cxdsg_has_coord(idim)) THEN
	      cxdsg_bad_val(idim) = ds_bad_flag(var)
	      lm = dsg_loaded_lm(var)
	      cxdsg_coord_lm(idim) = lm
	      cxdsg_is_obs_coord(idim) = lm_size(lm) .EQ. obs_dimlen
	   ENDIF
	ENDDO

	RETURN
	END
