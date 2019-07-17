	SUBROUTINE APPLY_DSG_E_CONSTRAINTS(dset, cx)

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
* The E (=instances =features) ww limits on this context have been altered
* so re-compute the ww constraints in this context to reflect the restricted
* feature collection

* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V720 5/17 *sh*

	include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'xtm_grid.cmn_text'
	include 'xdset_info.cmn_text'
	include 'xdyn_linemem.cmn_text'
	include 'xcontext.cmn'
	include 'xdsg_context.cmn'
	include 'xvariables.cmn'

* calling argument declarations:
	INTEGER dset, cx

* internal variable declarations:
	INTEGER idim, rslm, base, ifeature, flen
	REAL	cmin(4), cmax(4), lolims(4), hilims(4)

* initialize
	rslm = dsg_loaded_lm(dsg_row_size_var(dset))
	DO idim = 1, 4
	  cmin(idim) = arbitrary_large_val8
	  cmax(idim) = arbitrary_small_val8
	ENDDO

* find the base index of the lowest unmasked E feature
	base = 0
	DO ifeature = 1, cxdsg_lo_e-1
	   base = base + dsg_linemem(rslm)%ptr(ifeature)
	ENDDO

* find the min and max coordinate limits including only features that are 
* NOT masked out by E limits
	DO ifeature = cxdsg_lo_e, cxdsg_hi_e
	   flen = dsg_linemem(rslm)%ptr(ifeature)
	   CALL	DSG_ONE_FEATURE_LIMS(dset, base, flen, ifeature,
     .				     lolims, hilims)
	   DO idim = 1, 4
	      IF (lolims(idim) .LT. cmin(idim)) cmin(idim) = lolims(idim)
	      IF (hilims(idim) .GT. cmax(idim)) cmax(idim) = hilims(idim)
	   ENDDO
	   base = base + flen
	ENDDO

* narrow the XYZT context lims as indicated by what we've just learned
	DO idim = 1, 4
	   IF (cx_lo_ww(idim,cx) .EQ. unspecified_val8) THEN
	      cx_lo_ww(idim,cx) = cmin(idim)
	   ELSE
	      IF (cmin(idim) .GT. cx_lo_ww(idim,cx))
     .				  cx_lo_ww(idim,cx) = cmin(idim)
	   ENDIF

	   IF (cx_hi_ww(idim,cx) .EQ. unspecified_val8) THEN
	      cx_hi_ww(idim,cx) = cmax(idim)
	   ELSE
	      IF (cmax(idim) .LT. cx_hi_ww(idim,cx))
     .				  cx_hi_ww(idim,cx) = cmax(idim)
	   ENDIF

	ENDDO

	RETURN
	END
