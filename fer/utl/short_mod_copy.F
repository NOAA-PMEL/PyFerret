	SUBROUTINE SHORT_MOD_COPY( memory, res_lo, res_hi, idim,
     .				   tcom_cx, tcom_mr, res_cx, res_mr)

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
* copy into the result over the range lo:hi from the component data
* the range lo:hi is assumed to be not more than Nmod points

* copying takes place in 2 stages ... with possible voids before,between,
* and after.

* the "t" in tcom_cx and tcom_mr refer to the fact that these are temporary
* information -- they may be modified

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V420: 11/95 - to accomodate modulo stripped on the main INTERP_STACK
*	12/95 - fixed bug for negative subscripts
* V533: *sh* 6/01 - added logic to cope with strides
* V541: *sh*  2/02 - support for subspan modulo axes
*	See NOTE in code below
* 560: *acm* 4/04 - stride/modulo fixes: use NINT to convert REAL*8 to integer

	include	'ferret.parm'
	include	'errmsg.parm'
	include	'interp_stack.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xprog_state.cmn'
	include	'xtext_info.cmn'

* calling argument declarations:
	INTEGER	res_lo, res_hi, idim, tcom_cx, tcom_mr, res_cx, res_mr
	REAL	memory( mem_blk_size, max_mem_blks )

* internal variable declarations:
	INTEGER CAXIS_MODLEN, MODULO_PARITY,
     .		npts, com_lo, com_hi, com_nshf, res_nshf,
     .		nshift, xlate, uplim, idel,
     .		cx_lo_sav, cx_hi_sav, mr_lo_sav, mr_hi_sav


* --- end of introductory code ---

* The diagram below illustrates the component and result regions relative
* to the modulo axis replications

* |++++++++|--------|--------|--------|--------|   <-- replicated modulo axis
*                |--------|			   <-- component received
*                                 |-----|	   <-- result requested

* The copy takes place in two steps.

* Step 1: The component is shifted so that the start of the result
*	  region lies within the component span (if possible)
*	  As much data as is possible contiguously is then copied
* |++++++++|--------|--------|--------|--------|   <-- replicated modulo axis
*                         |--------|		   <-- shifted component
*                                 |-----|	   <-- result requested

* Step 2: The component is shifted so that the start of the component
*	  region lies within the result span (if possible)
*	  As much data as is possible contiguously is then copied
* |++++++++|--------|--------|--------|--------|   <-- replicated modulo axis
*                                  |--------|	   <-- shifted component
*                                 |-----|	   <-- result requested

* save to restore at end
	cx_lo_sav = cx_lo_ss(tcom_cx, idim)
	cx_hi_sav = cx_hi_ss(tcom_cx, idim)
	mr_lo_sav = mr_lo_ss(tcom_mr, idim)
	mr_hi_sav = mr_hi_ss(tcom_mr, idim)

* initialize
	npts = CAXIS_MODLEN( idim, tcom_cx )	! length of modulo axis
	com_lo = cx_lo_ss(tcom_cx, idim)
	com_hi = MIN( cx_hi_ss(tcom_cx,idim), com_lo+npts-1 )	
	IF (cx_delta(idim,res_cx) .EQ. unspecified_val8) THEN
	   idel = 1
	ELSE
	   idel = NINT(cx_delta(idim,res_cx))
	ENDIF

* initialize the modulo shifting
	IF ( com_lo .GT. 0 ) THEN
	   com_nshf = (com_lo - 1) / npts
	ELSE
	   com_nshf = (com_lo / npts ) - 1
	ENDIF
	IF ( res_lo .GT. 0 ) THEN
	   res_nshf = (res_lo - 1) / npts
	ELSE
	   res_nshf = (res_lo / npts) - 1
	ENDIF
	nshift = res_nshf - com_nshf

* STEP 1
* shift component so the start of the result lies within the component range
	xlate = npts * nshift
	IF ( res_lo.GT.com_hi+xlate ) xlate = xlate - npts
	IF ( res_lo.LT.com_lo+xlate ) xlate = xlate + npts
	IF ( res_lo.GE.com_lo+xlate .AND. res_lo.LE.com_hi+xlate ) THEN
* ... copy the region (if the parity matches)
	   uplim = MIN( res_hi, com_hi+xlate )
	   cx_lo_ss(tcom_cx, idim) = res_lo
	   cx_hi_ss(tcom_cx, idim) = uplim
	   mr_lo_ss(tcom_mr, idim) = mr_lo_sav + xlate
	   mr_hi_ss(tcom_mr, idim) = mr_hi_sav + xlate
	   IF ( MODULO_PARITY(cx_lo_sav,             npts,idel)
     .	   .EQ. MODULO_PARITY(cx_lo_ss(tcom_cx,idim),npts,idel) ) THEN
	      CALL COPY_INTO( memory(1,mr_blk1(tcom_mr)), tcom_mr,
     .			      tcom_cx,
     .			      memory(1,mr_blk1(res_mr)), res_mr )
	   ENDIF
* NOTE:  it is KNOWN that sometimes this routine needlessle copies the same
*	 region of component into the same region of result at both STEP 1
*	 and STEP 2.  This has not been debugged because the consequences
*	 are negligible (*sh* 11/02)
	   IF (uplim .EQ. res_hi) GOTO 1000	! STEP 2 not needed
	ENDIF

* STEP 2
* shift component so the start of the component lies within the result range
	xlate = npts * nshift
	IF ( com_lo+xlate.GT.res_hi ) xlate = xlate - npts
	IF ( com_lo+xlate.LT.res_lo ) xlate = xlate + npts
	IF ( com_lo+xlate.GE.res_lo .AND. com_lo+xlate.LE.res_hi ) THEN
* ... copy the region
	   uplim = MIN( res_hi, com_hi+xlate )
	   cx_lo_ss(tcom_cx, idim) = com_lo + xlate
	   cx_hi_ss(tcom_cx, idim) = uplim
	   mr_lo_ss(tcom_mr, idim) = mr_lo_sav + xlate
	   mr_hi_ss(tcom_mr, idim) = mr_hi_sav + xlate
	   IF ( MODULO_PARITY(cx_lo_sav,             npts,idel)
     .	   .EQ. MODULO_PARITY(cx_lo_ss(tcom_cx,idim),npts,idel) ) THEN
	      CALL COPY_INTO( memory(1,mr_blk1(tcom_mr)), tcom_mr,
     .			   tcom_cx,
     .			   memory(1,mr_blk1(res_mr)), res_mr )
	   ENDIF
	ENDIF

* clean up - restore as before this call
 1000	cx_lo_ss(tcom_cx, idim) = cx_lo_sav
	cx_hi_ss(tcom_cx, idim) = cx_hi_sav
	mr_lo_ss(tcom_mr, idim) = mr_lo_sav
	mr_hi_ss(tcom_mr, idim) = mr_hi_sav

	RETURN	
	END	

