	SUBROUTINE INIT_EMPTY_CONTEXT ( cx )

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
* set up a "null" context suitable to merge other contexts on top of

* V450: 1/97 - part of changes to accomodate grid-changing functions
* 9/97 *kob* - initialize cx_calendar as well
* V530:  8/00 *sh* - initialize data type to unspecified

#ifdef unix
	include 'ferret.parm'
	include 'xcontext.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
#endif

* calling argument declarations:
	INTEGER cx

* internal variable declarations
	INTEGER	idim

* all contexts contain axis region information
	DO 100 idim = 1,4
	   cx_lo_ss ( cx, idim )   = unspecified_int4
	   cx_hi_ss ( cx, idim )   = unspecified_int4
	   cx_lo_ww ( idim, cx )   = unspecified_val8
	   cx_hi_ww ( idim, cx )   = unspecified_val8
	   cx_trans ( idim, cx )   = trans_no_transform
	   cx_by_ss ( idim, cx )   = .TRUE.
	   cx_given ( idim, cx )   = .TRUE.		! <== (*)
	   cx_trans_arg(idim, cx)  = bad_val4
	   cx_delta ( idim, cx )   = unspecified_val8
	   cx_regrid_trans(idim,cx)=unspecified_int4
 100	CONTINUE

	cx_data_set( cx ) = unspecified_int4
	cx_dset_gvn( cx ) = .TRUE.			! <== (*)
	cx_category( cx ) = unspecified_int4
	cx_variable( cx ) = unspecified_int4
	cx_grid    ( cx ) = unspecified_int4
	cx_type    ( cx ) = unspecified_int4
!	cx_ovsiz   ( cx ) = JC_TODO:  what should this be?
	cx_bad_data( cx ) = bad_val4

	cx_unstand_grid ( cx ) = .FALSE.
	cx_impl_grid    ( cx ) = unspecified_int4

* kob 9/97 - need to initialize cx_calendar as well
		 IF (idim .eq. t_dim) 
     .                cx_calendar(cx) = .FALSE.

* (*) initialized to TRUE so that ANDing with another context will yield
*	TRUE only if "given" in that context

	RETURN
	END
