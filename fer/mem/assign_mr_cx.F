	SUBROUTINE ASSIGN_MR_CX( cx, mr )

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
* fill a slot in the memory variable table with context information

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* V200:  6/26/89 - extracted from ASSIGN_VARIABLE

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xvariables.cmn'
	include 'ferret_cmn/xcontext.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
#endif

* calling argument declarations:
	INTEGER		cx, mr

* internal variable declarations:
	INTEGER		idim, trans, grid
	REAL*8		TM_WORLD

* initialize
	grid = cx_grid( cx ) 

* background info
	mr_data_set ( mr )	= cx_data_set( cx )
	mr_category ( mr )	= cx_category( cx )
	mr_variable ( mr )	= cx_variable( cx )
	mr_grid     ( mr )	= grid
	mr_bad_data ( mr )	= cx_bad_data( cx )
	mr_unstand_grid( mr )   = cx_unstand_grid( cx )

* region, transformations and regridding transforms
	DO 100 idim = 1,4
	   trans = cx_trans( idim, cx )
	   mr_trans( idim, mr )   = trans
	   mr_lo_ss( mr, idim )   = cx_lo_ss( cx, idim )
	   mr_hi_ss( mr, idim )   = cx_hi_ss( cx, idim )
	   mr_trans_arg(idim, mr) = cx_trans_arg(idim, cx)
	   mr_regrid_trans( idim, mr ) = cx_regrid_trans( idim, cx )

* round out to world coordinate box edges except in some special cases:
*   i) indefinite integrals ( since they depend on exact integration limits)
*  ii) transforms (except MIN and MAX) that reduce to a point
* iii) user-defined variables that are based on transformed quantities
*      (e.g. DEFINE UAVE = U[K=1:3@AVE])
	   IF (  trans .EQ. trans_integ_indef
     .    .OR. ( trans .GT. trans_compress_code
     .      .AND.trans .NE. trans_max .AND. trans.NE.trans_min )
     .	  .OR. ( cx_lo_ss(cx,idim) .EQ. unspecified_int4 ) ) THEN
* ... catalog with exact world coordinate limits
	      mr_lo_ww( idim, mr ) = cx_lo_ww( idim, cx )
	      mr_hi_ww( idim, mr ) = cx_hi_ww( idim, cx )
	   ELSE
* ... round out limits out to indicate full span of end grid boxes
	      mr_lo_ww(idim, mr) = TM_WORLD( cx_lo_ss(cx,idim),
     .					     grid,
     .					     idim, box_lo_lim	)
	      mr_hi_ww(idim, mr) = TM_WORLD( cx_hi_ss(cx,idim),
     .					     grid,
     .					     idim, box_hi_lim	)
	   ENDIF
 100	CONTINUE

	RETURN
	END
