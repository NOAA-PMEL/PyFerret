	SUBROUTINE ADDL_CX_DOC( joint_cx, addl_cx, overlay, flags )

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
* joint context for a plot.  See GRP_CX_DOC

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* V200:	  2/2/90
* V230:  7/22/92 - added overlay argument for cases where the original plot
*                  was unspecified on an axis but the overlay is not
*                e.g. SHADE T_INDEPENDENT  CONTOUR/OVER T_DEPENDENT
* *kob* 11/96   - Linux port.  Linux didn't like tabs between "include"
*                 and what was being included.  removed them
#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xcontext.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
#endif

* calling argument declarations
	INTEGER joint_cx, addl_cx
	LOGICAL	overlay, flags(0:10)

* internal variable declarations
	INTEGER	idim, CX_DIM_LEN

* document the grid in the key ??
* ( as of FERRET v2.00 this is governed entirely by cx_unstand_grid )
!	flags(pdoc_regrid) = cx_unstand_grid(addl_cx)
!     .			 .AND. cx_unstand_grid(addl_cx)


* document the data set in the key ??
* (take any "real" data set over pdset_irrelevant)
	IF ( cx_data_set(joint_cx) .EQ. pdset_irrelevant ) THEN
	   cx_data_set(joint_cx) = cx_data_set(addl_cx)
           IF ( overlay ) THEN   ! 7/92
              flags(pdoc_dset) = .TRUE.
              flags(0) = .TRUE.
           ENDIF
	ELSEIF ( cx_data_set(addl_cx) .NE. pdset_irrelevant ) THEN
	   IF ( cx_data_set(addl_cx) .NE. cx_data_set(joint_cx) ) THEN
	      cx_data_set(joint_cx) = unspecified_int4
	      flags(pdoc_dset) = .TRUE.
	      flags(0) = .TRUE.
	   ENDIF
	ENDIF

* check each axis for documentation needs
* note: unspec_val8_2 indicates differing limits between contexts
*	unspecified_val8 indicates irrelevant (normal) limits
	DO 100 idim = 1, 4
	   IF ( cx_lo_ww(idim,addl_cx)  .EQ. unspecified_val8 ) GOTO 100
	   IF ( cx_lo_ww(idim,joint_cx) .EQ. unspec_val8_2    ) GOTO 100
	   IF ( cx_lo_ww(idim,joint_cx) .EQ. unspecified_val8 ) THEN
	      CALL TRANSFER_AXIS( idim, addl_cx, joint_cx )
              IF ( overlay ) THEN     ! 7/92
                 flags(idim) =  CX_DIM_LEN( idim, addl_cx ) .EQ. 1
                 IF ( flags(idim) ) flags(0) = .TRUE.
              ENDIF
	   ELSEIF(  cx_lo_ww(idim,joint_cx) .NE. cx_lo_ww(idim,addl_cx)
     .		.OR.cx_hi_ww(idim,joint_cx) .NE. cx_hi_ww(idim,addl_cx) ) THEN
	      cx_lo_ww(idim,joint_cx) = unspec_val8_2
	      cx_hi_ww(idim,joint_cx) = unspec_val8_2
	      flags(idim) = .TRUE.
	      flags(idim+pdoc_offset_xform) = .TRUE.
	      flags(0) = .TRUE.
	   ENDIF
	   IF( cx_trans_arg(idim,joint_cx) .NE. cx_trans_arg(idim,addl_cx)
     .	  .OR. cx_trans(idim,joint_cx) .NE. cx_trans(idim,addl_cx) ) THEN
	      cx_trans    (idim,joint_cx) = trans_no_transform
	      cx_trans_arg(idim,joint_cx) = bad_val4
	      flags(idim+pdoc_offset_xform) = .TRUE.
	      flags(0) = .TRUE.
	   ENDIF

 100	CONTINUE

	RETURN
	END
