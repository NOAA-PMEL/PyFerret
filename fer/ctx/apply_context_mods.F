	SUBROUTINE APPLY_CONTEXT_MODS
     .				( mods_cx, dest_cx, nest_new_cx, status )

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
* modify the indicated destination context based on mods in mods_cx
* apply only those modifications that can be done without for-certain knowledge
* of the underlying grid.  For example, translating subscript values into
* WORLD coordinates will be done in FINISH_CONTEXT_MODS instead of here

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 10/15/86
* revision 0.1 -   4/8/87 - mods not applied to normal axes
* revision 0.2 -  4/29/87 - fully incorporated relative (implied) coordinates
*			  - and added check on transformation nesting
* revision 0.3 - 5/12/87  - corrected bug of check being made to see if an axis
*			    was normal as a test for applying a mod.  This is
*			    invalid since the grid is not presumed to be known.
* revision 0.4 - 8/18/87  - added transfer of trans_arg (for @SHF)
* revision 0.5 - 9/11/87  - transfer delta only if command or if explicitly
*			    given in variable modifier
* revision 0.6 -  4/1/89  - apply regrididng transformations, too
* V200:  5/19/89 - no nested transformations possible
* 		 - eliminated "d" notation and "relative" logic
*		 - changed logic of cx_given and cx_dset_given
* V420: 9/95	 - Support for implicit grids
*	10/95	 - xfer delta specifications whenever mods to axis are explicit
*		 - also xfer cx_regrid_trans since dynamic axes may use
* *kob* 9/97 - support for negative time steps - make sure cx_calendar
*              gets transfered
* *acm* v580 11/04 Transfer cx_cal_id, the calendar ID used when computing 
*                  world coordinates
*       *acm* 3/12 Add E and F dimensions (use nferdims in tmap_dims.parm)
* V685 3/13 *sh* support for aux var regridding: var[gz(depth)=zax]
* V695  *acm* 2/15 cx_calendar stores info about calendar formatting of T and F axes
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

	IMPLICIT NONE

	include 'tmap_dims.parm'
	include 'errmsg.parm'
	include 'ferret.parm'
	include 'xcontext.cmn'
	include 'xvariables.cmn'
	include 'xprog_state.cmn'
	include 'xtext_info.cmn'

* calling argument declarations:
	INTEGER	mods_cx, dest_cx, status
	LOGICAL	nest_new_cx

* internal variable declarations:
	INTEGER		idim, var, trans, i

* first see if data_set or variable is being modified
	cx_dset_gvn( dest_cx ) = cx_dset_gvn( mods_cx )
	IF (cx_dset_gvn(mods_cx)) cx_data_set(dest_cx) = cx_data_set(mods_cx)
	IF ( cx_variable( mods_cx ) .NE. unspecified_int4 ) THEN
	   cx_variable( dest_cx ) = cx_variable( mods_cx )
	   cx_category( dest_cx ) = cx_category( mods_cx )
	ENDIF

* modify the grid if the user said to explicitly
	cx_unstand_grid ( dest_cx ) = cx_unstand_grid ( mods_cx )
	cx_has_impl_grid( dest_cx ) = cx_has_impl_grid( mods_cx )
	cx_has_impl_grid( mods_cx ) = .FALSE.	! house-keeping: only 1 cx SET
	cx_impl_grid    ( dest_cx ) = cx_impl_grid    ( mods_cx )
	IF ( cx_unstand_grid( mods_cx ) ) THEN
	   cx_grid( dest_cx ) = cx_grid( mods_cx )
	ENDIF
* ... apply regrid transform to each axis (some may be "unspecified")
	DO idim = 1, nferdims
 	   cx_regrid_trans(idim,dest_cx) = cx_regrid_trans(idim,mods_cx)
	ENDDO

* apply modifications for each axis ( except normal axes, of course )
	DO 100 idim = 1, nferdims
	   IF ( .NOT.cx_given( idim, mods_cx )	   )	GOTO 100

* ... "given" flag on dest context to represent limits (not transform) given
	   cx_given( idim, dest_cx ) = 
     .		cx_lo_ss( mods_cx, idim ) .NE. unspecified_int4
     .	   .OR. cx_lo_ww( idim, mods_cx ) .NE. unspecified_val8

* ... transfer coordinates if they were given (e.g. L=@SBX has no coords)
	   IF ( cx_given( idim, dest_cx ) ) THEN
	      cx_by_ss( idim, dest_cx ) = cx_by_ss( idim, mods_cx ) 
	      IF ( cx_by_ss( idim, mods_cx ) ) THEN
	         cx_lo_ss( dest_cx ,idim) = cx_lo_ss( mods_cx,idim)
	         cx_hi_ss( dest_cx ,idim) = cx_hi_ss( mods_cx,idim)
	         cx_lo_ww( idim,dest_cx ) = unspecified_val8
	         cx_hi_ww( idim,dest_cx ) = unspecified_val8
* kob 9/97 - need to reset cx_calendar as well
		 IF (idim.EQ.t_dim .OR. idim.EQ.f_dim) THEN
                    cx_calendar(idim,dest_cx) = .FALSE.
                    cx_cal_id(dest_cx) = unspecified_int4
                 ENDIF
	      ELSE
	         cx_lo_ww( idim,dest_cx ) = cx_lo_ww( idim,mods_cx)
	         cx_hi_ww( idim,dest_cx ) = cx_hi_ww( idim,mods_cx)
	         cx_lo_ss( dest_cx ,idim) = unspecified_int4
	         cx_hi_ss( dest_cx ,idim) = unspecified_int4
* kob 9/97 - need to transfer cx_calendar as well
		 IF (idim.EQ.t_dim .OR. idim.EQ.f_dim) THEN
                    cx_calendar(idim,dest_cx) = cx_calendar(idim,mods_cx)
                    cx_cal_id(dest_cx) = cx_cal_id(mods_cx)
                 ENDIF

	      ENDIF
	      cx_delta( idim, dest_cx )   = cx_delta( idim, mods_cx ) ! 10/95
	   ENDIF

*  ... transfer transformation ( imposes @nul if nothing was given )
	   trans = cx_trans(idim,mods_cx)
!	   IF ( itsa_uvar ) THEN
!* ... check for illegal nesting of transforms
!* (note: in V2.0 omitting this doesn't generate errors - just inappropriate
!*  behavior like @SBX on an @AVE axis generates a field of bad data)
!* ... cannot use because @SBX is valid on U-U[K=@AVE] although it is need_xact
!	     IF ( ( uvar_given( idim, var ) .EQ. uvlim_gvn_xact
!     .	       .OR. uvar_given( idim, var ) .EQ. uvlim_need_xact )
!     .	     .AND.( trans .NE. trans_no_transform ) ) GOTO 5100
!	   ENDIF
	   cx_trans   ( idim, dest_cx ) = trans
	   cx_trans_arg(idim, dest_cx ) = cx_trans_arg(idim, mods_cx )

*   ... transfer delta always for command mod or if explicit for variable mod
	   IF ( .NOT.nest_new_cx )
     .	   cx_delta( idim, dest_cx ) = cx_delta( idim, mods_cx ) 

 100	CONTINUE

* transfer aux var info
	cx_naux( dest_cx ) = cx_naux( mods_cx )
	IF ( cx_naux(dest_cx) .GE. 1 ) THEN
	   DO 200 i = 1, cx_naux( dest_cx )
	      cx_aux_arg_start(i, dest_cx) = cx_aux_arg_start(i, mods_cx)
	      cx_aux_arg_end  (i, dest_cx) = cx_aux_arg_end  (i, mods_cx)
 200	   CONTINUE
	ENDIF

	status = ferr_ok
 5000	RETURN
! 5100	CALL ERRMSG( ferr_trans_nest, status,
!     .		'invalid nesting of @'//ALG_TRANS_CODE(trans)//
!     .		' on '//ww_dim_name(idim)//' axis', *5000 )
	END
