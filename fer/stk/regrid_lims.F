	SUBROUTINE REGRID_LIMS( dst_cx, src_cx, axis, status, * )

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
* given a context describing data to be regridded to a destination grid
* - select one of the axes needing regridding
* - create, if necessary, an intermediate grid that differs from the
*	destination grid by only this single axis
* - compute the region on the source grid (possibly the newly created
*	intermediate grid) that will supply the re-griddable data
* When called the source context should be a copy of the destination context 

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 -  7/25/88
* revision 0.1 -   9/1/89 - corrected bug in destination world limits
*			  - added xy averaging
* revision 0.2 -   6/2/89 - modified grid limits calc. for modulo axes
* V200:  7/31/89 - 4D symmetrical version
*		 - fixed bug following GRID_SUBSCRIPT_EXTREMES for modulo axes
*	 11/3/89 - if destination limits are unknown flag error
*	 12/1/89 - handles XY averaging - (uses RQST_REGRID)
* V420: 10/95	 - Support for implicit axes: intermed. grids inherit
*		   responsibilities (use counts) for implicit axes
*	11/95	   check that destination axis is modulo for modulo regrid
*		   and take abnormal return if it dest region is
*		   modulo-wrapped
* V440: 9/96	 - Bug: failed to call TM_USE_LINE for dynamic line in
*		   intermediate grid (err430_deallo_dyn_line)
*		   Also corrected undetected bug: in intermediate grid for
*		   XY averaging need to deallo the y_dim not the "axis"
* V500 *kob* 3/99- up VAR_CODE  to 64 chars 
*      *sh*  5/99 - incorporate logic for netCDF strides. Allow for multi-axes
*		    use same framework to handle XY averaging
*	*sh* 4/00 - all grids and axes now maintain use counts and have
*			line_class available
* V533 *sh* 6/01 - allow strides and modulo to be combined
* V550 *sh*  2/02 - support for subspan modulo axes
* V552 *acm* 4/03- up VAR_CODE  to 128 chars 
* v554 *acm* 3/04 - When a strides request is made, and a grid had been defined
*                   for use in comparisons (define grid/like=var), the request 
*                   got turned into a regrid instead of a stride. Change the test
*                   for may_have_strides.
* V671  6/11 *acm*  New 2D linear xy regridding, fixing ticket 1862
*       *acm* 3/12 Add E and F dimensions (use nferdims in tmap_dims.parm)
* V74+  5/18 *acm* issue 1868: gridding transforms that will grid to same axis
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include 'implicit.parm'
	include	'xtm_grid.cmn_text'
	external xgt_grid_data
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xprog_state.cmn'
	include	'xtext_info.cmn'

* calling argument declarations:
	INTEGER	src_cx, axis, dst_cx, status

* internal variable declarations:
	LOGICAL	ITSA_MODULO_REGRID, ITSA_MODULO_AXIS,  TM_ITS_CDF,
     .		REGRID_TO_SELF,
     .		xy_regrid, last_axis, may_have_strides, has_strides,
     .		unacceptable_modulo
	INTEGER	KNOWN_GRID, CAXIS_MODLEN, TM_AXIS_STRIDE, CX_DIM_LEN,
     .		idim, axis_list(nferdims), naxis, var_grid, src_grid, dst_grid,
     .		next_axis, dst_axis, var_axis, i, axlen, i1
	REAL	fraction
	CHARACTER VAR_CODE*128, dtype*4

* initialize
* this is a bit of a hack for implementing netCDF strides
* Under circumstances listed near bottom we may abort the intention to use
* strides In that case, we come back and do this routine over again.
* We save a copy of the src_cx in case of this eventuality
	CALL TRANSFER_CONTEXT(src_cx, cx_buff)

* insert source grid ( defining grid for this variable )
	dst_grid = cx_grid( dst_cx )
	var_grid = KNOWN_GRID( cx_data_set(src_cx),
     .			  cx_category(src_cx),
     .			  cx_variable(src_cx)	)

* initially select a single axis for regridding
* The listed transformations will operate even on the same axis
	DO axis = 1, nferdims
 	   IF (grid_line(axis,dst_grid) .NE. grid_line(axis,var_grid)) GOTO 20
	   IF (REGRID_TO_SELF (cx_regrid_trans(axis,src_cx) )) GOTO 20
	ENDDO
* ... signal calling routine that there's nothing to regrid
	cx_unstand_grid( dst_cx ) = .FALSE.
	status = ferr_ok
	RETURN

* special requirement for modulo regrid
 20	IF ( ITSA_MODULO_REGRID(dst_cx,axis) ) THEN
	   IF ( .NOT.ITSA_MODULO_AXIS(dst_cx,axis) ) GOTO 5100
* ... strip off the destination modulo wraps if any with abnormal return
* (so we compute just the base limits, 1-N, of the modulo axis)
	   IF ( cx_lo_ss(dst_cx,axis) .NE. unspecified_int4 ) THEN
	      IF (cx_hi_ss(dst_cx,axis ) .GT. CAXIS_MODLEN(axis,dst_cx)
     .       .OR. cx_lo_ss(dst_cx,axis ) .LT. 1 ) RETURN 1
	   ENDIF
	ENDIF

* (5/99) should more than one axis be regridded at once?
* yes if XY average regridding
* yes if multiple axis strides in a netCDF file
* for stride axes, we cannot have modulo and cannot have other
* types of regridding needed also

* If the regrid is of the form netCDFvar[i=lo:hi:del] then it is
* currently represented as a regrid to a dynamic axis. Since the netCDF
* library can do strides, we will allow it to as long as a variety
* of conditions are met. The "flag" that will trigger strides in the netCDF
* IO is the presence of a cx_delta value (see rqst_regrid.F and read_tm.F)
* Note that modulo regrids cannot be handled with strides: suppose X axis
* is 1:360 degrees by one degree, then V[I=1:720:7] will select different
* points in the 1:360 range than in the 361:720 range.
	may_have_strides =  cx_category(dst_cx) .EQ. cat_file_var

	IF (may_have_strides) THEN
!	   may_have_strides = TM_ITS_CDF(cx_data_set(dst_cx), dtype )
!     .                  .AND. cx_has_impl_grid(dst_cx) 

	   may_have_strides = TM_ITS_CDF(cx_data_set(dst_cx), dtype )

	ENDIF
 50	naxis = 1
	axis_list(1) = axis	
	has_strides = .FALSE.
	xy_regrid = (cx_regrid_trans(axis,src_cx) .EQ. prgrd_xy_ave).OR. 
     .           (cx_regrid_trans(axis,src_cx) .EQ. prgrd_xy_lin)
	IF ( xy_regrid)   THEN
	   naxis = 2
	   axis_list(2) = y_dim
	ELSEIF ( may_have_strides ) THEN
	  DO idim = axis, nferdims
	    dst_axis = grid_line(idim,dst_grid)
	    var_axis = grid_line(idim,var_grid)
	    
	   IF (dst_axis.EQ.var_axis .AND. 
     .      REGRID_TO_SELF (cx_regrid_trans(axis,src_cx) )) GOTO 110

	    IF (dst_axis.EQ.var_axis) CYCLE
	    IF (cx_regrid_trans(idim,dst_cx).EQ.prgrd_xact_pts
     .     .OR. cx_regrid_trans(idim,dst_cx).EQ.prgrd_exact) THEN
	      IF ( line_class(dst_axis) .EQ. pline_class_stride ) THEN
	        IF (idim .NE. axis) naxis = naxis + 1
	        axis_list(naxis) = idim
	      ELSE
	        GOTO 110	! abort
	      ENDIF
	    ELSE
	      GOTO 110		! abort
	    ENDIF
	  ENDDO
	  has_strides = .TRUE.  ! all axes meet the conditions
 110	  IF (.NOT.has_strides) naxis = 1
	ENDIF

* will there be more axes to regrid after this/these?
	next_axis = axis_list(naxis) + 1
	last_axis = .TRUE.
	DO idim = next_axis, nferdims
 	   IF ( grid_line(idim,dst_grid) .NE. grid_line(idim,var_grid) )
     .							last_axis = .FALSE.
     	ENDDO
	IF (.NOT.last_axis) has_strides = .FALSE.  ! (can never happen)

* if its the last axis then the source data will be on its defining grid
	IF ( last_axis ) THEN
	   src_grid = var_grid
	   cx_unstand_grid( src_cx ) = .FALSE.
	   is_phase( isp )   = prgrd_from_source
	ELSE		
* we need an intermediate grid
	   cx_unstand_grid( src_cx ) = .TRUE.
	   CALL ALLO_GRID( src_grid,status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   CALL TM_COPY_GRID_W_LINE_USE( dst_grid, src_grid ) 
	   grid_name( src_grid ) = ww_dim_name(axis)//'NTERMED'	! for diag out
	   CALL TM_DEALLO_DYN_LINE(grid_line(axis,src_grid))
	   grid_line( axis, src_grid ) = grid_line( axis, var_grid )
	   CALL TM_USE_LINE(grid_line(axis,var_grid)) ! 9/96
	   IF (xy_regrid) THEN
! 9/96: changed DEALLO grid_line(axis,src_grid) to grid_line(y_dim,src_grid)
	      CALL TM_DEALLO_DYN_LINE(grid_line(y_dim,src_grid))
	      grid_line(y_dim,src_grid) = grid_line(y_dim,var_grid)
	      CALL TM_USE_LINE(grid_line(y_dim,var_grid)) !9/96
	   ENDIF
	   is_phase( isp ) = prgrd_from_ntrmed
	ENDIF
	cx_grid( src_cx ) = src_grid

* get limits to request
	DO idim = 1, naxis
	   CALL RQST_REGRID( dst_cx, src_cx, axis_list(idim),
     .			     has_strides, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	ENDDO

* if doing strides then we may decide to abort
* 1) if it turns out to be a modulo stride operation (e.g. I=1:720:11 on
*    a 360 point modulo longitude axis. Note that we could have checked for
*    this above, but since RQST_REGRID does all the work, we waited
* 2) if the complete source data is already in cache don't reread w/ strides
	IF (has_strides) THEN
	   unacceptable_modulo = .FALSE.

	   DO i = 1, naxis
	      idim = axis_list(i)
	      axlen = CAXIS_MODLEN(idim, src_cx)
	      IF (CX_DIM_LEN(idim,src_cx) .GT. axlen) THEN
*     ...  for multi-cycle modulo stride must be an integer factor of ax length
	         fraction = FLOAT(axlen) /
     .			    TM_AXIS_STRIDE(grid_line(idim,dst_grid), i1)
	         unacceptable_modulo = unacceptable_modulo
     .				 .OR.  fraction .NE. INT(fraction)
	      ENDIF
	   ENDDO

! THE COMMENTED OUT CODE BELOW CAN BE USED TO TURN ON LOGIC THAT WILL REUSE
! CACHED (NON_STRIDES) DATA RATHER THAN REREADING USING STRIDES. IT IS
! COMMENTED OUT AS NOT ENTIRELY DESIRABLE: 1) IT CREATES MULTIPLE CODE PATHS
! FOR THE SAME OPERATION; 2) IT CAN CAUSE UNACCEPTABLE MEMORY USAGE AS THE
! NON-STRIDED BASE REGION GETS MODULO-REPLICATED BEFORE SUBSAMPLING
* ... if full source region is already in memory without strides then
*     do striding from it in memory -- no new IO needed (optimization)
!	   CALL FIND_MEM_VAR( src_cx, mr, search_by_value )
!           IF ( unacceptable_modulo .OR. mr.GT.mr_nonexist ) THEN
           IF ( unacceptable_modulo ) THEN
	      may_have_strides = .FALSE.
	      CALL TRANSFER_CONTEXT( cx_buff, src_cx)
	       IF (mode_diagnostic) THEN
	        CALL WARN('Not using NetCDF strides for multi-cycle modulo')
	        CALL WARN('Stride value not a factor of axis length')  
	       ENDIF
	      GOTO 50			! re-do w/out strides
	   ENDIF
	ENDIF

* success
        status = ferr_ok
	RETURN

* error exit
 5000	RETURN
 5100	CALL ERRMSG( ferr_regrid, status,
     .          '@MOD regridding to a non-modulo axis: '
     .          //VAR_CODE(cx_category(dst_cx),cx_variable(dst_cx)),
     .		*5000)

	END
