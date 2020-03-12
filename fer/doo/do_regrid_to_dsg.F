	SUBROUTINE DO_REGRID_TO_DSG( src_cx, dst_cx, dset, nfeatures, nobs,
     .				     src, dst)

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
* sample the gridded source data at the ragged points of the destination DSG

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* V720:  *sh* 5/17
* V751: *acm* 8/19 tweaks for grid-to-trajectory 
* V751: *acm* 8/19 apply obs-level masking 
* V751: *acm* 8/19 Handle regrids to timeseries or trajectory
*                  where the grid doesn't have z but the DSG data does
*                  and vice versa (untested). 
* V76  1/20 *acm* working with Point-type dsg data 

        IMPLICIT NONE
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
	include 'xmem_subsc.cmn'

* calling argument declarations:
	INTEGER	src_cx, dst_cx, dset, nfeatures, nobs
        REAL    src( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,
     .               m1lot:m1hit,m1loe:m1hie,m1lof:m1hif ),
     .          dst(nobs)

* local variable declarations:
	LOGICAL ITSA_DSG_TRAJ, fmask(nfeatures), 
     .		process_obs(nobs), sampled_ax(4), skipz

* ... sample_ax is not the same as cxdsg_has_coord cuz the source data may
*     have fixed coordinates on some of the DSG coordinates
	INTEGER ISUBSCRIPT, TM_DSG_DSET_FROM_GRID, TM_LENSTR1,
     .		src_grid, dst_grid, row_size_lm,
     .		idim, ifeature, iobs, ptr, base, ivar, obs_in_feature,
     .		ss(nferdims), orient, status
	REAL	TDEST_CONVERT, ww_lo(4), ww_hi(4), coord
	CHARACTER*20  CD_DSG_FEATURENAME, buff

! DSGTBD need to offer linear interpolation option
!      as well as current point-in-box
! DSGTBD need to check T input grids.  X modulo does work

* initialize
	src_grid = cx_grid(src_cx)
	dst_grid = cx_grid(dst_cx)
	dset = TM_DSG_DSET_FROM_GRID(dst_grid)
	dst_grid = dsg_xlate_grid(dset)

!	bad_src = cx_bad_data( src_cx )   ! not needed here but
!	bad_dst = cx_bad_data( dst_cx )   ! would be needed for interpolation

! set up conversions between DSG times and the time axis of the source grid
	CALL TDEST_WORLD_INIT(dst_grid, src_grid, t_dim, status)
	IF (status .NE. ferr_ok) RETURN

* DSG-specific initialize
	row_size_lm = dsg_loaded_lm(dsg_row_size_var(dset))


* if the orientation is trajectory, timeseries, or point, and if only one of the source 
* and destination grids has z coordinates then ignore it in the other. 

	skipz = .FALSE.
	orient = dsg_orientation(dset)

	IF (orient.NE.z_dim) THEN 
	   IF ( (grid_line(z_dim,src_grid).NE.mnormal .AND. grid_line(z_dim,dst_grid).EQ.mnormal) .OR.
     .          (grid_line(z_dim,src_grid).EQ.mnormal .AND. grid_line(z_dim,dst_grid).NE.mnormal) ) THEN
	      skipz = .TRUE.
	      buff = CD_DSG_FEATURENAME(orient)

	      IF (grid_line(z_dim,dst_grid).EQ.mnormal) CALL WARN ('Gridded data has a Z axis, but '//
     .         buff(:TM_LENSTR1(buff))//' data does not. Use surface data from gridded dataset')
	   ENDIF 
	ENDIF

* determine relevant axes, coordinate limits, and unsampled subscripts
	DO idim = 1, 4
	   IF (idim.EQ.z_dim .AND. skipz) CYCLE
	   sampled_ax(idim) = cx_regrid_trans(idim,dst_cx)
     .			  .NE. unspecified_int4
	   IF ( sampled_ax(idim) ) THEN
* ... the axis of regridding
	      ww_lo(idim) = cx_lo_ww(idim,src_cx)
	      ww_hi(idim) = cx_hi_ww(idim,src_cx)
	   ELSE
* ... all other spatial axes
	      IF (cx_lo_ss(src_cx,idim).NE.cx_hi_ss(src_cx,idim)) GOTO 5100
	      ss(idim) = cx_lo_ss(src_cx,idim)
	   ENDIF
	ENDDO
	ss(e_dim) = cx_lo_ss(src_cx,e_dim)
	ss(f_dim) = cx_lo_ss(src_cx,f_dim) ! DSGTBD F axis support TBD
! F axis will become an outer DO LOOP around all the logic

* get feature mask
        CALL MAKE_DSG_FEATURE_MASK(dset, dst_cx, fmask, nfeatures)

* ****** PERFORM THE SAMPLING *****
	base = 0    ! obs index at end of preceding feature
* ... loop over the features
	DO ifeature = 1, nfeatures 
	   obs_in_feature = dsg_linemem(row_size_lm)%ptr(ifeature)
	   IF (orient .EQ. e_dim) obs_in_feature = 1

* ... have the user's constraints removed this feature?
	   IF (.NOT.fmask(ifeature)) GOTO 500

* ... get the instance-level subscripts in the source data
*     screen out of range instance coordinates in the process
	   DO idim = 1, 4
	      IF (idim.EQ.z_dim .AND. skipz) CYCLE
	      IF ( sampled_ax(idim) ) THEN
	         IF (.NOT.cxdsg_is_obs_coord(idim)) THEN
	            coord = dsg_linemem(cxdsg_coord_lm(idim))%ptr(ifeature)
	            IF (idim .EQ. t_dim) coord = TDEST_CONVERT(coord)
	            IF (coord.LE.ww_lo(idim) .OR. coord.GT.ww_hi(idim)) THEN
	               GOTO 500
	            ELSE
	               ss(idim) = ISUBSCRIPT(coord, src_grid, idim, round_dn)
	            ENDIF
	         ENDIF
	      ENDIF
	   ENDDO
	   IF (skipz) ss(3) = m1loz

* ... get observation-level mask for this feature
	   IF (orient .EQ. e_dim) THEN
	      process_obs(1) = .TRUE.
	   ELSE
	      CALL MAKE_DSG_OBS_MASK(dset, dst_cx, ifeature, base,
     .				  process_obs, obs_in_feature)
	   ENDIF

! DSGTBD Performance optimization possible using a variation on ISUBSCRIPT
! ISUBSCRIPT is where the compute-intensive parts of this procedure occur. 
! A version that pre-checks whether the next subscript is unchanged or
! neighbor to the previously found one on this axis could save time

* ... loop over the observations within each feature
	   DO iobs = 1, obs_in_feature
	      ptr = base + iobs  ! pointer (index) into the ragged contig array

* ... have the user's constraints removed this observation?
	      IF (.NOT.process_obs(iobs)) CYCLE

* ... get the obs-level subscripts in the source data
	      DO idim = 1, 4
	         IF (idim.EQ.z_dim .AND. skipz) CYCLE
	         IF ( sampled_ax(idim) ) THEN
	            IF (cxdsg_is_obs_coord(idim)) THEN
	               coord = dsg_linemem(cxdsg_coord_lm(idim))%ptr(ptr)
	               IF (idim .EQ. t_dim) coord = TDEST_CONVERT(coord)
	               IF (coord.LE.ww_lo(idim) .OR. coord.GT.ww_hi(idim)) THEN
	                  GOTO 400
	               ELSE
	                  ss(idim) = ISUBSCRIPT(coord,src_grid,idim,round_dn)
	               ENDIF
	            ENDIF
	         ENDIF
	      ENDDO

* ... pluck out the gridded source value and insert into the DSG output
	      dst(ptr) = src(ss(1),ss(2),ss(3),ss(4),ss(5),ss(6))  

! DSGTBD: creating equivalences to ss() will slightly shorten code
!         I didn't do this because equivs are invisible in Totalview

 400	      CONTINUE
	   ENDDO ! obs loop
 500	   base = base + obs_in_feature
	ENDDO    ! feature loop

	RETURN

* untrapped error -- SHOULD NEVER HAPPEN -- but if it did anyway, this is nicer
* to the user than a STOP
 5000	RETURN
 5100	CALL ERRMSG( ferr_internal, status, 'DO_REGRID_TO_DSG', *5000)
	END

