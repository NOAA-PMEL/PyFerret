	SUBROUTINE DO_REGRID_FROM_DSG( dsg_cx, grd_cx, dset, nfeatures, nobs,
     .				       dsg, grd, grd_lo, grd_hi)

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
* regrid the DSG features along their feature-orientation axis onto a single,
* shared target grid axis

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* V720:  *sh* 5/17

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
* "dsg" - the source data - features
* "grd" - the target data - a grid
	INTEGER	dsg_cx, grd_cx, dset, nfeatures, nobs, grd_lo, grd_hi, maxobs
        REAL    grd( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,
     .               m2lot:m2hit,m2loe:m2hie,m2lof:m2hif ),
     .		dsg(nobs)

* local variable declarations:
	LOGICAL fmask(nfeatures), repeat_of_same_coords, has_valid, xact
	INTEGER TM_DSG_DSET_FROM_GRID,
     .		dsg_grid, grd_grid, dsg_fmt_grid,
     .		row_size_lm, orient, dsg_coord_lm,
     .		i, idim, ifeature, base, lastbase, flen,
     .		ss(nferdims), status, igrd, idsg, ss12(grd_lo:grd_hi)
	REAL	TM_WORLD, TDEST_CONVERT,
     .          coef(grd_lo:grd_hi), c1, c2, v1, v2,
     .		grd_coord(grd_lo:grd_hi), bad_dsg, bad_grd

* local parameters
	INTEGER   pnolastyet
	PARAMETER (pnolastyet = -9)
	LOGICAL   ascending
	PARAMETER (ascending = .TRUE.)

! DSGTBD only linear interpolation is offered as of 5/2017
!      that may be sufficient, but need to have error traps in case the user
!      tries to request a different regridding algorithm

* initialize
	lastbase = pnolastyet
	dsg_grid = cx_grid(dsg_cx)
	grd_grid = cx_grid(grd_cx)
	bad_dsg  = cx_bad_data(dsg_cx)
	bad_grd  = cx_bad_data(grd_cx)

* DSG-specific initialize
	dset = TM_DSG_DSET_FROM_GRID(dsg_grid)
        CALL MAKE_DSG_FEATURE_MASK(dset, dsg_cx, fmask, nfeatures)
	row_size_lm  = dsg_loaded_lm(dsg_row_size_var(dset))
	dsg_fmt_grid = dsg_xlate_grid(dset)
	orient       = dsg_orientation(dset)
	dsg_coord_lm = cxdsg_coord_lm(orient)

* initialize for conversion of the gridded time axis onto DSG times
	CALL TDEST_WORLD_INIT(grd_grid, dsg_fmt_grid, t_dim, status)
	IF (status .NE. ferr_ok) RETURN   ! untrapped error

* set the indices on the fixed, unvarying coordinate axes of the grid
* only the E and orient axes will be over-ridden later
	DO idim = 1, nferdims
	   ss(idim) = cx_lo_ss(grd_cx,idim)
	ENDDO
! DSGTBD eventually F axis should become an outer DO LOOP around all the logic

* copy the gridded (destination) axis coordinates into a buffer
	DO i = grd_lo, grd_hi
	   grd_coord(i) = TM_WORLD(i,grd_grid,orient,box_middle)
	ENDDO
* ... convert gridded coordinate buffer into the DSG time encoding
	IF (orient .EQ. t_dim) THEN
	   DO i = grd_lo, grd_hi
	      grd_coord(i) = TDEST_CONVERT(grd_coord(i))
	   ENDDO
	ENDIF

* get feature mask

* ****** PERFORM THE REGRIDDING *****
	base = 0    ! obs index at end of preceding feature
* Loop over the features
	DO ifeature = 1, nfeatures 
	   flen = dsg_linemem(row_size_lm)%ptr(ifeature)

* ... have the user's constraints removed this feature?
	   IF (.NOT.fmask(ifeature)) GOTO 500

* ... set the gridded target E axis subscript 
	   ss(e_dim) = ifeature

* ... see if the current feature axis is identical to the last
	   IF (lastbase .NE. pnolastyet) THEN 
!xxxx	      if ifeature GT 1 Loop 1 to flen comparing ifeature to ifeature-1
!   dsg_linemem(dsg_coord_lm)%ptr(base+1)  and "lastbase + 1"
	      repeat_of_same_coords = .FALSE.  ! for debugging purposes
	   ENDIF

* ... analyze the feature coordinates against the target axis coordinates
*     to find neighboring subscripts and interpolation coefficients
	  IF (.NOT. repeat_of_same_coords) THEN
!	     CALL GET_LINEAR_DSG_COEF(flen, dsg_linemem(dsg_coord_lm)%ptr(base+1),
!     .				      ngrd, grd_coord,
!     .				        ss, coef )
! partially developed a new routine, then changed my mind and 
! decided to reuse a routine developed for auxiliary coordinate regridding
             CALL FULL_SCAN_INDICES( dsg_linemem(dsg_coord_lm)%ptr(base+1),
     .				     1, flen,
     .                               1, flen,
     .                               bad_dsg, grd_coord,
     .                               ss12, coef, grd_lo, grd_hi,
     .                               ascending, has_valid )
	  ENDIF

! *sh* Not doing obs-level masking, as it can interfere with interpolation
!      May need to return to this with more experience later.
!* ... get observation-level mask for this feature
!	   CALL MAKE_DSG_OBS_MASK(dset, cx, ifeature, base,
!     .				  process_obs, flen)

* Loop over the target gridded axis points
	   DO igrd = grd_lo, grd_hi

* ... set the gridded target orient axis subscript 
	      ss(orient) = igrd

* ... are interpolants available for this DSG coordinate?
	      IF (coef(igrd) .EQ. bad_val4) THEN
                 grd(ss(1),ss(2),ss(3),ss(4),ss(5),ss(6)) = bad_grd
	         CYCLE
	      ENDIF

* ... interpolation coefficients into the ragged array data
              idsg = base + ss12(igrd)
              c1 = coef(igrd)
              c2 = 1.0 - c1
	      xact  = c1 .EQ. 1.0

* ... surrounding dsg (ragged array source) data points
              v1 = dsg(idsg)
	      IF (.NOT.xact) v2 = dsg(idsg+1)  ! avoid index out of bounds

* ... interpolate
              IF (v1.EQ.bad_dsg .OR. v2.EQ.bad_dsg ) THEN
                 grd(ss(1),ss(2),ss(3),ss(4),ss(5),ss(6)) = bad_grd
              ELSEIF (xact) THEN 
                 grd(ss(1),ss(2),ss(3),ss(4),ss(5),ss(6)) = v1
              ELSE
                 grd(ss(1),ss(2),ss(3),ss(4),ss(5),ss(6)) = c1*v1 + c2*v2
              ENDIF

! DSGTBD: creating equivalences to ss() will slightly shorten code
!         I didn't do this because equivs are invisible in Totalview

	   ENDDO ! obs loop
	   lastbase = base
 500	   base = base + flen
	ENDDO    ! feature loop

	RETURN

! 5000	RETURN
	END

