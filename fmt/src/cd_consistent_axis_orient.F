      SUBROUTINE CD_CONSISTENT_AXIS_ORIENT (dset, reversed, perm, status)


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
* Checks that axis orientation is the same for all grids in the dataset

* Programmer Ansley Manke
* NOAA/PMEL, Seattle, WA - Thermal Modeling and Analysis Program

* revision 0.0 - 1/11 
* V67 1/11 *acm*  Ticket 1774. (Git issue 1046) Changes for making sure axes have the same 
*                 direction in all  grids of a dataset.
* V672 8/11 *acm* When reordering axes, dont wipe out an existing axis that is not being moved.
* V674 2/12 *acm* 6D Ferret: use nferdims rather than 4 for dimension indexing
* V698 3/16 *sh*  changed an array declaration to match grid_ceiling change
* V751 6/19 *acm* Fix issue 1941 and comment at the end of GIT issue 1046. When this routine
*                 is called to handle grids with axes in unknown directions for a second dataset, 
*                 it was clobbering the negative sign on ds_ordering that indicates axis reversal 
*                 is needed.  (in loop 230 ds_ordering(idim,tmvar) incorrectly set to 0)
* V751 6/19 *acm* Also allocate all the working storage used by this routine dynamically,
*                 cleaning up unnecessary space and processing time.
* V751 12/19 *acm* Issue 1962: When can't get consistent grids, give some coaching and
*                 declare it an error.
*  v76   5/2020   *acm* USE/ORDER= when grids in files have different dimensions
*  v7.65 5/2022   *acm* Issue 1988, still some inconsistencies; files that should
*                 be useable but can't be initialized.

* argument definitions:
*      dset        - TMAP data set pointer
*      reversed    - logical array to record which axes have reversed coords
*      perm        - user-specified permutation (e.g. "YXZT"=2,1,3,4)
*      status       - return status

* include files
      include 'netcdf.inc'
      include 'tmap_errors.parm'
      include 'tmap_dims.parm'
#include "tmap_dset.parm"
#include "gt_lib.parm"
      include 'xbuild_grids.cmn'
      include 'xdset_info.cmn_text'
      external xdset_info_data
      include 'xio.cmn_text'
      external xio_data
      include 'xtm_grid.cmn_text'
            include 'xunits.cmn_text'
      external xunits_data
      include 'xdset_info.cd_equiv'
      include 'errmsg.parm'

* argument declarations
      LOGICAL reversed(max_lines:line_ceiling)
      INTEGER dset, perm(*), status

* local variable declarations:

      INTEGER TM_LENSTR1, i, j, igrid, igrid_min, igrid_max, idim, iaxis, 
     .        disordered, reordered, n_in_grid,  
     .        grid_save(nferdims), igrid1, ipos1, igrid2, ipos2, kk, 
     .        iaxis_last, iax, iorder, tmvar, ivar, vartyp, nvdims, 
     .        vdims(8), nvatts, vlen, istat, lastdim, tmplines(nferdims), 
     .        i1, i2, iter, ndim, isave_order, ncheck, g1, g2, l1, l2
      LOGICAL shift_ok, need_iterate, coordvar, all_outflag, permuted
      CHARACTER vname*128

      LOGICAL, allocatable :: already_noted (:)
      INTEGER, allocatable :: grid_ndims (:)
      INTEGER, allocatable :: line_use_dir (:)
      INTEGER, allocatable :: copy_of_grids ( : , : ) 
      
      REAL, allocatable ::  axes_out_of_order(:)

* We don't need to do all of this for DSG datasets. Their grids
* are defined by that structure.

      IF (dsg_ragged(dset)) RETURN

      iter = 0
      reordered = 0

* Allocate temporary storage for storing grid info.
* Count variables in this dataset and range of grids
* Get range of line numbers.

      g1 = max_dyn_grids
      g2 = 0
      l1 = max_dyn_lines
      l2 = 0
      DO tmvar = 1,maxvars
         IF ( ds_var_setnum(tmvar) .EQ. dset ) THEN
            igrid = ds_grid_number(tmvar)
	    g1 = MIN(g1, igrid)
	    g2 = MAX(g2, igrid)

	    DO idim = 1, nferdims
	       iaxis = grid_line(idim, igrid)
	       IF (iaxis.NE.line_0) l1 = MIN(l1, iaxis)
	       IF (iaxis.NE.line_0) l2 = MAX(l2, iaxis)
	    ENDDO
	 ENDIF
      ENDDO

      ALLOCATE ( grid_ndims(g1:g2) )
      ALLOCATE ( copy_of_grids(nferdims, g1:g2) )

      ALLOCATE ( line_use_dir(l1:l2) )
      ALLOCATE ( already_noted(l1:l2) )
      ALLOCATE ( axes_out_of_order(l2-l1+1) )

* Initialize temporary storage

      already_noted = .FALSE.
      axes_out_of_order = 0.
      line_use_dir = 0
      grid_ndims = 0
      copy_of_grids = 0

 1000 CONTINUE
      disordered = 0

      need_iterate = .FALSE.  ! is this the right place to set this?

* Loop over the grids created for this dataset

      igrid = 0               ! to get start of linked list
  10  CALL TM_NEXT_TMP_GRID( igrid, *19)

      DO 100 idim = 1, nferdims
         iaxis = grid_line(idim, igrid)

         IF (iaxis  .GT. 0) THEN
            IF (line_use_dir (iaxis) .EQ. 0) THEN
               line_use_dir(iaxis) = idim
            ELSE
               IF ( line_use_dir(iaxis) .NE. idim) THEN
                  disordered = disordered + 1
                  axes_out_of_order(disordered) = FLOAT(iaxis)
               ENDIF
            ENDIF
         ENDIF
  100 CONTINUE

      GOTO 10
  19  CONTINUE
      
      IF (disordered .EQ. 0) GOTO 2000

      CALL HEAP2_V (axes_out_of_order, 0., disordered)

! Count dimensions of the grids.

      igrid = 0 
  20  CALL TM_NEXT_TMP_GRID( igrid, *29)
      grid_ndims(igrid) = 0
      DO 110 idim = 1, nferdims
         IF (grid_line(idim,igrid) .GT. 0) 
     .            grid_ndims(igrid) = grid_ndims(igrid) + 1 
  110 CONTINUE
      GOTO 20
  29  CONTINUE  ! counting dimensions

      iaxis_last = 0
      iorder = 0
      DO 220 i = 1, disordered
         iorder = iorder + 1
         iaxis = INT(axes_out_of_order(iorder))
	 IF (iaxis.EQ.iaxis_last .OR. iaxis.EQ.0) CYCLE  ! out of loop 220
         iaxis_last = iaxis

C Axis marked with a direction but is in inconsistent position in grids?
         IF (line_direction(iaxis).NE.'NA' .AND. .NOT.already_noted(iaxis)) THEN
	    vlen = TM_LENSTR1(line_name(iaxis)) 
            CALL TM_NOTE( 'Axis '//line_name(iaxis)(1:vlen)//' marked as '//
     .          line_direction(iaxis), lunit_errors )
            CALL TM_NOTE(
     .          'but it has different directions in grids. ', lunit_errors)
            GOTO 5100
         ELSE

c Find the grid which contains this axis and has the most dimensions.
            igrid1 = 0
            ipos1 = 0
            n_in_grid = 0
	    
            igrid = 0 
  40        CALL TM_NEXT_TMP_GRID( igrid, *49)
            DO 130 idim = 1, nferdims
               IF (grid_line(idim, igrid) .EQ. iaxis .AND. 
     .             grid_ndims(igrid) .GT. n_in_grid) THEN
                  igrid1 = igrid
                  ipos1 = idim
                  n_in_grid = grid_ndims(igrid)
               ENDIF
  130        CONTINUE

            GOTO 40
  49        CONTINUE  ! Grid with this axis and most dimensions

* Adjust grids based on position already determined for the grid with most axes.

            igrid = 0               ! Loop over all grids comparing with grid1
  90        CALL TM_NEXT_TMP_GRID( igrid, *99)
            IF (igrid .NE. igrid1) THEN
               ipos2 = 0
               igrid2 = 0
               DO 150 idim = 1, nferdims
                  IF (grid_line(idim, igrid).EQ. iaxis .AND. 
     .                 idim .NE. ipos1)  THEN
                     igrid2 = igrid
                     ipos2 = idim
                  ENDIF
  150          CONTINUE

	       IF (igrid2 .NE. 0) THEN
                  i1 = MIN(ipos1, ipos2)
                  i2 = MAX(ipos1, ipos2)
                  shift_ok = .FALSE.
                  DO 160 idim = i1, i2
                     iax = grid_line(idim,igrid2)
                     IF (grid_line(idim,igrid2) .LE. 0 .OR.
     .                       line_direction(iax) .EQ. 'NA') 
     .                       shift_ok = .TRUE.
  160             CONTINUE
                  IF (shift_ok) THEN
                     DO 170 idim = 1, nferdims
                        grid_save(idim) = grid_line(idim,igrid2)
  170                CONTINUE

		     grid_line(ipos1,igrid2) = grid_line(ipos1,igrid1)
                     line_use_dir(grid_line(ipos1,igrid1)) = ipos1
                     grid_line(ipos2,igrid2) = grid_save(ipos1)

		     need_iterate = .TRUE.
                     reordered = reordered + 1
                  ELSE
                     vlen = TM_LENSTR1(line_name(iaxis)) 

                     CALL TM_NOTE('Could not make grid containing axis '// 
     .                       line_name(iaxis)(1:vlen), lunit_errors)

                     GOTO 5100
                  ENDIF
                  
               ENDIF  ! igrid2 .ne. 0
            ENDIF
             
            GOTO 90
  99        CONTINUE  ! Loop over all grids comparing with grid1

          ENDIF
  220 CONTINUE ! disordered
      iter = iter + 1
      IF (iter .LT. 10) GOTO 1000
* Here loop back if need_iterate is true.

* Reset ds_grid_start and ds_grid_end

 2000 CONTINUE 

      IF (reordered .GT. 0) THEN
* * * * loop through each variable in this TMAP data set * * *
      DO 250 tmvar = 1,maxvars
         IF ( ds_var_setnum(tmvar) .NE. dset ) CYCLE ! loop 250
         igrid = ds_grid_number(tmvar)

! save a copy of the grid
         DO idim = 1, nferdims
            copy_of_grids(idim, igrid) = grid_line(idim, igrid)
         ENDDO

* Check.  Did we wind up with a different number of axes after
* attempt to fix the grids?

         ncheck = 0
         DO 200 idim = 1, nferdims
	       IF (grid_line(idim,igrid) .GT. 0) ncheck = ncheck + 1
  200     CONTINUE 

         IF (ncheck .NE. grid_ndims(igrid)) THEN
* Issue warnings
            ivar = cd_varid(tmvar)
            CALL CD_GET_VAR_INFO(dset, ivar, vname, vartyp, nvdims,
     .                 vdims, nvatts, coordvar, all_outflag, istat)
            vlen = TM_LENSTR1(vname) 
            CALL TM_NOTE(' ', lunit_errors)
            CALL TM_NOTE('Could not adjust grid for variable '// 
     .                 vname(1:vlen), lunit_errors)
            GOTO 5100

         ENDIF

	 ndim = 0
	 DO 230 idim = 1, nferdims
	    iaxis = grid_line(idim, igrid)
	    ds_grid_start(idim,tmvar) = 1
	    ds_grid_end(idim,tmvar) = 1
            IF (iaxis .NE. 0) THEN
	       ds_grid_end(idim,tmvar) = line_dim(iaxis)
	       ndim = ndim + 1
	       isave_order = ds_ordering(ndim,tmvar)  ! want to keep the sign
	       ds_ordering(ndim,tmvar) = idim

	       IF ( reversed(iaxis) ) THEN
	          ds_ordering(ndim,tmvar)
     .			      = -1 * ds_ordering(ndim,tmvar)
               ELSE
	          IF ( perm(idim).NE.unspecified_int4 .AND. perm(idim).LT.0) THEN
		     ds_ordering(ndim,tmvar) = -1 * ds_ordering(ndim,tmvar)

                  ELSE   ! if already had a reversed coord indicated, save that.
                     IF ( (ABS(isave_order) .EQ. ABS(ds_ordering(ndim,tmvar)) )
     .	                 .AND.(isave_order .NE. ds_ordering(ndim,tmvar)) )
     .	               ds_ordering(ndim,tmvar) = isave_order

                  ENDIF
               ENDIF
	    ENDIF

  230    CONTINUE 

* User-supplied ordering and reversals were applied in get_generic_grids


* get the variable name
         IF (permuted) THEN
            ivar = cd_varid(tmvar)
            CALL CD_GET_VAR_INFO(dset, ivar, vname, vartyp, nvdims,
     .                 vdims, nvatts, coordvar, all_outflag, istat)

            vlen = TM_LENSTR1(vname)
            CALL TM_AXIS_ORDER(tmplines, perm, ds_ordering(1,tmvar),
     .                           nvdims, vname(:vlen))
         ENDIF

  250 CONTINUE
      ENDIF
	  
5000  CONTINUE
      DEALLOCATE ( grid_ndims )
      DEALLOCATE ( copy_of_grids )

      DEALLOCATE ( line_use_dir )
      DEALLOCATE ( already_noted )
      DEALLOCATE ( axes_out_of_order )

      RETURN
 5100 CONTINUE
      CALL ERRMSG (ferr_inconsist_grid, status, 
     .       'File needs axis attributes on coordinate variables '// 
     .       'indicating correct directions', *5000)

      END
