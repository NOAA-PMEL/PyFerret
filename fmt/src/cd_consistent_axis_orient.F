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
* V67 1/11 *acm*  Ticket 1774. Changes for making sure axes have the same direction
*                 in all  grids of a dataset.

* argument definitions:
*      dset        - TMAP data set pointer
*      reversed    - logical array to record which axes have reversed coords
*      perm        - user-specified permutation (e.g. "YXZT"=2,1,3,4)
*      status       - return status

* include files
      include 'netcdf.decl'
#include "netcdf.inc"
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
      external xgt_grid_data
      include 'xunits.cmn_text'
      external xunits_data
      include 'xdset_info.cd_equiv'

* argument declarations
      LOGICAL  reversed(max_lines:line_ceiling)
      INTEGER dset, perm(4), status

* local variable declarations:

      INTEGER TM_LENSTR1, i, j, igrid, igrid_min, igrid_max, idim, iaxis, 
     .        disordered, reordered, axes_out_of_order(100), n_in_grid,  
     .        grid_save(4), igrid1, ipos1, igrid2, ipos2, kk, 
     .        iaxis_last, iax, iorder, tmvar, ivar, vartyp, nvdims, 
     .        vdims(8), nvatts, vlen, istat, lastdim, tmplines(4)
      LOGICAL shift_ok, need_iterate, coordvar, all_outflag, permuted

      CHARACTER vname*128

      INTEGER grid_ndims (max_grids:grid_ceiling), 
     .        line_use_dir(0:line_ceiling)

      iter = 0
      reordered = 0
      igrid_min = grid_ceiling
      igrid_max = 0

      DO i = 1, 100
         axes_out_of_order(i) = 0
      ENDDO
      DO i = 0, line_ceiling
         line_use_dir(i) = 0
      ENDDO
      DO i = max_grids, grid_ceiling
         grid_ndims(i) = 0
      ENDDO

 1000 CONTINUE
      disordered = 0

      need_iterate = .FALSE.  ! is this the right place to set this?

* Loop over the grids created for this dataset

      igrid = 0               ! to get start of linked list
  10  CALL TM_NEXT_TMP_GRID( igrid, *19)

      igrid_min = MIN(igrid, igrid_min)
      igrid_max = MAX(igrid, igrid_max)
      DO 100 idim = 1, 4
         iaxis = grid_line(idim, igrid)
         IF (iaxis  .GT. 0) THEN
            IF (line_use_dir (iaxis) .EQ. 0) THEN
               line_use_dir(iaxis) = idim
            ELSE
               IF ( line_use_dir(iaxis) .NE. idim) THEN
                  disordered = disordered + 1
                  axes_out_of_order(disordered) = iaxis
               ENDIF
            ENDIF
         ENDIF
  100 CONTINUE

      GOTO 10
  19  CONTINUE

      IF (igrid_max .EQ. 0) RETURN
      IF (disordered .EQ. 0) GOTO 2000

      CALL HEAP2_V (axes_out_of_order, 0., disordered)

      DO 120 igrid = igrid_min, igrid_max
         grid_ndims(igrid) = 0
         DO 110 idim = 1, 4
            IF (grid_line(idim,igrid) .GT. 0) 
     .            grid_ndims(igrid) = grid_ndims(igrid) + 1 
  110    CONTINUE
  120 CONTINUE

      iaxis_last = 0
      iorder = 0
      DO 200 i = 1, disordered
         iorder = iorder + 1
         iaxis = axes_out_of_order(iorder)
	 IF (iaxis.EQ.iaxis_last .OR. iaxis.EQ.0) GOTO 200
         iaxis_last = iaxis

C Can this happen? Axis marked with a direction but is in inconsistent position in grids?
         IF (line_direction(iaxis) .NE. 'NA') THEN
	    vlen = TM_LENSTR1(line_name(iaxis)) 
            CALL TM_NOTE( 'Axis '//line_name(iaxis)(1:vlen)//' marked as '//
     .          line_direction(iaxis), lunit_errors )
            CALL TM_NOTE(
     .          'but it has different directions in grids. '//
     .          'Axes in grids may be inconsistent.', lunit_errors)
         ELSE

c Find the grid which contains this axis and has the most dimensions.
            igrid1 = 0
            ipos1 = 0
            n_in_grid = 0
            DO 140 igrid = igrid_min, igrid_max
               DO 130 idim = 1, 4
                  IF (grid_line(idim, igrid) .EQ. iaxis .AND. 
     .                grid_ndims(igrid) .GT. n_in_grid) THEN
                     igrid1 = igrid
                     ipos1 = idim
                     n_in_grid = grid_ndims(igrid)
                  ENDIF
  130           CONTINUE
  140        CONTINUE

* Adjust grids based on position already determined for the grid with most axes.

             DO 190 igrid = igrid_min, igrid_max
                IF (igrid .NE. igrid1) THEN
                   ipos2 = 0
                   igrid2 = 0
                   DO 150 idim = 1, 4
                      IF (grid_line(idim, igrid).EQ. iaxis .AND. 
     .                    idim .NE. ipos1)  THEN
                         igrid2 = igrid
                         ipos2 = idim
                      ENDIF
  150              CONTINUE

                   IF (igrid2 .NE. 0) THEN
                      i1 = MIN(ipos1, ipos2)
                      i2 = MAX(ipos1, ipos2)
                      shift_ok = .FALSE.
                      DO 160 idim = i1, i2
                         iax = grid_line(idim,igrid2)

                         IF (grid_line(idim,igrid2) .LE. 0 .OR.
     .                       line_direction(iax) .EQ. 'NA') 
     .                       shift_ok = .TRUE.
  160                 CONTINUE
                      IF (shift_ok) THEN
                         DO 170 idim = 1, 4
                            grid_save(idim) = grid_line(idim,igrid2)
  170                    CONTINUE
                         grid_line(ipos1,igrid2) = grid_line(ipos1,igrid1)
			 line_use_dir(grid_line(ipos1,igrid1)) = ipos1
                         grid_line(ipos2,igrid2) = mpsnorm

                         DO 180 idim = ipos1+1, 4
                            grid_line(idim,igrid2) = grid_save(idim-1)
                            IF (grid_save(idim-1) .NE.mpsnorm) need_iterate = .TRUE.
  180                    CONTINUE

                         reordered = reordered + 1
                      ELSE
	                 vlen = TM_LENSTR1(line_name(iaxis)) 

	                 CALL TM_NOTE('Could not adjust grid2 '// 
     .                         grid_name(igrid2)(1:vlen), lunit_errors)

	                 CALL TM_NOTE('Axes in grids may be inconsistent.',
     .                         lunit_errors)
                      ENDIF
    
                   ENDIF  ! igrid2 .ne. 0
                ENDIF
             
  190        CONTINUE  ! Loop over all grids comparing with grid1

          ENDIF
  200 CONTINUE ! disordered

      iter = iter + 1
      IF (iter .LT. 10) GOTO 1000
* Here loop back if need_iterate is true.

* Reset ds_grid_start and ds_grid_end

 2000 CONTINUE 
      IF (reordered .GT. 0) THEN
* * * * loop through each variable in this TMAP data set * * *
      DO 230 tmvar = 1,maxvars
         IF ( ds_var_setnum(tmvar) .NE. dset ) GOTO 230
         igrid = ds_grid_number(tmvar)
	 ndim = 0
	 DO 210 idim = 1, 4
	    iaxis = grid_line(idim, igrid)
	    ds_grid_start(idim,tmvar) = 1
	    ds_grid_end(idim,tmvar) = 1
            IF (iaxis .EQ. 0) THEN
	       ds_ordering(idim,tmvar) = 0
	    ELSE
	       ds_grid_end(idim,tmvar) = line_dim(iaxis)
	       ndim = ndim + 1
	       ds_ordering(ndim,tmvar) = idim
	       IF ( reversed(iaxis) ) ds_ordering(ndim,tmvar)
     .			      = -1 * ds_ordering(ndim,tmvar)
	       IF ( perm(idim).NE.unspecified_int4 .AND. perm(idim).LT.0) 
     .		  ds_ordering(ndim,tmvar) = -1 * ds_ordering(ndim,tmvar)
	    ENDIF
  210    CONTINUE 

* Now apply user-supplied ordering and reversals
	 permuted  = .FALSE.

        lastdim = 0
        DO 220 idim = 1, 4
           tmplines(idim) = ds_ordering(idim,tmvar)

           IF (perm(idim).EQ.0 .OR. perm(idim).EQ.unspecified_int4) GOTO 220
           IF ( ABS(perm(idim)) .LE. lastdim ) permuted = .TRUE.
           IF ( perm(idim).NE.unspecified_int4 .AND. perm(idim).LT.0 )
     .		   permuted = .TRUE.
           lastdim = ABS(perm(idim))
 220     CONTINUE

* get the variable name
         IF (permuted) THEN
            ivar = cd_varid(tmvar)
            CALL CD_GET_VAR_INFO(dset, ivar, vname, vartyp, nvdims,
     .                 vdims, nvatts, coordvar, all_outflag, istat)

            vlen = TM_LENSTR1(vname)
            CALL TM_AXIS_ORDER(tmplines, perm, ds_ordering(1,tmvar),
     .                           nvdims, vname(:vlen))
         ENDIF

  230 CONTINUE
      ENDIF

      RETURN
      END
