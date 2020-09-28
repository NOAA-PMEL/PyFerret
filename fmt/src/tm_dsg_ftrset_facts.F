	SUBROUTINE TM_DSG_FTRSET_FACTS( grid, dset, orientation, nfeatures, 
     .			nftrsets, is_dsg, is_cmpnd, status )

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
* return the feature orientation and netCDF dim length of the observations
* in a DSF file
* THIS ROUTINE SHOULD ONLY BE CALLED FOR GRIDS KNOWN TO BE DSG 

* V720 *sh* 4/17
* V74+ *acm* 9/18 Allow an abstract axis (e.g. merging grids)
* V761 *acm* 7/20 Return a flag for a compound type, timeseriesProfile or
*                 trajectoryProfile.
* V763 *acm* 9/20 Return if grid not defined.

      include 'netcdf.inc'
      include 'tmap_errors.parm'
      include 'tmap_dims.parm'
      include 'implicit.parm'
#include "gt_lib.parm"
#include "tmap_dset.parm"
      include 'xtm_grid.cmn_text'
      include 'xdset_info.cmn_text'
      include 'xfr_grid.cmn'

* argument declarations
	LOGICAL is_dsg, is_cmpnd
	INTEGER grid, dset, orientation, nfeatures, nftrsets

* internal variable declarations
	LOGICAL in_dset
	INTEGER TM_DSG_DSET_FROM_GRID, TM_DSG_NFEATURES, TM_DSG_NF2FEATURES,
     .		i, f_line, vgrid, mabstract, status

	mabstract = grid_line( x_dim, mgrid_abstract )
	status = merr_ok
	nfeatures = 0
	nftrsets = 0
	dset = 0
	
	IF (grid .LE. 0) RETURN

* For the ftrset grid, station-id or trajectory-id in a timeseriesProfile
* or trajectoryProfile dataset, it's the F direction\

	f_line = grid_line(f_dim,grid)
	IF (f_line .EQ. mpsnorm) THEN
	   is_dsg = .FALSE.
	   RETURN
	ENDIF
	dset = line_dsg_dset(f_line)
	
	IF (dset .EQ. pline_not_dsg) THEN
	   status = merr_dsg_grid  ! expecting ftrset-grid
	   RETURN
	ENDIF

	IF (f_line.EQ.mpsnorm .OR. f_line.EQ.mabstract) THEN
	   is_dsg = .FALSE.
	   GOTO 9999
	ENDIF

	orientation = dsg_orientation(dset)

	is_cmpnd = orientation.EQ.pfeatureType_TrajectoryProfile .OR. 
     .		   orientation.EQ.pfeatureType_TimeseriesProfile

* Get nfeatures for the dataset from the grid of an instance or obs variable.

	in_dset = .FALSE.
	DO i = 1, maxvars
	   IF ( ds_var_setnum(i) .EQ. dset) THEN
	      in_dset = .TRUE.
	      vgrid = ds_grid_number(i)
	      IF (vgrid.NE.int4_init .AND. 
     .            TM_DSG_DSET_FROM_GRID(vgrid) .EQ. dset) THEN
                 nfeatures = TM_DSG_NFEATURES(vgrid)
                 nftrsets = TM_DSG_NF2FEATURES (dset)
		 grid = vgrid  ! return a dsg- instance or obs grid
		 is_dsg = .TRUE.
		 EXIT  ! from loop
	      ENDIF
	   ELSE
	      IF (in_dset) EXIT  ! we've done this dataset. exit maxvars loop.
	   ENDIF

	ENDDO
	

 9999	CONTINUE
	IF (nfeatures .EQ. 0) status = merr_dsg_grid  ! expecting ftrset-grid

        RETURN

        END








