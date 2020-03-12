	SUBROUTINE CD_DSG_GRIDS ( dset, cdfid, temp_axnams, status )

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
* Loads the common block XGRID with grid definitions from a CF-DSG
* netCDF file 
* also initialize grid limits (lo hi bounds on each axis) for these variables

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V720 4/17 *sh*
* V74+ 4/18 *acm* Getting DSG grids
* V74+ 4/18 *acm* Allow for the NCEI-style datasets which are one feature
*                 per file but they put the obs variables on 2D grids, 
*                 temp(feature, nobs)
* V74+ 4/18 *acm* Allow scalar variables on normal grid
* V74+ 6/18 *acm* Check the feature direction and report errors. 
* V74+ 7/18 *acm* Handle a rowsize variable that we have synthesized 
*                 when there's not one in the file. 
* V74+ 7/18 *acm* Better handling of grids of char variables
* V76  1/20 *acm* Fix access to rowsize variable. 
*                 Handle point datasets. 
* V76  1/20 *acm* Orientation of Point-type data set to e_dim

* argument definitions:
*       dset        - TMAP data set pointer
*	cdfid	    - netCDF id number for already opened CDF file
*       temp_axnams - nferdims axis names per grid: temp until grids are assembled
*	status	    - return status

* include files

        include 'netcdf.inc'
	include 'tmap_errors.parm'
	include 'tmap_dims.parm'
	include 'errmsg.parm'
#include "tmap_dset.parm"
#include "gt_lib.parm"
        include 'xbuild_grids.cmn'
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'xio.cmn_text'
	external xio_data
	include 'xtm_grid.cmn_text'
	        include 'xdset_info.cd_equiv'

* argument declarations
	INTEGER	 dset, cdfid, status
	CHARACTER*128 temp_axnams(nferdims,max_temp_grid)

* local variable declarations:
* *kob* explicit definitions not allowed on decl line in linux...
*       moved them to parameter line 10/96
      LOGICAL coordvar, one_by_nobs
      INTEGER TM_LENSTR1, TM_FIND_GRID_SLOT,
     .	      i, tmvar, ivar, vartyp, nvdims, vdims(8), nvatts, vlen,
     .	      all_outflag, idim, tmpgrid, obs_orient, igrid, grid_cnt,
     .	      istat, ax_len, check_dir
      CHARACTER name*128, vname*128
      CHARACTER*20 CD_DSG_FEATURENAME
      CHARACTER*2  direc(4)

      DATA	direc(1) / 'XY' /,
     .		direc(2) / 'XY' /,
     .		direc(3) / 'Z ' /,
     .		direc(4) / 'T' /

* initialize temporary grid name pointer
	num_tmp_grids = 0
	DO i = max_grids, grid_ceiling
	   tmp_nam_ptr(i) = 0
	ENDDO
	obs_orient = dsg_orientation(dset)
	
	check_dir = x_dim
	IF (obs_orient .EQ. pfeatureType_Trajectory) check_dir = x_dim
	IF (obs_orient .EQ. pfeatureType_Profile   ) check_dir = z_dim
	IF (obs_orient .EQ. pfeatureType_TimeSeries) check_dir = t_dim 

* * * * loop through each variable in this TMAP data set * * *
* if it doesn't have a grid pointer create a suitable one and point to it
        DO tmvar = 1,maxvars
           IF ( ds_var_setnum(tmvar) .NE. dset ) CYCLE
           ivar = cd_varid(tmvar)

* get the vital statistics for this variable
           CALL CD_GET_VAR_INFO(dset, ivar, vname, vartyp, nvdims,
     .                 vdims, nvatts, coordvar, all_outflag, istat)
           IF (istat.NE.merr_ok) CALL TM_ERRMSG
     .     ( istat+pcdferr,status,'CD_DSG_GRIDS',cdfid,ivar,
     .       no_errstring, no_errstring, *5900 )
           vlen = TM_LENSTR1(vname)

* The rowsize variable may have been constructed in cd_dsg_scan_vars. If so,
* use the grid from the feature-id variable.

	   IF (tmvar .EQ.  dsg_row_size_var(dset)) THEN
	      IF (vname .EQ. 'ferRowSize' .AND.
     .            ds_grid_number(dsg_feature_var(dset)) .NE. int4_init) THEN
	         i = dsg_feature_var(dset)
		 ds_grid_number(tmvar) = ds_grid_number(i)
	         DO idim = 1, nferdims
                    ds_grid_start(idim,tmvar) = ds_grid_start(idim,i)
                    ds_grid_end  (idim,tmvar) = ds_grid_end  (idim,i)
	         ENDDO

		 GOTO 5000
	      ENDIF 
	   ENDIF


* if a string variable, the first dimension is the string axis (ignore it)
	   IF (vartyp .EQ. NF_CHAR) THEN
	      IF (nvdims .GT. 1) nvdims = nvdims - 1
	      vdims(1) = vdims(2)
	   ENDIF

* use the next grid building slot to build a grid for this variable
           tmpgrid = num_tmp_grids + 1

* initialize the grid axis info
	   DO idim = 1, nferdims
              temp_axnams(idim,tmpgrid) = 'NORMAL'
	      ds_ordering  (idim,tmvar) = 0
              ds_grid_start(idim,tmvar) = 1
              ds_grid_end  (idim,tmvar) = 1
	   ENDDO

* all DSG grids have the E axis of instances (scalars handled below).
             
	   temp_axnams(5,tmpgrid) = line_name(dsg_instance_axis)
	   ds_grid_end(5,tmvar)   = line_dim (dsg_instance_axis)

* assign the remaining grid info
	   IF (vdims(1) .EQ. dsg_instance_dim) THEN
* ... 1D instance grid
	      ds_ordering(1,tmvar) = 5

* ... This allows for the NCEI-style files which are a single feature
*     with obs variables on a nobs by 1 grid

	      one_by_nobs = (nvdims.EQ.2 .AND. 
     .             line_dim (dsg_instance_axis).EQ.1 .AND. 
     .             vdims(2).EQ.dsg_obs_dim) 

	      IF (one_by_nobs) THEN
	         temp_axnams(obs_orient,tmpgrid) = line_name(dsg_obs_axis)
	         ds_grid_end(obs_orient,tmvar)   = line_dim (dsg_obs_axis)
	         ds_ordering(2,tmvar) = 5
	         ds_ordering(1,tmvar) = obs_orient
	      ENDIF

	   ELSEIF (vdims(1) .EQ. dsg_obs_dim) THEN
* ... 2D observation grid 
	      temp_axnams(obs_orient,tmpgrid) = line_name(dsg_obs_axis)
	      ds_grid_end(obs_orient,tmvar)   = line_dim (dsg_obs_axis)
	      ds_ordering(2,tmvar) = 5
	      ds_ordering(1,tmvar) = obs_orient

* scalar variables are allowed, normal grid.
	   ELSEIF (nvdims.EQ.0) THEN
              temp_axnams(5,tmpgrid) = 'NORMAL'
	      ds_grid_end(5,tmvar)   = 1

	   ELSE
	      
* ... variable does not belong with this DSG dataset

* TODO: maybe other variables should be ignored (removed from dset)
*         rather than aborting the DSG processing

	      GOTO 4100
	   ENDIF

* Does the data have obs data along the direction relative to the feature type?
* e.g. timeseries feature type must have time on the obs axis, trajectory data
* must have data in X,Y

	   IF (tmvar .EQ. dsg_coord_var(check_dir, dset) .AND. 
     .         dsg_feature_type(dset).NE.pfeatureType_Point) THEN
	      IF ( ds_grid_end(obs_orient,tmvar) .NE. line_dim(dsg_obs_axis) ) 
     .              GOTO 4200
	   ENDIF

	   IF (nvdims.GE.2 .AND. line_dim (dsg_instance_axis).GT.1 .AND. 
     .             vdims(2).EQ.dsg_obs_dim) GOTO 4100

* ... does Ferret's "native taxis" apply to a DSG?  TBD
* TBD??:  we could create a dataset time axis using min/max and
* time encoding from the DSG time coordinate variable
* would that make the Ferret time code easier to use?
* Would need to pass this back as an argument
!	   native_tax = mpsnorm

* is the temporary grid we just created unique ?
* (code copied unaltered from CD_GET_GENERIC_GRIDS)
* DSGTBD: worth it to make this block a subroutine?
	   igrid = 0
 300	   CALL TM_NEXT_TMP_GRID( igrid, *320)
	      grid_cnt = tmp_nam_ptr(igrid)
              DO 310 idim = 1, nferdims
                 IF ( temp_axnams(idim,tmpgrid )
     .          .NE.  temp_axnams(idim,grid_cnt) ) GOTO 300
 310          CONTINUE
* ... not unique - exact match with one that already exists
              GOTO 350
 320	   CONTINUE
 
* allocate a new grid
* find next location to store info; if too many grids - give error
	  CALL TM_ALLO_TMP_GRID(igrid, status)
	  IF (status .NE. merr_ok) GOTO 5900

* make up a unique grid name "GXXnn" where XX is hashed from data set name
          grid_name(igrid) = 'G'//cd_dset_code(dset)(:2)
          CALL TM_NEW_GRID_NAME( grid_name(igrid), name )
          grid_name(igrid) = name

* basic defaults
          grid_rotation(igrid) = 0.0
          DO 340 idim = 1, nferdims
 340      grid_out_prod(idim,igrid) = .TRUE.

* save a pointer to this grid for the TMAP variable
 350      ds_grid_number(tmvar) = igrid

       ENDDO  ! loop tmvar over vars

* successful completion
 5000 	status = merr_ok
        RETURN

* error exits
* soft error - reject this as a dsg_ragged dataset, but soldier on as a netCDF
 4100	CALL TM_NOTE("invalid CF-DSG variable: "//vname(:vlen)//
     .		     " in file: "//ds_name(dset), lunit_errors)
! DSGTBD: Have not checked that aborted DSG leads to valid gridded file.
!     Probably not.  Need to decide how much trouble to go to.
	GOTO 5900

 4200	CONTINUE

	name = CD_DSG_FEATURENAME (dsg_user_direction)
	vlen = TM_LENSTR1( name )

 	CALL TM_ERRMSG (merr_dsg_grid, status, 'CD_DSG_GRIDS',
     .     no_descfile, no_stepfile,
     .     'FeatureType '//name(:vlen)//
     .     ' must have observations along '//direc(dsg_user_direction),
     .     no_errstring, *5900)

* error exit
 5900	CONTINUE
	CALL TM_GARB_COL_GRIDS (dset)
	CALL CD_RELEASE_DSG_INSTANCE_VARS(dset)
	dsg_ragged(dset) = .FALSE.
	
        RETURN
	END
