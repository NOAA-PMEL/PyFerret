	SUBROUTINE CD_DSG_SCAN_VARS(dset, cdfid, errstr, status)

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
* Setup the special variables and axes that are needed for a DSG dataset

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V720  4/17 *sh* 
* V74+  4/18 *acm* Getting DSG grids
* V74+ 4/18 *acm* Allow for the NCEI-style datasets which are one feature
*                 per file but they put the obs variables on 2D grids, 
*                 temp(feature, nobs)
* V74+ 4/18 *acm* Handle datasets with multiple varibles that have axis attributes
*                 in the same direction.  
* V74+ 6/18 *acm* Better handling/reporting of grid errors or inconsistencies
* V74+ 7/18 *acm* More on DSG attribute and grid errors. Correctly deallocate axes
* V74+ 7/18 *acm* Use axis attribute and secondarily standard_name to find coordinate
*                 variables. Improve checks for presence of coord vars. remove redundant
*                 calls in the second loop over nvars.
* V74+ 7/18 *acm* If no rowsize variable, for a single-feature file, synthesize one.
* V74+ 7/18 *acm* Better handling of grids of char variables
* V74+ 8/18 *acm* For timeseries or trajectory plots, ignore an invalid Z-coordinate
*                 variable as a coordinate
* V74+ 10/18 *acm* Add a test for whether row-size sums to the obs length 
* V74+  2/19 *acm* Set dsg_obs_dim using the dimension of the coordinate variable in the  
*                  direction of the feature-type instead of time. Some of the NCEI 
*                  examples have a single time per profile, so obs varies in z not t
* V75+  5/19 *acm* Test the value of cf_role attribute for validity before using it.
* v751 *acm*  6/19 Allow for longer strings as attribute values
* V76  1/20 *acm* working with Point-type dsg data 
* V76  3/20 *acm* Improve notes about ambiguous coord info.
* v7.6 *acm* 5/20 Issue 1876: can read string-typed data, write as char.
* V7.6 *acm* 6/20 Issue 1980: When possible open invalid DSG datasets anyway with NOTES.
* V7.61 *acm* 7/20 Adding logic for TrajectoryProfile and TimeseriesProfile datasets.

* argument definitions:
*       dset    - pointer to TMAP data set
*	cdfid   - id of currently open CDF file
*	status	- return status
*
* for TrajectoryProfile and TimeseriesProfile datasets, notes from the CF standard:
*
* H.5.3. Ragged array representation of time series profiles
* ...[The standard] uses the contiguous ragged array representation for each profile (9.5.43.3), 
* and the indexed ragged array representation to organise the profiles into time series (9.3.54). 
* The canonical use case is when writing real-time data streams that contain profiles from many 
* stations, arriving randomly, with the data for each entire profile written all at once. 
*
* H.6.3. Ragged array representation of trajectory profiles
* ...[The standard]  uses the contiguous ragged array representation for each profile (9.3.3), and 
* the indexed ragged array representation to organise the profiles into time series (9.3.4). 
* The canonical use case is when writing real-time data streams that contain profiles from many 
* trajectories, arriving randomly, with the data for each entire profile written all at once.
*
* We will use feature to talk about the profiles for these data types and ftrset to refer to the
* trajectory or the timeseries station, containing a set of profiles.

* include files
        include 'netcdf.inc'
	include 'tmap_errors.parm'
#include "gt_lib.parm"
	include 'tmap_dims.parm'
#include "tmap_dset.parm"
#include "tmap_dtypes.parm"
	include 'xdset_info.cmn_text'
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include 'xbuild_grids.cmn'
	external xdset_info_data
        include 'xdset_info.cd_equiv'
        include 'xio.cmn_text'
	include 'xtext_info.cmn' 

* argument declarations:
	INTEGER	      dset, cdfid, status
	CHARACTER*(*) errstr

* local variable declarations:
*  ivtm - TMAP library variable ID
*  ivcd - netCDF file variable ID
	LOGICAL NC_GET_ATTRIB, TM_LEGAL_NAME,
     .		got_it, got_axis, got_stdname, coordvar, got_a_variable,
     .		has_DSG_row_size, has_DSG_coordinates, has_DSG_feature_id, 
     .		need_coord(4), udsg_err, cmpnd_ftype, redo_lonlat
	INTEGER STR_UPCASE, TM_LENSTR1,
     .		NCF_INQ_DS, STR_CASE_BLIND_COMPARE, STR_SAME,
     .		ivcd, ivtm, vartyp, nvdim, vdims(8), nvatts, all_outflag,
     .		i, ii, maxlen, vlen, slen, eax_len, ax_len, ax2_len, fax_len, 
     .		is_dim,cdfstat, attlen, attoutflag, istat, las_pos, 
     .		first_pos,dset_num, ndims, nvars, ngatts, recdim, idim,
     .		ax_att_var(4), ax_att_provisional(4), isend, ivtm1, orient,
     .		index_varid, index_var_lm, dsg_time_var, ignore_dim, dset_ftype
	CHARACTER vname*128, buff*1028, buff1*128, dname*128, axdim*1, 
     .		lunits*64, dir_code*2, index_dir_buff*20, ftrname*20, ignore_what*1
	REAL	vals(2), bad

	CHARACTER*20 CD_DSG_FEATURENAME
	CHARACTER*2  direc(4)
	CHARACTER*5  dirname(4)

* local parameter definitions:
	LOGICAL	    do_warn
	INTEGER     str_eq
	PARAMETER ( str_eq = 0,
     .		    do_warn = .TRUE. )

	DATA	direc(1) / 'XY' /,
     .		direc(2) / 'XY' /,
     .		direc(3) / 'Z ' /,
     .		direc(4) / 'T' /
	DATA	dirname(1) / 'lon ' /,
     .		dirname(2) / 'lat ' /,
     .		dirname(3) / 'vert' /,
     .		dirname(4) / 'time' /

* initialize

	has_DSG_row_size    = .FALSE.
	has_DSG_coordinates = .FALSE.
	has_DSG_feature_id  = .FALSE.
        maxlen = 1028
	eax_len = int4_init
	fax_len = int4_init
	las_pos = 1
	bad = -1.E34
	dsg_instance_axis = int4_init
	dsg_instnc_ftrset_axis = int4_init
	dsg_ftrset_var(dset) = int4_init
	ignore_dim = unspecified_int4
	ignore_what = ' '
	udsg_err = .FALSE.
	redo_lonlat = .FALSE.

* The feature-id variable is identified by having a feature_id attribute
* according to the feature type

* get the vital statistics for the data set 
        dset_num = dset
        CALL CD_GET_DS_INFO (dset_num, ndims, nvars, ngatts, recdim, status)
	istat = NCF_INQ_DS (dset_num, ndims, nvars, ngatts, recdim )

	orient = dsg_orientation(dset) 


* If its TimeseriesProfile must (for now) use/feature=profile

	CALL CD_DSG_GET_FILE_FEATURETYPE (dset, dset_ftype, do_warn)
	IF ( (dset_ftype .EQ. pfeatureType_TrajectoryProfile .OR.
     .        dset_ftype .EQ. pfeatureType_TimeseriesProfile) .AND. 
     .        dsg_user_direction .NE. z_dim) GOTO 5600

* If its TimeseriesProfile and they said use/feature=profile, need to
* get info on the lon/lat coordinates from the timeseries dimension and
* put it into the profile-axis direction.

	IF (dsg_user_direction .EQ. z_dim) THEN

	   IF (dset_ftype .EQ. pfeatureType_TrajectoryProfile .OR.
     .         dset_ftype .EQ. pfeatureType_TimeseriesProfile)
     .                       orient = dset_ftype
	   redo_lonlat =  (dset_ftype .EQ. pfeatureType_TimeseriesProfile)

	ENDIF

	cmpnd_ftype = (orient .EQ. pfeatureType_TrajectoryProfile .OR.
     .                 orient .EQ. pfeatureType_TimeseriesProfile)


	ftrname = CD_DSG_FEATURENAME ( orient ) 
	IF (cmpnd_ftype) THEN
	   i = MAX(INDEX(ftrname, 'Profile'), INDEX(ftrname, 'profile'))
	   index_dir_buff = ftrname(:i-1)//'_id'
	ENDIF

* First find probable coordinate variabes - if the axis attribute is
* present these will be the coords even if other variables have units
* indicating an axis direction. This to distinguish e.g between deployment
* time and location vs the observation times and locations.

* Prioritize attributes for direction: 
*  axis attribute: X, Y, Z, T
*  standard_name:  time, longitude, latitude, height/depth 
*         (distinguish from e.g. time_coverage_start or  geospatial_lon_min)
* When these are found, use units to nail down the direction.
*
* Check: need lon, lat, time for all feature types. Profiles need depth/height.
* Can get away without z direction for trajectories or timeseries.

	DO idim = 1, 4
	   ax_att_var(idim) = int4_init
	   ax_att_provisional(idim) = int4_init
	   need_coord(idim) = .TRUE.
	   IF ((orient.EQ.x_dim .OR. orient.EQ.t_dim) .AND. 
     .          idim.EQ.z_dim ) need_coord(z_dim) = .FALSE.
	ENDDO

*** Here as we're searching for the coordinate variables, also look for the 
*** instance_dimension attribute for TimeseriesProfile or TrajectoryProfile datasets. 
***   This will simplify finding the essential elements of those datasets.

        DO ivcd = 1, nvars
           CALL CD_GET_VAR_INFO (dset, ivcd, vname, vartyp,
     .                 nvdim, vdims, nvatts, coordvar, 
     .                 all_outflag, status)
	   
           vlen = TM_LENSTR1(vname)
	   got_axis = NC_GET_ATTRIB( dset, ivcd, 'axis',
     .                 do_warn, vname(:vlen), maxlen, attlen, attoutflag, 
     .                 buff, vals )

	   got_stdname = NC_GET_ATTRIB( dset, ivcd, 'standard_name',
     .                 do_warn, vname(:vlen), maxlen, attlen, attoutflag, 
     .                 buff, vals )

           IF (got_stdname) THEN  ! is it a direction?
	      IF (STR_SAME (buff, 'time')      .EQ. 0 .OR. 
     .            STR_SAME (buff, 'longitude') .EQ. 0 .OR.
     .            STR_SAME (buff, 'latitude')  .EQ. 0 .OR.
     .            STR_SAME (buff, 'depth')     .EQ. 0 .OR. 
     .            STR_SAME (buff, 'altitude')  .EQ. 0 ) THEN
	      ELSE 
	         got_stdname = .FALSE.
	      ENDIF
           ENDIF
	   
	   IF (got_axis .OR. got_stdname) THEN  ! look at units for direction
	   
              lunits = ''
              got_it = NC_GET_ATTRIB( dset, ivcd, 'units',
     .                 do_warn, vname(:vlen), maxlen, attlen, attoutflag, 
     .                 buff, vals )
              IF ( got_it) lunits = buff(:64)

              CALL CD_GET_LINE_DIRECTION (dset, ivcd,
     .             vname(:vlen), lunits, do_warn, .TRUE., dir_code)
              status = STR_UPCASE(axdim, dir_code(1:1))

              IF (dir_code .EQ. 'NA') CYCLE

              is_dim = 0
	      IDIMLOOP: DO idim = 1, 4
                 ! 'X','Y','Z','T'
                 IF (axdim .EQ. ww_dim_name(idim)) is_dim = 1
                 IF (idim .EQ. 1 .AND. axdim .EQ. 'W') is_dim = 2
                 IF (idim .EQ. 2 .AND. axdim .EQ. 'S') is_dim = 2
                 IF (idim .EQ. 3 .AND. (axdim .EQ. 'U' .OR. axdim .EQ. 'D')) is_dim = 2
                 IF (idim .EQ. 4 .AND. (dir_code .EQ. 'TI')) is_dim = 2

*    Prioritize variable with standard name or axis attrib

                 IF (is_dim .GT. 0) THEN

                    IF (ax_att_var(idim).NE.int4_init .AND.got_axis) THEN 
* ... corrupted file:  two variables claiming to be same coordinate
                       ivtm = ax_att_var(idim)
		       CALL CD_GET_VAR_VARNAME (dset, ivtm, buff, status)
		       IF (axdim .EQ. 'W') axdim = 'E'
		       IF (axdim .EQ. 'S') axdim = 'N'
		       IF (axdim .EQ. 'U' .OR. axdim .EQ. 'D') axdim = 'Z'
		       CALL WARN('Ambiguous DSG coordinate info. '//
     .                        'Multiple variables claiming to be '//
     .                         ww_dim_name(idim)//' axis: '// buff(:TM_LENSTR1(buff) )//
     .                         ', '// vname(:vlen) //'. Using '//
     .                         buff(:TM_LENSTR1(buff))//'.'   )
                    ELSE
		       IF (got_axis .AND. is_dim.EQ.2) ax_att_var(idim) = ivcd
		       IF (got_stdname .AND. .NOT.got_axis) ax_att_provisional(idim) = ivcd
		       IF (got_axis .AND. is_dim.EQ.1) ax_att_provisional(idim) = ivcd
                    ENDIF
		    EXIT IDIMLOOP

                 ENDIF ! is_dim
              ENDDO  IDIMLOOP
     
           ENDIF  ! got axis or std name

* For Trajectory-Profile or Timeseries-Profile data, locate the 
* trajectory or timeseries id variable and its dimension.

	   IF (cmpnd_ftype) THEN

	      got_it = NC_GET_ATTRIB( dset, ivcd, 'instance_dimension',
     .                 do_warn, vname(:vlen), maxlen, attlen, attoutflag, 
     .                 buff, vals )
	      IF (got_it) THEN
	         CALL CD_GET_DIM_ID (dset, buff, dsg_instnc_ftrset_dim)
		 index_varid = ivcd
	      ENDIF

	      got_it = NC_GET_ATTRIB( dset, ivcd, 'cf_role',
     .                 do_warn, vname(:vlen), maxlen, attlen, attoutflag, 
     .                 buff, vals )

* Is the cf_role either trajectory_id or timeseries_id? 

	      IF (got_it) THEN
	         IF (STR_SAME (buff, index_dir_buff) .EQ. 0 ) dsg_ftrset_var(dset) = ivcd

* the dataset is traj-profile or tser-profile, but we are doing use/featrure=profile.
* Get the trajectory or timeseries dimension and ignore variables in that direction.

		 IF (orient .EQ. pfeatureType_Profile) THEN
		    IF (STR_SAME (buff, 'trajectory_id') .EQ. 0 ) THEN
		       dsg_ftrset_var(dset) = ivcd
		       CALL CD_GET_VAR_DIMS_INFO (dset, ivcd, vname, nvdim, 
     .                     vdims, status)
		       ignore_dim = vdims(nvdim)
		       ignore_what = 'X'
		    ENDIF
		    IF (STR_SAME (buff, 'timeseries_id') .EQ. 0 ) THEN
		       dsg_ftrset_var(dset) = ivcd
		       CALL CD_GET_VAR_DIMS_INFO (dset, ivcd, vname, nvdim, 
     .                     vdims, status)
		       ignore_dim = vdims(nvdim)
		       ignore_what = 'T'
		    ENDIF 

	         ENDIF
	      ENDIF

	   ENDIF !cmpnd_ftype

        ENDDO ! ivcd

* Are there required directions not yet found?  Try any found from std name, otherwise
* this is an error.

	DO idim = 1, 4
	   IF (ax_att_var(idim).EQ.int4_init) 
     .             ax_att_var(idim) = ax_att_provisional(idim)
	ENDDO

	vname = ' '
	slen = 1
	DO idim = 1, 4
	   IF (ax_att_var(idim).EQ.int4_init .AND. need_coord(idim)) THEN
	      IF (slen .EQ. 1) THEN
	         vname(slen:) = dirname(idim)
		 slen = slen+4
              ELSE
	         vname(slen:) = ', '//dirname(idim)
                 slen = slen+6
	      ENDIF
	   ENDIF
	ENDDO
	IF (slen .GT. 1) THEN
	   slen = TM_LENSTR1( ftrname ) 
	   errstr = 'Required coordinate variables not found. '//
     .	          ftrname(:slen)//
     .            ' file is missing directions: '//vname(:vlen) 
	   status = merr_dsg_struc
	   GOTO 6000
	ENDIF

	got_a_variable = .FALSE.
	dsg_ifeature_dim = no_dim

	dsg_time_var = ax_att_var(t_dim)  ! used in setting up for timeseriesProfile


*** process the key instance variables in the file
        ivcdloop: DO ivcd = 1,nvars

* get the vital statistics for this variable
           CALL CD_GET_VAR_INFO (dset, ivcd, vname, vartyp,
     .                 nvdim, vdims, nvatts, coordvar, 
     .                 all_outflag, status)
           IF ( status .NE. merr_ok ) GOTO 5400
           vlen = TM_LENSTR1(vname)

           IF (vartyp .EQ. NF_CHAR) THEN
              nvdim = nvdim - 1
	      vdims(1) = vdims(2)
	      IF (nvdim .GT. 1)  vdims(2) = vdims(3)
	   ENDIF

* Are we ignoring timeseries-direction or trajectory-direction variables
* in a use/feature=profile command?

	   IF (vdims(1) .EQ. ignore_dim) CYCLE
****
* register the variable as a dependent variable in the dataset
	   
	   CALL CD_GET_1_DEP_VAR ( dset, cdfid, ivcd, las_pos, bad, 
     .				   ivtm, status )
	   IF (status .NE. merr_ok) GOTO 5400
	   IF ( ivtm .NE. 0 ) THEN
	      las_pos = ivtm	! optimizes next hunt for slot
	      IF (.NOT.got_a_variable) THEN
	         first_pos = ivtm
	         got_a_variable = .TRUE.
	      ENDIF
	   ENDIF

* Make a coordinate axis for the timeseries or trajectories in the compound datasets.

	   IF (cmpnd_ftype .AND. ivcd .EQ. dsg_ftrset_var(dset)) THEN
	      CALL CD_GET_DS_DIMS (dset, vdims(1), dname, ax_len, status)

	      fax_len = ax_len
	      CALL CD_GET_DSG_INSTANCE_VAR(dset, cdfid, ivcd, ivtm,
     .                 nvdim, fax_len, pdsg_fs_dim, dsg_time_var,
     .                 dsg_loaded_lm(ivtm), errstr, status)

	      CYCLE ivcdloop
	   ENDIF

* Save the index varible for the timeseries or trajectory dimension in compound datasets.
* its dimension is the number of profiles.

	   IF (cmpnd_ftype .AND. ivcd .EQ. index_varid) THEN
	      dsg_index_var(dset) = ivtm
	      CALL CD_GET_DSG_INSTANCE_VAR(dset, cdfid, ivcd, ivtm,
     .                 nvdim, ax_len, pdsg_indx_var, dsg_time_var,
     .                 dsg_loaded_lm(ivtm), errstr, status)

	      index_var_lm = dsg_loaded_lm(ivtm)
	      IF (eax_len .NE. int4_init .OR. status.NE.merr_ok) THEN
	         IF (ax_len .NE. eax_len) GOTO 5500
	      ENDIF
	      CYCLE ivcdloop
	   ENDIF

	   IF (nvdim.EQ.0) CYCLE  ivcdloop  ! scalar variables

****
* Check for the feature-id variable and store it so it can be read.
* Ideally would read the id's for use in labels and possibly masks. ?

* It is identified by its "cf_role" attribute, with value feature_id, eg. 
*  :cf_role = "trajectory_id" ;
* Compare the value: invalid to have a cf_role with value other than 
* e.g. trajectory_id for one of the valid feature-types.

	   got_it = NC_GET_ATTRIB( dset, ivcd, 'cf_role',
     .                 do_warn, vname(:vlen), maxlen, attlen, attoutflag, 
     .                 buff, vals )
	   IF (got_it) THEN

* IF we found a trajectory_id or timeseries_id for a compound-feature-type
* dataset, the variable hadn't got an ivtm yet.
	      IF (ivcd .EQ. dsg_ftrset_var(dset)) dsg_ftrset_var(dset) = ivtm

	      buff1 = ftrname(:TM_LENSTR1(ftrname))//'_id'

* For the compound types, look for profile_id. There is also a trajectory_id
* or timeseries_id, taken care of elsewhere.

	      IF (cmpnd_ftype) buff1 = 'profile_id'

* Get the dimension of the ID variable, eax_len

	      IF (STR_SAME (buff, buff1) .EQ. 0) THEN
	         has_DSG_feature_id = .TRUE.
	         dsg_feature_var(dset) = ivtm 

	         cdfstat = NF_INQ_DIMLEN(cdfid, vdims(1), eax_len)
	         CALL CD_GET_DS_DIMS (dset, vdims(1), dname, eax_len, status)
	         dsg_ifeature_dim = vdims(1)
	         CYCLE  ivcdloop

	      ELSE

* For the compound types, already have the trajectory_id and timeseries_id

	         IF (cmpnd_ftype) CYCLE  ivcdloop

* check other valid feature types: If use/feature=timeseries for a trajectory dataset 
* then that is valid.  If use/feature=profile for a timeseriesprofile or trajectoryprofile
* dataset then will ignore the cf_role="timeseries_id" or cf_role="trajectory_id" attrib.

	         IF (dset_ftype .EQ. x_dim) buff = 'timeseries_id' 
		 IF (STR_SAME (buff, buff1) .EQ. 0) THEN
		    has_DSG_feature_id = .TRUE.
		    dsg_feature_var(dset) = ivtm 

		    cdfstat = NF_INQ_DIMLEN(cdfid, vdims(1), eax_len)
		    CALL CD_GET_DS_DIMS (dset, vdims(1), dname, eax_len, status)
		    dsg_ifeature_dim = vdims(1)
		    CYCLE  ivcdloop
		 ENDIF

c		 CALL WARN('Invalid cf_role attribute ignored on variable '//
c     .               vname(:vlen) )
	      ENDIF

	   ELSE
	      IF (vartyp .EQ. NF_CHAR) CYCLE ! no string var coords or rowSize
	   ENDIF 

! DSGTBD - currently code is very rigid about what a DSG file can have
! DSGTBD - can have scalars, instance vars and obs vars.  Else reject as a DSG.
! - NCEI template for single feature does not include a RowSize variable.
!          Create one on the fly, of length 1, value npts.

! The NCEI templates all have the obs variables on grids of (obs, traj) where traj = 1.
! Allow for this case but not other multi-dimensioned variables.

	   cdfstat = NF_INQ_DIMLEN(cdfid, vdims(1), ax_len)
	   CALL CD_GET_DS_DIMS (dset, vdims(1), dname, ax_len, status)

	   IF (nvdim .GT. 1) THEN
	      cdfstat = NF_INQ_DIMLEN(cdfid, vdims(2), ax2_len)
	      CALL CD_GET_DS_DIMS (dset, vdims(2), dname, ax2_len, status)
	      IF (ax2_len .GT. 1) THEN
	         errstr = 'DSG ragged array: observation variables must be 1-D '//vname(:vlen)
	         IF (status .NE. merr_dsg_grid) status = merr_dsg_struc
	         GOTO 6000
	      ENDIF
	   ENDIF

* the DSG row size variable is identified by its "sample_dimension" attribute
           got_it = NC_GET_ATTRIB( dset, ivcd, 'sample_dimension',
     .                 do_warn, vname(:vlen), maxlen, attlen, attoutflag, 
     .                 buff, vals )
	   IF (got_it) THEN
	      ax_len = 0
	      IF (dsg_ifeature_dim .GT. no_dim) THEN
	         cdfstat = NF_INQ_DIMLEN(cdfid, dsg_ifeature_dim, ax_len)
	         CALL CD_GET_DS_DIMS (dset, dsg_ifeature_dim, dname, ax_len, status)
	      ENDIF
	      IF (ax_len.EQ.0) THEN
* ... first encounter with instance axis
	         dsg_ifeature_dim = vdims(1)
		 
	         cdfstat = NF_INQ_DIMLEN(cdfid, dsg_ifeature_dim, eax_len)
	         CALL CD_GET_DS_DIMS(dset, dsg_ifeature_dim, dname, eax_len, status)
	      ENDIF
	      dsg_row_size_var(dset) = ivtm
	      dsg_row_size_varid(dset) = ivcd
	      CALL CD_GET_DSG_INSTANCE_VAR(dset, cdfid, ivcd, ivtm,
     .                 nvdim, eax_len, pdsg_row_size, dsg_time_var,
     .                 dsg_loaded_lm(ivtm), errstr, status)

	      IF (status .NE. merr_ok) THEN
	         IF (status .NE. merr_dsg_grid) status = merr_dsg_struc
	         GOTO 6000
	      ENDIF
	      has_DSG_row_size = .TRUE.

* ... the sample_dimension attribute provides the name of the obs dimension.
*     Get its id number from its name.
	      slen = TM_LENSTR1(buff)
	      cdfstat = NF_INQ_DIMID ( cdfid, buff(:slen), dsg_obs_dim)
	      CALL CD_GET_DIM_ID (dset, buff(:slen), dsg_obs_dim)
	      IF ( cdfstat .NE. NF_NOERR )  GOTO 5400
	      CYCLE 
	   ENDIF

* the DSG coordinate arrays

	   DO idim = 1, 4
	      IF (ax_att_var(idim) .NE. ivcd) CYCLE

	      IF (dsg_coord_var(idim,dset) .EQ. int4_init) THEN

		 CALL CD_GET_DSG_INSTANCE_VAR(dset, cdfid, ivcd, 
     .	             ivtm, nvdim, ax_len, idim, dsg_time_var,
     .	             dsg_loaded_lm(ivtm), errstr, status)

		 IF (status .EQ. 0) THEN 
		     CYCLE  ! corrupted var, ignore as a coord
		 ENDIF

		 IF (status .NE. merr_ok) THEN
		    IF (status .NE. merr_dsg_grid) status = merr_dsg_struc
		    GOTO 6000
		 ENDIF
		 dsg_coord_var(idim,dset) = ivtm
		 IF (idim .EQ. orient) has_DSG_coordinates = .TRUE.

	      ENDIF
	   ENDDO
	   CYCLE ivcdloop  ! unrecognized DSG axis orientation. 

	ENDDO ivcdloop

	IF (.NOT.has_DSG_coordinates .AND. cmpnd_ftype ) THEN
	   
	   ndims = 0
	   DO idim = 1, 4
	      IF (ax_att_var(idim) .NE. int4_init) ndims = ndims + 1
	   ENDDO
	   
	   has_DSG_coordinates = (ndims .EQ. 4)
	ENDIF

* If this is a timeseriesProfile dataset, then the X-Y coordinates are in
* the timeseries instance-dimension direction. Use the station index variable
* to redefine the dsg-coordiate variables for the translation grid to list these 
* coordinates in the profile dimension.  The test when orient is profile is
* for the case when the user has said USE/feature=profile for a TimeseriesProfile set.

	IF (redo_lonlat) CALL CD_DSG_TP_TO_PROF (dset)

* A valid DSG contiguous ragged dataset must have a rowSize variable
* and coordinates on the dsg_orientation axis. If they erroneously set
* the /FEATURETYPE to an invalid direction, that's an error. Otherwise
* continue without dsg-ness.

	IF (.NOT.has_DSG_coordinates .AND. dsg_user_direction.NE.int4_init) GOTO 5100

* Point type data does not have a RowSize variable and may not have a feature-id variable.
* All the variables are on the instance axis, so find that axis in the first variable
* and nominally set the feature-id variable to the time variable and obs_dimlen to e-axis length

	IF (dsg_feature_type(dset) .EQ. pfeatureType_Point) THEN
	   IF (.NOT. has_DSG_feature_id ) THEN
	      
              ivcd = 1
	      CALL CD_GET_VAR_INFO (dset, ivcd, vname, vartyp,
     .                 nvdim, vdims, nvatts, coordvar, 
     .                 all_outflag, status)
              IF ( status .NE. merr_ok ) GOTO 5400

              IF (vartyp .EQ. NF_CHAR) THEN
                 nvdim = nvdim - 1
	         vdims(1) = vdims(2)
	         IF (nvdim .GT. 1)  vdims(2) = vdims(3)
	      ENDIF
	      cdfstat = NF_INQ_DIMLEN(cdfid, vdims(1), eax_len)
	      CALL CD_GET_DS_DIMS (dset, vdims(1), dname, eax_len, status)
	      has_DSG_feature_id = .TRUE.
	      dsg_feature_var(dset) = dsg_coord_var(t_dim,dset) 
	      dsg_ifeature_dim = vdims(1)
	      dsg_obs_dimlen(dset) = eax_len
	      dsg_row_size_var(dset) = 1

	   ENDIF
	   
	ENDIF


* Single-feature files may not have the RowSize variable (NCEI standard unfortunately)
* Use the length of the feature-id variable, and the length of the time axis
* to set things up. Construct a dataset-variable ferRowSize.

	IF (.NOT. has_DSG_row_size ) THEN
	   IF (has_DSG_feature_id) THEN
	      CALL CD_GET_VAR_DIMS_INFO (dset, dsg_feature_var(dset), vname, nvdim, 
     .            vdims, status)
	
	      IF (dsg_ifeature_dim .EQ. no_dim) THEN
	          cdfstat = NF_INQ_DIMLEN(cdfid, vdims(1), eax_len )
		  CALL CD_GET_DS_DIMS (dset, vdims(1), dname, eax_len, status)
	          IF (eax_len .EQ. 1) dsg_ifeature_dim = vdims(1)
	      ENDIF

	      IF (eax_len.EQ.1 .OR. dsg_feature_type(dset).EQ.pfeatureType_Point) THEN
	      
* ... instance axis is the 'feature' axis - may already be set

	         DO ivtm = 1, maxvars
	            IF ( STR_SAME(ds_var_code(ivtm),char_init) .EQ. 0 ) exit
		 ENDDO
	         ds_var_setnum(ivtm)   = dset
		 vname = 'ferRowSize'
	         CALL string_array_modify(ds_var_code_head, ivtm, 
     .               vname, 10)

	         ds_var_title(ivtm)    = 'synthesized rowsize'
	         ds_var_titl_mod(ivtm) = ' '
	         ds_var_units(ivtm)    = 'count'
	         ds_missing_flag(ivtm) = bad
	         ds_bad_flag(ivtm)     = bad
	         ds_var_type(ivtm)     = ptype_float

	         ds_grid_start(e_dim,ivtm) = 1
	         ds_grid_end(e_dim,ivtm)   = eax_len
		 
		 ivcd = nvars + 1
		 cd_scale_factor(ivtm) = 1.
		 cd_add_offset(ivtm)   = 0.
		 cd_scaled(ivtm)       = .FALSE.
		 cd_varid(ivtm)       = ivcd  

* Add the synthesized variable to the linked list structure
		 nvdim = 1  ! and vdims is as for feature-id var, gotten above.
		 
		 vname = 'ferRowSize'
		 coordvar = .FALSE.

! use ivtm not ivcd, so that  cd_varid(ivtm) = ivtm is correct?
		 CALL CD_ADD_VAR( dset, vname, ivcd, ds_var_title(ivtm), 
     .             ds_var_units(ivtm), NCFLOAT, coordvar, ds_bad_flag(ivtm), status)

!! check:     
		 CALL CD_GET_VAR_INFO (dset, ivcd, buff, vartyp,
     .                 nvdim, vdims, nvatts, coordvar, 
     .                 all_outflag, status)

* Point data has neither a Rowsize variable nor an OBS axis. Synthesize
* rowsize as an axis that's 1:npoints, each value being 1, and an obs
* axis of length npoints

	         IF (dsg_feature_type(dset) .EQ. pfeatureType_Point) THEN


* Don't need to allocate memory for the synthesized Obs axis
* just use the basic-axis call in CD_DSG_SETUP_PTDATA (?)

	            DO ivtm1 = ivtm, maxvars
	               IF ( STR_SAME(ds_var_code(ivtm1),char_init) .EQ. 0 ) exit
		    ENDDO

		    CALL CD_DSG_SETUP_PTDATA (dset, eax_len, 
     .                 dsg_loaded_lm(ivtm), dsg_loaded_lm(ivtm1), status)
	         ELSE
		    isend = -1*ivcd
		    CALL CD_GET_DSG_INSTANCE_VAR(dset, cdfid, isend, ivtm,
     .                 nvdim, eax_len, pdsg_row_size, dsg_time_var,
     .                 dsg_loaded_lm(ivtm), errstr, status)

	         ENDIF

		 IF (status .NE. merr_ok) THEN
		    IF (status .NE. merr_dsg_grid) status = merr_dsg_struc
		    GOTO 6000
		 ENDIF

	         dsg_row_size_var(dset) = ivtm
	         dsg_row_size_varid(dset) = unspecified_int4
	         has_DSG_row_size = .TRUE.

* ... Set dsg_obs_dim using dimension of the coordinate variable in the direction of the feature-type
*     For point data just nominally use the first variable with a direction.

	         idim = orient
		 IF (idim .EQ. e_dim) THEN
		    DO idim = 1, 4
		       IF (ax_att_var(idim) .NE. int4_init) EXIT  ! from loop
		    ENDDO
		 ENDIF
		 CALL CD_GET_VAR_DIMS_INFO (dset, ax_att_var(idim),  
     .             vname, nvdim, vdims, status)
	         IF (nvdim .EQ. 1) THEN
		    dsg_obs_dim = vdims(1)
		 ELSE
		    DO idim = 1, nvdim
		       IF (vdims(idim) .NE. dsg_ifeature_dim) dsg_obs_dim = vdims(idim)
		    ENDDO
		 ENDIF

	      ENDIF

	   ENDIF

	   IF ( .NOT.has_DSG_feature_id .OR. (eax_len.GT.1 .AND.  
     .             dsg_feature_type(dset).NE.pfeatureType_Point) ) THEN
	      errstr = 'Row Size variable w/ sample_dimension attribute not found. ' 
	      status = merr_dsg_struc
	      GOTO 6000
	   ENDIF

	ENDIF

*   * * * * * * * * * * * * * * * * * * * * * * * * 
* pass through the variable names making sure that case-sensitive
* duplicates (e.g. 'var' and 'VAR') retain case and that names which
* would be illegal variable names to Ferret are retained in original form
	IF (got_a_variable) THEN
	  DO 310 ivtm = first_pos,las_pos
	    IF (ds_var_setnum(ivtm) .NE. dset) GOTO 310
	    IF (.NOT.TM_LEGAL_NAME(ds_var_code(ivtm))) GOTO 310  ! leave as-is
* ... see if this name case-blind duplicates any other in the data set
	    DO 300 i = first_pos,las_pos
* ... *kob* v541 8/02 - need to also check and make sure we are still dealing with the
*     same dataset.  if not, when variables from different data sets are interleaved 
*     in ds_var_code this can cause Ferret to not properly upcase variables from 
*     one dset that have the same name as variables in another dset. 
	      IF ( i .EQ. ivtm .OR. ds_var_setnum(i) .NE. dset) GOTO 300
	      istat = STR_CASE_BLIND_COMPARE(ds_var_code(ivtm),
     .                                       ds_var_code(i)    )
	      IF ( istat .EQ. str_eq ) GOTO 310                  ! leave as-is
 300	    CONTINUE
* ... (finally) upcase the name
*	  istat = STR_UPCASE( ds_var_code(ivtm), ds_var_code(ivtm) )
          CALL STRING_ARRAY_MODIFY_UPCASE(ds_var_code_head,ivtm,
     .                                    ds_var_code(ivtm),
     .                                    LEN(ds_var_code(ivtm)))
 310    CONTINUE
	ENDIF

* Checklist:
*  coordinate variables for at least, X, Y, T. (checked above)
*  feature-id variable
*  row-size variable  (checked above)

	IF (.NOT. has_DSG_feature_id) THEN
	   errstr = 'Feature-ID variable not found or '//
     .      'invalid value of cf_role attribute'
	   status = merr_dsg_struc
	   GOTO 6000
	ENDIF

	IF (cmpnd_ftype) THEN
	   buff = ''

	   IF (fax_len .EQ. int4_init) THEN
	      buff = 'ID variable not found:'//index_dir_buff(:TM_LENSTR1(index_dir_buff))
	      vlen = TM_LENSTR1(buff)
	   ENDIF
	ENDIF

*  No mask is set, but initialize the memory location to unspecified_int4
	dsg_msk_lm(dset) = unspecified_int4

* check the time data: is it monotonically increasing?  For now this is
* an error; later perhaps we'll sort unsorted data internally? This checks 
* for strict monotonic.  Could perhaps use tm_check_coords-type logic

* The routine also checks that rowsize sums to the observation-axis length

	IF ( dsg_feature_type(dset).NE.pfeatureType_Point) 
     .     CALL CD_DSG_CHECK_TIMECOORD ( eax_len,  dsg_obs_dimlen(dset),
     .     dsg_loaded_lm(dsg_row_size_var(dset)), 
     .     dsg_loaded_lm(dsg_coord_var(t_dim,dset)), errstr, status )

	IF (status .NE. merr_ok) THEN
	   IF (status .NE. merr_dsg_grid) status = merr_dsg_struc
	   GOTO 6000
	ENDIF
	
	
*...  No mask for timeseries or trajecotry on tseries-profile or traj-profile data
*      is set, but initialize the memory location to unspecified_int4

	IF (cmpnd_ftype) dsg_mask_ftrset_lm(dset) = unspecified_int4

* successful completion
	status = merr_ok
 5000	RETURN

* Invalid DSG: Clean up initialization as DSG and go back, try to 
* initialize it as a non-DSG file

 6000   CONTINUE
	udsg_err = (.NOT.udsg_err .AND. dsg_user_direction .NE. int4_init)

	IF (dsg_instance_axis .NE. int4_init) CALL TM_DEALLO_DYN_LINE(dsg_instance_axis)
	IF (dsg_instnc_ftrset_axis .NE. int4_init) CALL TM_DEALLO_DYN_LINE(dsg_instnc_ftrset_axis)
	DO idim = 1, nferdims
	   IF (ds_basic_axes(idim,dset) .NE. int4_init)
     .        CALL TM_DEALLO_DYN_LINE(ds_basic_axes(idim,dset) )
	ENDDO
	CALL CD_RELEASE_DSG_INSTANCE_VARS(dset)
	DO  i = 1, maxvars
	   IF (ds_var_setnum(i) .EQ. dset) ds_var_setnum(i) = set_not_open
	ENDDO
	dsg_ragged(dset) = .FALSE.


* hard errors

* It's an error if they try USE/FEATURE= on invalid DSG dataset
	IF (udsg_err) THEN
	   buff = 'Requested USE/FEATURE='//ftrname(:vlen)//
     .        ', but dataset is not a valid DSG file '
	   CALL TM_ERRMSG (merr_invalid, status, 'CD_DSG_SCAN_VARS',
     .         no_descfile, no_stepfile,
     .         buff, no_errstring, *5000)
        ENDIF

	GOTO 5000
  
 5100   vlen = TM_LENSTR1(ftrname)
	buff = 'USE/FEATURE='//ftrname(:vlen)//
     .     ' ... must have observations along '//direc(dsg_user_direction)
	udsg_err = .TRUE.
	CALL TM_ERRMSG (merr_dsg_grid, status, 'CD_DSG_SCAN_VARS',
     .     no_descfile, no_stepfile, buff, no_errstring, *6000)

 5400   CALL TM_ERRMSG (merr_invalid, status, 'CD_DSG_SCAN_VARS',
     .     no_descfile, no_stepfile,
     .     buff, no_errstring, *6000)

 5500   vlen = TM_LENSTR1(ftrname)
	buff = 'Invalid Index variable in '//ftrname(:vlen)//' dataset. '//
     .         'Incorrect length or values not in range of num-profiles'
	CALL TM_ERRMSG (merr_invalid, status, 'CD_DSG_SCAN_VARS',
     .     no_descfile, no_stepfile,
     .     buff, no_errstring, *6000)

 5600   ftrname = CD_DSG_FEATURENAME ( dset_ftype ) 
	vlen = TM_LENSTR1(ftrname)
	buff = ftrname(:vlen)//' Feature Type not yet implemented. Try "'//
     .         'USE/FEATURE=profile "'
	CALL TM_ERRMSG (merr_invalid, status, 'CD_DSG_SCAN_VARS',
     .     no_descfile, no_stepfile,
     .     buff, no_errstring, *6000)

	RETURN
	END
