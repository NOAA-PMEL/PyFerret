	SUBROUTINE CD_DSG_SCAN_VARS(dset, cdfid, status)

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
*  derivative works and supporting documentation.  Further, the user5/4/2005
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

* argument definitions:
*       dset    - pointer to TMAP data set
*	cdfid   - id of currently open CDF file
*	status	- return status

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

* local variable declarations:
*  ivtm - TMAP library variable ID
*  ivcd - netCDF file variable ID
	LOGICAL NC_GET_ATTRIB, TM_LEGAL_NAME,
     .		got_it, got_axis, got_stdname, coordvar, got_a_variable,
     .		has_DSG_row_size, has_DSG_coordinates,
     .		has_DSG_feature_id, need_coord(4)
	INTEGER STR_UPCASE, TM_LENSTR1,
     .		NCF_INQ_DS, STR_CASE_BLIND_COMPARE, STR_SAME,
     .		ivcd, ivtm, vartyp, nvdim, vdims(8), nvatts, all_outflag,
     .		i, maxlen, vlen, slen, eax_len, ax_len, ax2_len, is_dim,
     .		cdfstat, attlen, attoutflag, istat, las_pos, first_pos,
     .		dset_num, ndims, nvars, ngatts, recdim, idim,
     .		ax_att_var(4), ax_att_provisional(4), isend, ivtm1, orient
	CHARACTER    vname*128, buff*1028, buff1*128, name*8, axdim*1, lunits*64, dir_code*2
	REAL	vals(2), global_bad

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
	eax_len = 0
	las_pos = 1
	global_bad = -1.E34
	dsg_instance_axis = int4_init

* The feature-id variable is identified by having a feature_id attribute
* according to the feature type

* get the vital statistics for the data set 
        dset_num = dset
        istat = NCF_INQ_DS (dset_num, ndims, nvars, ngatts, recdim )
	orient = dsg_orientation(dset) 

* First find probable coordinate variabes - if the axis attribute is
* present these will be the coords even if other variables have units
* indicating an axis direction. This to distinguish e.g between deplooyment
* time and location vs the observation times and locations.

* Prioritize attributes for direction: 
*  axis attribute: X, Y, Z, T
*  standard_name:  time, longitude, latitude, height/depth 
*         (distinguish from e.g. time_coverage_start or  geospatial_lon_min)
* When these are found, use units to nail down the direction.
*
* Check: need lon, lat, time for all feature types. Profiles need depth/height.

	DO idim = 1, 4
	   ax_att_var(idim) = int4_init
	   ax_att_provisional(idim) = int4_init
	   need_coord(idim) = .TRUE.
	   IF (idim.EQ.z_dim .AND. orient.NE.z_dim) need_coord(z_dim) = .FALSE.
	ENDDO

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
	   
	   IF (vartyp.LT.nf_byte .OR. vartyp.GT.nf_string) GOTO 5700

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
	   buff1 = CD_DSG_FEATURENAME(orient)
	   vlen = TM_LENSTR1( buff1 ) 
	   buff = 'Required coordinate variables not found. '//
     .	          buff1(:vlen)//
     .            ' file is missing directions: '//vname(:slen)

	   GOTO 5300
	ENDIF
	
	got_a_variable = .FALSE.
	dsg_instance_dim = no_dim

*** process the key instance variables in the file
        ivcdloop: DO ivcd = 1,nvars

* get the vital statistics for this variable
           CALL CD_GET_VAR_INFO (dset, ivcd, vname, vartyp,
     .                 nvdim, vdims, nvatts, coordvar, 
     .                 all_outflag, status)
           IF ( status .NE. merr_ok ) GOTO 5200
           vlen = TM_LENSTR1(vname)

           IF (vartyp .EQ. NF_CHAR) THEN
              nvdim = nvdim - 1
	      vdims(1) = vdims(2)
	      IF (nvdim .GT. 1)  vdims(2) = vdims(3)
	   ENDIF


****
* register the variable as a dependent variable in the dataset
	   CALL CD_GET_1_DEP_VAR ( dset, cdfid, ivcd, las_pos, global_bad, 
     .				   ivtm, status )
	   IF (status .NE. merr_ok) GOTO 5200
	   IF ( ivtm .NE. 0 ) THEN
	      las_pos = ivtm	! optimizes next hunt for slot
	      IF (.NOT.got_a_variable) THEN
	         first_pos = ivtm
	         got_a_variable = .TRUE.
	      ENDIF
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
	      
	      buff1 = CD_DSG_FEATURENAME ( orient ) ! compare with buff...
	      buff1 = buff1(:TM_LENSTR1(buff1))//'_id'

	      IF (STR_SAME (buff, buff1) .EQ. 0) THEN
	         has_DSG_feature_id = .TRUE.
	         dsg_feature_var(dset) = ivtm 

	         cdfstat = NF_INQ_DIMLEN(cdfid, vdims(1), eax_len)
	         dsg_instance_dim = vdims(1)
	         CYCLE  ivcdloop

* check other valid feature types (if use/feature= then the type may not match)
	      ELSE

	         DO idim = 1, 4
		    IF (idim .EQ. 2) CYCLE 
		    buff1 = CD_DSG_FEATURENAME ( idim ) 
		    buff1 = buff1(:TM_LENSTR1(buff1))//'_id'
	            IF (STR_SAME (buff, buff1) .EQ. 0) THEN
		       has_DSG_feature_id = .TRUE.
		       dsg_feature_var(dset) = ivtm 

		       cdfstat = NF_INQ_DIMLEN(cdfid, vdims(1), eax_len)
		       dsg_instance_dim = vdims(1)
	               CYCLE  ivcdloop
		    ENDIF
		 ENDDO

		 CALL WARN('Invalid cf_role attribute ignored on variable '//
     .               vname(:vlen) )
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

	   IF (nvdim .GT. 1) THEN
	      cdfstat = NF_INQ_DIMLEN(cdfid, vdims(2), ax2_len)
	      IF (ax2_len .GT. 1) THEN
	         buff = 'DSG ragged array: observation variables must be 1-D '//vname(:vlen)
		 GOTO 5400
	      ENDIF
	   ENDIF

* the DSG row size variable is identified by its "sample_dimension" attribute
           got_it = NC_GET_ATTRIB( dset, ivcd, 'sample_dimension',
     .                 do_warn, vname(:vlen), maxlen, attlen, attoutflag, 
     .                 buff, vals )
	   IF (got_it) THEN
	      cdfstat = NF_INQ_DIMLEN(cdfid, dsg_instance_dim, ax_len)
	      IF (ax_len.EQ.0) THEN
* ... first encounter with instance axis
	         dsg_instance_dim = vdims(1)
	         cdfstat = NF_INQ_DIMLEN(cdfid, dsg_instance_dim, eax_len)
	      ENDIF
	      dsg_row_size_var(dset) = ivtm
	      dsg_row_size_varid(dset) = ivcd
	      CALL CD_GET_DSG_INSTANCE_VAR(dset, cdfid, ivcd, ivtm,
     .                 vdims, nvdim, eax_len, pdsg_row_size,
     .                 dsg_loaded_lm(ivtm), status)
	      IF (status .NE. merr_ok) GOTO 5200
	      has_DSG_row_size = .TRUE.

* ... the sample_dimension attribute provides the name of the obs dimension.
*     Get its id number from its name.
              slen = TM_LENSTR1(buff)
	      cdfstat = NF_INQ_DIMID ( cdfid, buff(:slen), dsg_obs_dim)
	      IF ( cdfstat .NE. NF_NOERR ) GOTO 5200
	      CYCLE 
	   ENDIF

* the DSG coordinate arrays

	   DO idim = 1, 4
	      IF (ax_att_var(idim) .NE. ivcd) CYCLE

	      IF (dsg_coord_var(idim,dset) .EQ. int4_init) THEN

		 CALL CD_GET_DSG_INSTANCE_VAR(dset, cdfid, ivcd, ivtm,
     .	             vdims, nvdim, ax_len, idim, dsg_loaded_lm(ivtm), status)
		 IF (status .EQ. 0) THEN 
		     CYCLE  ! corrupted var, ignore as a coord
		 ENDIF

		 IF (status .NE. merr_ok) GOTO 5200
		 dsg_coord_var(idim,dset) = ivtm
		 IF (idim .EQ. orient)
     .			has_DSG_coordinates = .TRUE.

		 EXIT 
	      ENDIF
	   ENDDO
	   CYCLE ivcdloop  ! unrecognized DSG axis orientation. 

	ENDDO ivcdloop

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
              IF ( status .NE. merr_ok ) GOTO 5200

              IF (vartyp .EQ. NF_CHAR) THEN
                 nvdim = nvdim - 1
	         vdims(1) = vdims(2)
	         IF (nvdim .GT. 1)  vdims(2) = vdims(3)
	      ENDIF
	      cdfstat = NF_INQ_DIMLEN(cdfid, vdims(1), eax_len)

	      has_DSG_feature_id = .TRUE.
	      dsg_feature_var(dset) = dsg_coord_var(t_dim,dset) 
	      dsg_instance_dim = vdims(1)
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
	
	      IF (dsg_instance_dim .EQ. no_dim) THEN
	          cdfstat = NF_INQ_DIMLEN(cdfid, vdims(1), eax_len)
	          IF (eax_len .EQ. 1) dsg_instance_dim = vdims(1)
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
	         ds_missing_flag(ivtm) = global_bad
	         ds_bad_flag(ivtm)     = global_bad
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
     .                 vdims, nvdim, eax_len, pdsg_row_size,
     .                 dsg_loaded_lm(ivtm), status)
	         ENDIF

		 IF (status .NE. merr_ok) GOTO 5200

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
		       IF (vdims(idim) .NE. dsg_instance_dim) dsg_obs_dim = vdims(idim)
		    ENDDO
		 ENDIF

	      ENDIF

	   ENDIF

	   IF ( .NOT.has_DSG_feature_id .OR. (eax_len.GT.1 .AND.  
     .             dsg_feature_type(dset).NE.pfeatureType_Point) ) THEN
	      buff = 'Row Size variable w/ sample_dimension not found. '
	      GOTO 5300
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
	   buff = 'Feature-ID variable not found or invalid value of cf_role attribute'
	   GOTO 5300
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
     .     dsg_loaded_lm(dsg_coord_var(t_dim,dset)), buff, status )

	IF (status .NE. merr_ok) GOTO 5600
	
* successful completion
	status = merr_ok
 5000	RETURN

* hard errors

 5500   CONTINUE
	IF (dsg_instance_axis .NE. int4_init) CALL TM_DEALLO_DYN_LINE(dsg_instance_axis)
	DO idim = 1, nferdims
	   IF (ds_basic_axes(idim,dset) .NE. int4_init)
     .        CALL TM_DEALLO_DYN_LINE(ds_basic_axes(idim,dset) )
	ENDDO
	CALL CD_RELEASE_DSG_INSTANCE_VARS(dset)
	dsg_ragged(dset) = .FALSE.
	GOTO 5000
  
 5100   buff1 = CD_DSG_FEATURENAME(dsg_user_direction)
	buff = 'USE/FEATURE='//buff1(:vlen)//
     .     ' ... must have observations along '//direc(dsg_user_direction)
	CALL TM_ERRMSG (merr_dsg_grid, status, 'CD_DSG_SCAN_VARS',
     .     no_descfile, no_stepfile, buff, no_errstring, *5500)

 5200   buff = ' '
	CALL TM_ERRMSG
     .     ( merr_dsg_struc, status, 'CD_DSG_INIT',
     .     no_descfile, no_stepfile,
     .     buff, no_errstring, *5500 )

 5300   CALL TM_ERRMSG (merr_dsg_struc, status, 'CD_DSG_SCAN_VARS',
     .     no_descfile, no_stepfile,
     .     buff, no_errstring, *5500)

 5400   CALL TM_ERRMSG (merr_invalid, status, 'CD_DSG_SCAN_VARS',
     .     no_descfile, no_stepfile,
     .     buff, no_errstring, *5500)

 5600   CALL TM_ERRMSG (merr_dsg_struc, status, 'CD_DSG_SCAN_VARS',
     .     no_descfile, no_stepfile,
     .     buff, no_errstring, *5500)


 5700   CALL TM_ERRMSG
     .     ( merr_notsupport, status, 'CD_SCAN_VARS', dset_num,
     .         no_stepfile, 
     .         'String Data Type not implemented: variable '//vname(:vlen),
     .         no_errstring,*5500)

	RETURN
	END
