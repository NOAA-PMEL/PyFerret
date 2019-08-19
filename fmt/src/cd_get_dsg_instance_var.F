        SUBROUTINE CD_GET_DSG_INSTANCE_VAR(dset, cdfid, ivcd, ivtm, vdims, 
     .                                     nvdim, npts, idim, stored_lm, status)


*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*.
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
* Set up a special DSG instance variable from a netCDF dataset

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* v720 4/2017 *sh*
* V74+ 4/18 *acm* Getting DSG grids
* V74+ 4/18 *acm* Allow for the NCEI-style datasets which are one feature
*                 per file but they put the obs variables on 2D grids, 
*                 temp(feature, nobs)
* V74+ 6/18 *acm* Store the min and max row-size lengths in attribute actual_range
* V74+ 6/18 *acm* Apply time units and calendar from the file
* V741 7/18 *acm* Handle missing-flags and details of bad or missing coordinates
* V74+ 7/18 *acm* Working with a rowsize variable that we have synthesized 
*                 when there's not one in the file. 
* V74+ 7/18 *acm* More robust handling of coordinate ranges
* V74+ 7/18 *acm* Attempt to set positive-up or positive-down for Z coordinates
* V74+ 8/18 *acm* For timeseries or trajectory plots, ignore an invalid Z-coordinate
*                 variable as a coordinate
* v751 *acm*  6/19 Allow for longer strings as attribute values - std names -->256


* argument definitions:
*       cdfid     - netCDF id number for already opened CDF file
*       dset      - dataset number for already opened CDF file
*       ivcd      - netCDF variable id number
*       ivtm      - XDSET_INFO COMMON variable number
*       vdims     - var may be on a 1-by-npts grid
*       nvdim     - number of dims, 1 or 2
*       npts      - number of values to read into memory
*       idim      - orientation of the coordinate (or "0" for rowSize)
*       stored_lm - dynamic memory line number of the instance variable read from the file 
*       status    - return status

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
        include 'xdyn_linemem.cmn_text'
        include 'xtm_grid.cmn_text'
        include 'xunits.cmn_text'

* argument declarations
        INTEGER  cdfid, dset, ivcd, ivtm, npts, idim, stored_lm, 
     .           vdims(*), nvdim, status

* local variable declarations:
        LOGICAL NC_GET_ATTRIB, TM_HAS_STRING, got_it, ok_cal, has_bad, got_z
        INTEGER TM_LENSTR1, TM_UNIT_ID, STR_SAME, 
     .          TM_GET_CALENDAR_ID, TM_UNITS_CAL, STR_UPCASE,
     .          dtype, attlen, vlen, attoutflag, iaxis,
     .          start(2), numpts(2), i, vtype, since_t0, days_neg4713,
     .		units, cal_id, slen, flen
        REAL    vals(2), delta, coord, coord_min, coord_max, bad
        CHARACTER CD_DSG_FEATURENAME*20, axname*9, buff*256, ftrname*20

* local parameter declaration
	LOGICAL	    do_warn
        INTEGER maxlen
        INTEGER*8 rqst_n
        PARAMETER (maxlen = 256      ,   ! size of std_name
     .		   do_warn = .TRUE. )

* If there was no RowSize variable the line is 1 long, npts = 1
* coord should be the # of obs

* initialize
        vlen = TM_LENSTR1(ds_var_code(ivtm))

	ftrname = CD_DSG_FEATURENAME( dsg_orientation(dset))
	flen = TM_LENSTR1(ftrname)


* allocate memory for the variable
        rqst_n = npts
        CALL TM_GET_LM_DYNMEM(rqst_n, stored_lm, status)
        IF (status .NE. merr_ok) GOTO 5000

* read the 1D variable from the file
        dtype = 0    ! never used

	IF (ivcd .LT. 0) THEN
	   coord = dsg_obs_dimlen(dset)
	   CALL PUT_LINE_COORD ( dsg_linemem(stored_lm)%ptr, 1, coord )   

	ELSE
	   IF (nvdim .EQ. 1) THEN
              CALL CD_RD_R8_ARR( cdfid, ivcd, 1, npts, dtype,
     .                          "reading "//ds_var_code(ivtm)(:vlen)
     .                          //" for DSG",
     .                           dsg_linemem(stored_lm)%ptr, status )
	   ELSE
	      start(1) = 1
	      start(2) = 1
	      numpts(1) = npts
	      numpts(2) = 1
	      CALL CD_RD_R8_BNDS( cdfid, ivcd, start, numpts, dtype,
     .                          "reading "//ds_var_code(ivtm)(:vlen)
     .                          //" for DSG",
     .                           dsg_linemem(stored_lm)%ptr, 
     .                           1, 1, 1, npts, status )
	   ENDIF
	ENDIF

        IF (status .NE. merr_ok) GOTO 5100

        IF (idim .EQ. pdsg_row_size) THEN


* Get the missing and fill-value attributes, use them to set ds_bad_flag.
* Consolidate to one bad-value 

	   CALL CD_GET_BAD_FLAGS (dset, ABS(ivcd), ds_bad_flag(ivtm), 
     .         ds_missing_flag(ivtm), bad_r8, 
     .         dsg_linemem(stored_lm)%ptr, npts)


* *** create the instance (E) axis for the dataset
	   axname = "features"
           iaxis = 0
	   CALL TM_MAKE_BASIC_AXIS( axname, 1.0D0, 1.0D0, npts,
     .				    iaxis, status )
	   IF ( status .NE. merr_ok ) GOTO 5000
	   line_direction(iaxis) = axis_orients(E_dim)
	   line_units(iaxis)    = un_name(pun_instance)
* ... save this axis to be used during grid creation
	   dsg_instance_axis = iaxis

* Does it have an actual_range attribute?  If not compute one and store it.

	   IF (ivcd .LT. 0 ) THEN

	      coord_min = dsg_obs_dimlen(dset)
	      coord_max = dsg_obs_dimlen(dset)
	      got_it = .FALSE.
	      ivcd = ABS(ivcd)

	   ELSE
	      got_it = NC_GET_ATTRIB( dset, ivcd, 'actual_range',
     .                 do_warn, ds_var_code(ivtm)(:vlen),
     .                 maxlen, attlen, attoutflag, 
     .                 buff, vals )

* Check it and also test for missing values. Cannot have missing row-size data
	      
	      bad = ds_bad_flag(ivtm)
	      coord_min = arbitrary_large_val8
	      coord_max = -1*arbitrary_large_val8

	      DO i = 1, lm_size(stored_lm)
	         coord = dsg_linemem(stored_lm)%ptr(i)
	         IF (coord .EQ. bad) GOTO 5300
	         IF (coord .LT. coord_min) coord_min = coord
	         IF (coord .GT. coord_max) coord_max = coord
	      ENDDO

     	   ENDIF ! < 0

* here could compare vals(1), coord_min and vals(2), coord_max
* Store the attribute actual_range

     	   IF (.NOT. got_it) THEN
              CALL CD_GET_VAR_TYPE (dset, ivcd, ds_var_code(ivtm), vtype, status)

              attlen = 2
              attoutflag = 0
              vals(1) = coord_min
              vals(2) = coord_max
              CALL CD_PUT_NEW_ATTR (dset, ivcd, 'actual_range', 
     .              vtype, attlen, attoutflag, buff, vals, status) 
     	   ENDIF


        ELSE  ! pdsg_row_size

* Get the missing and fill-value attributes, use them to set ds_bad_flag.
* Consolidate to one bad-value 

	   CALL CD_GET_BAD_FLAGS (dset, ivcd, ds_bad_flag(ivtm), 
     .         ds_missing_flag(ivtm), bad_r8, dsg_linemem(stored_lm)%ptr, npts)
     
	   bad = ds_bad_flag(ivtm)

* Missing coordinate data is not allowed. For now just in Time
* DSGTBD: tools to fill in missing-coordinates? 

	   IF (idim .EQ. T_dim) THEN
	      DO i = 1, lm_size(stored_lm)
                 coord = dsg_linemem(stored_lm)%ptr(i)
                 IF (coord .EQ. bad) GOTO 5400
	      ENDDO
	   ENDIF

* *** create a coordinate axis corresponding to this variable
*     disallow missing coordinate data.

* ... find the extremes of this coordinate variable
	   CALL CD_GET_DSG_COORD_EXTREMES(dset, ivcd, ivtm,
     .      idim, coord_min, coord_max, status)

	   IF (status.NE.merr_ok .OR. coord_min.EQ.arbitrary_large_val8 ) THEN
	      IF (idim .EQ. z_dim .AND. dsg_orientation(dset).EQ.Z_dim) THEN
	         GOTO 5600
	      ELSE
	         GOTO 5700  
	      ENDIF
	   ENDIF

* ...  Basic checks on data ranges   dsg_orientation(dset).EQ.Z_dim
*      Allow constant time, lon, lat on profile data, 
*      or constant lon,lat on timeseries data or profile data
*      or constant depth on timeseries or trajectory data.  For this case,
*      we just ignore depth as a coordinate

	   IF (coord_min.EQ.coord_max .AND. idim.EQ.t_dim ) THEN 
	      IF (dsg_feature_type(dset) .EQ. pfeatureType_profile) THEN 
	         CALL WARN ('Constant time step in '//ftrname(:flen)//'  file.')	     
	      ELSE
	         GOTO 5500
	      ENDIF
	   ENDIF
	   
	   IF (coord_min.EQ.coord_max .AND. (idim.EQ.x_dim .OR. idim.EQ.y_dim)  
     .        .AND. (dsg_feature_type(dset) .EQ. pfeatureType_profile 
     .         .OR.  dsg_feature_type(dset) .EQ. pfeatureType_Timeseries) ) THEN 
	         CALL WARN ('Constant lon/lat in '//ftrname(:flen)//' file.')	     
	   ENDIF

	   IF (coord_min.EQ.coord_max .AND. idim.EQ.z_dim) THEN 
	      IF (dsg_feature_type(dset) .EQ. pfeatureType_Timeseries
     .         .OR. dsg_feature_type(dset) .EQ. pfeatureType_Trajectory ) THEN
	         GOTO 5700 
	      ELSE IF (dsg_feature_type(dset).EQ.pfeatureType_profile) THEN
	         GOTO 5600
	      ENDIF
	   ENDIF

* ... create a simple axis
	   axname = "FEATURE_"//axis_dimname(idim) ! e.g. "FEATURE_T"
           IF (npts.EQ.1) THEN
              delta = 1.  ! arbitrary
           ELSE
	      delta = (coord_max-coord_min)/DBLE(npts-1)
           ENDIF
           iaxis = 0
	   CALL TM_MAKE_BASIC_AXIS( axname, coord_min, delta, npts,
     .				    iaxis, status )
	   IF ( status .NE. merr_ok ) GOTO 5100
	   line_direction(iaxis) = axis_orients(idim)

	   line_name_orig(iaxis) = ds_var_code(ivtm)
 
* ... decode and insert units

           got_it = NC_GET_ATTRIB( dset, ivcd, 'units',
     .                 do_warn, ds_var_code(ivtm)(:vlen),
     .                 maxlen, attlen, attoutflag, 
     .                 line_units(iaxis), vals )
           line_unit_code(iaxis) = TM_UNIT_ID(line_units(iaxis))

           IF (idim .EQ. T_dim) THEN

              IF (.NOT. got_it .OR. STR_SAME (buff(1:4), 'none') .EQ. 0) GOTO 5100 ! units = 'none'

* calendar if given
              line_cal_name(iaxis) = 'GREGORIAN'   ! default
              got_it = NC_GET_ATTRIB( dset, ivcd, 'calendar',
     .                 do_warn, ds_var_code(ivtm)(:vlen),
     .                 maxlen, attlen, attoutflag, 
     .                 buff, vals )
              IF (got_it) THEN
                 CALL CD_GET_CALENDAR_NAME( buff, ok_cal)
                 IF (.NOT.ok_cal) GOTO 5100  ! bad calendar attribute
		 line_cal_name(iaxis) = buff
              ENDIF

              cal_id = TM_GET_CALENDAR_ID (line_cal_name(iaxis))
 
* time origin
	      buff = line_units(iaxis)
	      since_T0 = MAX( INDEX(buff,'since'), INDEX(buff,'SINCE') )
              IF ( since_T0 .GT. 2 ) THEN
                 line_units(iaxis) = buff(:since_T0-1)  ! just the "tunits" part 

                 slen = TM_LENSTR1(buff)
                 IF ( slen .GT. since_T0 + 5) THEN
	 
                    CALL CD_GET_TIME_UNITS (buff, cal_id, line_units(iaxis), 
     .			            line_t0(iaxis), days_neg4713, status)
                    IF (status .NE. merr_ok) GOTO 5000

* Named calendar with unit=month or year gets length according to calendar's year length.
                    units = TM_UNIT_ID( line_units(iaxis) )
                    units = TM_UNITS_CAL( units, cal_id, .FALSE. )
		    
                    line_unit_code(iaxis) = units
                    line_tunit(iaxis) = un_convert(line_unit_code(iaxis))
                 ELSE
	            GOTO 5200
		 ENDIF

              ELSE  ! since_t0
	         GOTO 5200
              ENDIF

           ENDIF

* Variable has an axis attribute saying Z but we don't know if it's positve-up or down.
* Check for attributes. Start with an arbitrary assumption of profile datasets.

           IF (idim .EQ. z_dim) THEN
	      got_z = .FALSE.

	      IF (dsg_feature_type(dset) .EQ. pfeatureType_profile) 
     .            line_direction(iaxis) = axis_orients(7) 

* is there a standard name indicating height or altitude/
	      got_it = NC_GET_ATTRIB( dset, ivcd, 'standard_name',
     .                 do_warn, ds_var_code(ivtm), maxlen, attlen, attoutflag, 
     .                 buff, vals )

	      IF (got_it) THEN  
	         IF ( TM_HAS_STRING(buff, 'height') ) line_direction(iaxis) = axis_orients(idim)
	         IF ( TM_HAS_STRING(buff, 'altitude') ) line_direction(iaxis) = axis_orients(idim)
	         IF ( TM_HAS_STRING(buff, 'depth') ) line_direction(iaxis) = axis_orients(7)
	         got_z = .TRUE.
	      ENDIF

* is there an attribute positive= ? Not in the standard, but it would help us here.
	      got_it = NC_GET_ATTRIB( dset, ivcd, 'positive',
     .                 do_warn, ds_var_code(ivtm), maxlen, attlen, attoutflag, 
     .                 buff, vals )

	      IF (.NOT.got_z .AND. got_it) THEN
	         IF ( TM_HAS_STRING(buff, 'UP') ) THEN
		    line_direction(iaxis) = 'DU'
		    got_z = .TRUE.
	         ENDIF

	         IF ( TM_HAS_STRING(buff, 'DOWN') ) THEN
		    line_direction(iaxis) = 'UD'
		    got_z = .TRUE.
	         ENDIF

	      ENDIF

           ENDIF

! DSGTBD: need to think through how to handle longitude ranges -- ERDDAP
! always encodes using 0:360, which breaks Pacific ocean regions

! DSGTBD: users can get to the coordinates of a DSG via pseudo-vars. Need more?
! Maybe insert global dataset attributes here that point to
! the coordinate variable names, e.g. ':dsg_x = "longitude";'
! Note that I created a new routine utl/set_var_att_val.F to make it quicker
! to define new attributes ... though in the end I didn't use it.

* ... assign this axis to the dataset
           ds_basic_axes(idim,dset) = iaxis
        ENDIF

* save other important DSG feature axis information
        IF (idim .EQ. dsg_orientation(dset)) THEN
	   dsg_obs_axis	= iaxis    ! makes life easy during grid creation
	   dsg_obs_dimlen(dset) = npts
! DSGTBD: Are there feature types with no time axis? no lon/lat?
!         for now requiring x, y, t, and depth/height is optional. 7/18

	   IF (idim .EQ. T_dim) ds_time_axis(dset) = iaxis

        ENDIF   ! pdsg_row_size

* flag that this is a DSG axis
	line_dsg_dset(iaxis) = dset

* successful completion
        RETURN

* error exit
 5000   RETURN

* Corrupted coordinate variable or other errors.

 5100   CALL TM_FREE_LM_DYNMEM(stored_lm)
        stored_lm = int4_init
        GOTO 5000
 
 5200   CALL WARN ('Time coordinate must have calendar units: Time units are:'//
     .   line_units(iaxis)(:TM_LENSTR1(line_units)) )
        status = 0   ! soft error
        GOTO 5100
 
 5300   CALL WARN ('Missing values in the RowSize variable: '//
     .   ds_var_code(ivtm)(:vlen) )
        status = merr_dsg_grid
        GOTO 5100

 5400   CALL WARN ('Missing values in coordinate variable: '//
     .   ds_var_code(ivtm)(:vlen) )
        status = 0   ! soft error
        GOTO 5100

 5500   CALL WARN("Corrupted time coordinates in DSG file: "
     .                  //ds_var_code(ivtm)(:vlen)  )
        status = merr_dsg_grid
        GOTO 5100

 5600   CALL WARN("Corrupted vertical coordinate in DSG profile; no range or all missing: "
     .                  //ds_var_code(ivtm)(:vlen) )
        status = merr_dsg_grid
        GOTO 5100
	
 5700   CONTINUE
* bad coords in depth/altitude for a trajectory or timeseries dataset. 
* Get rid of this as a coord and continue.
        status = 0   ! soft error
        GOTO 5100

        END

