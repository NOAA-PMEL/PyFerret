      SUBROUTINE CD_GET_LINE_DIRECTION (dset, ivar,
     .     vname, linunits, do_warn, full, dir_code)
     

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

*  code to determine the direction of a coordinate axis. Pulled out of cd_get_1_axis
*  V683 ACM 9/2012
*
*  V692 ACM 7/2014 A cf_role attribute indicates the E direction.
*  V694 ACM 2/2015 Ensemble axis may have a standard-name attribute "realization"
*  V694 ACM 4/2015 Fix ticket 2260: set X axis when standard_name is "longitude"
*  V72+ ACM 10/2015 If using axis names, compare the right # of characters
*  V73+ ACM 12/2016 Change arguments slightly. Instead of passing in iaxis
*                   and setting dir_code, set a 2-letter code
*  V73+ ACM  1/2018 issue 1851: If direction is reduced down to using axis names, 
*                   see if there were unrelated units that should trump the name
* 
* Arguments
*  dset		Ferret dataset number
*  ivar		varid
*  vname	variable name
*  linunits	variable units
*  do_warn	Turn warnings on or off
*  full		True to use units, false to stop after looking for at the 
*               axis, coordinate_axis, CoordinateAxisType, cf_role, long_name, 
*               and standard_name.  If false, it does look for a time origin in the units
*  dir_code     return: two-letter direction code, either XX,YY,... or WE, SN, UD, DU, TI
*

*      include files: 
      
        IMPLICIT NONE
        include 'netcdf.inc'
        include 'tmap_errors.parm'
        include 'tmap_dims.parm'
#include "tmap_dset.parm"
        include 'xtm_grid.cmn_text'
        external xgt_grid_data
        include 'xunits.cmn_text'
        external xunits_data
        include 'errmsg.parm'


* calling arguments
      LOGICAL do_warn, full
      INTEGER dset, ivar, idir
      CHARACTER*(*) vname, linunits, dir_code

* local declarations
      LOGICAL TM_HAS_STRING, NC_GET_ATTRIB, got_it, yyyymmdd_time
      CHARACTER*1 axis_dir*1, vupcase*128

      INTEGER STR_UPCASE, STR_CASE_BLIND_COMPARE, TM_LENSTR1, 
     .	      TM_UNIT_ID, i, units, attlen, attoutflag, attype, iatt,  
     .        vlen, slen,maxlen, istat, since_T0, epic_t2var, status

      REAL val, dummy

      INTEGER bufflen
      PARAMETER (bufflen = 256)
      CHARACTER*256 buff, upbuff
      INTEGER     str_eq, no_units
      PARAMETER ( str_eq = 0, no_units = 0 )

* (could check for the record axis and if so, set it to T)

* Use attributes to try to determine the orientation and 
* direction of the axis

      dir_code = 'NA'

      vlen = TM_LENSTR1(vname)

* ... First check for an AXIS attribute

      axis_dir = char_init01
      maxlen = 1
      got_it = NC_GET_ATTRIB(dset, ivar, 'axis',
     .             do_warn, vname(:vlen), maxlen, attlen, 
     .             attoutflag, axis_dir, val)
      IF (got_it) THEN
         CALL CD_GET_VAR_ATT_ID (dset, ivar, 'axis', iatt, status)
         CALL CD_GET_VAR_ATT_INFO (dset, ivar, iatt, buff, attype, 
     .     attlen, attoutflag, status )
         IF (attype .NE. NCCHAR) got_it = .FALSE.
      ENDIF
      IF (got_it) THEN
         istat = STR_UPCASE( axis_dir, axis_dir )
         IF (axis_dir.EQ.'T' .OR. axis_dir.EQ.'L' .OR.
     .             axis_dir.EQ.'t' .or. axis_dir.eq.'l') THEN 
            dir_code = 'TI'  ! time
          ELSE IF (axis_dir.EQ.'X' .OR. axis_dir.EQ.'I' .OR.
     .             axis_dir.eq.'x' .or. axis_dir.eq.'i') THEN
            dir_code = 'XX'
         ELSE IF (axis_dir.EQ.'Y' .OR. axis_dir.EQ.'J' .OR.
     .            axis_dir.eq.'y' .or. axis_dir.eq.'j') THEN
            dir_code = 'YY'
         ELSE IF (axis_dir.EQ.'Z' .OR. axis_dir.EQ.'K' .OR.
     .            axis_dir.eq.'z' .or. axis_dir.eq.'k') THEN
            dir_code = 'DU'
         ELSE IF (axis_dir.EQ.'E' .OR. axis_dir.EQ.'M' .OR.
     .            axis_dir.eq.'e' .or. axis_dir.eq.'m') THEN
            dir_code = 'EE'
         ELSE IF (axis_dir.EQ.'F' .OR. axis_dir.EQ.'N' .OR.
     .            axis_dir.eq.'f' .or. axis_dir.eq.'n') THEN
            dir_code = 'FF'
         ENDIF
      ELSE
         axis_dir = char_init01
      ENDIF


* ... check for a CARTESIAN_AXIS attribute, if still undetermined

      IF (dir_code .EQ. 'NA') THEN

         maxlen = 1
         got_it = NC_GET_ATTRIB(dset, ivar, 'cartesian_axis',
     .              do_warn, vname(:vlen), maxlen, attlen, 
     .              attoutflag, axis_dir, val)
         IF (got_it) THEN
            CALL CD_GET_VAR_ATT_ID (dset, ivar, 'cartesian_axis', iatt, status)
            CALL CD_GET_VAR_ATT_INFO (dset, ivar, iatt, buff, attype, 
     .     attlen, attoutflag, status )
            IF (attype .NE. NCCHAR) got_it = .FALSE.
         ENDIF

         IF (got_it) THEN
            IF (axis_dir.EQ.'T' .OR. axis_dir.EQ.'L' .OR.
     .          axis_dir.EQ.'t' .or. axis_dir.eq.'l') THEN 
               dir_code = 'TI'  ! time
            ELSE IF (axis_dir.EQ.'X' .OR. axis_dir.EQ.'I' .OR.
     .               axis_dir.eq.'x' .or. axis_dir.eq.'i') THEN
            dir_code = 'WE'
            ELSE IF (axis_dir.EQ.'Y' .OR. axis_dir.EQ.'J' .OR.
     .               axis_dir.eq.'y' .or. axis_dir.eq.'j') THEN
             dir_code = 'SN'
            ELSE IF (axis_dir.EQ.'Z' .OR. axis_dir.EQ.'K' .OR.
     .               axis_dir.eq.'z' .or. axis_dir.eq.'k') THEN
               dir_code = 'DU'
            ELSE IF (axis_dir.EQ.'E' .OR. axis_dir.EQ.'M' .OR.
     .            axis_dir.eq.'e' .or. axis_dir.eq.'m') THEN
               dir_code = 'EE'
            ELSE IF (axis_dir.EQ.'F' .OR. axis_dir.EQ.'N' .OR.
     .            axis_dir.eq.'f' .or. axis_dir.eq.'n') THEN
               dir_code = 'FF'
            ENDIF
         ELSE
            axis_dir = char_init01
         ENDIF
      ENDIF

* ... check for a cf_role attribute, if still undetermined.
*     Put these axes in the E direction. This attribute may
*     be on a non-coordinate variable. Those are found in 
*     cd_get_generic_grids.

      IF (dir_code .EQ. 'NA') THEN

         maxlen = 20
         got_it = NC_GET_ATTRIB(dset, ivar, 'cf_role',
     .              do_warn, vname(:vlen), maxlen, attlen, 
     .              attoutflag, buff, val)
         IF (got_it) dir_code = 'EE'
      ENDIF

* ... check for a _CoordinateAxisType attribute, if still undetermined

      IF (dir_code .EQ. 'NA') THEN

         maxlen = 20
         got_it = NC_GET_ATTRIB(dset, ivar, '_CoordinateAxisType',
     .              do_warn, vname(:vlen), maxlen, attlen, 
     .              attoutflag, buff, val)
         IF (got_it) THEN
            CALL CD_GET_VAR_ATT_ID (dset, ivar,
     .        '_CoordinateAxisType', iatt, status)
            CALL CD_GET_VAR_ATT_INFO (dset, ivar, iatt, upbuff, attype, 
     .        attlen, attoutflag, status )
            IF (attype .NE. NCCHAR) got_it = .FALSE.
         ENDIF

         IF (got_it) THEN
	    i = STR_UPCASE(upbuff, buff)
            IF (upbuff.EQ.'TIME') THEN 
               dir_code = 'TI'  ! time
            ELSE IF (upbuff.EQ.'LON' .OR. upbuff.EQ.'GEOX') THEN
            dir_code = 'WE'
            ELSE IF (upbuff.EQ.'LAT' .OR. upbuff.EQ.'GEOY') THEN
             dir_code = 'SN'
            ELSE IF (upbuff.EQ.'HEIGHT') THEN
               dir_code = 'UD'
            ELSE IF (upbuff.EQ.'PRESSURE') THEN
               dir_code = 'DU'
            ELSE IF (upbuff.EQ.'ENSEMBLE') THEN
               dir_code = 'EE'
            ELSE IF (upbuff.EQ.'RUNTIME') THEN
               dir_code = 'FF'
            ENDIF
	 
	    IF (upbuff.EQ.'HEIGHT' .OR. 
     .               upbuff.EQ.'PRESSURE') THEN
               maxlen = 20
               got_it = NC_GET_ATTRIB(dset, ivar, 
     .              'CoordinateZisPositive',
     .              do_warn, vname(:vlen), maxlen, attlen, 
     .              attoutflag, buff, val)
               IF (got_it) THEN
	          i = STR_UPCASE(upbuff, buff)
	          IF (upbuff.EQ.'UP')
     .             dir_code = 'DU'
	          IF (upbuff.EQ.'DOWN')
     .             dir_code = 'UD'
               ENDIF
            ENDIF
         ENDIF

      ENDIF

* ... next check units
               
* If this is called by cd_get_1_axis, we have incoming units from 
* line_units(iaxis) already. Otherwise, get basic units.

* Time ax may contain "tunits since T0_date"

      since_T0 = MAX( INDEX(linunits,'since'), INDEX(buff,'SINCE') )
      IF (since_t0 .GT. 2) dir_code = 'TI'

      IF (linunits .EQ. char_init16) THEN

         linunits = ' '   ! default
         maxlen = bufflen
         got_it = NC_GET_ATTRIB( dset, ivar, 'units', do_warn,
     .                        vname(:vlen), maxlen, attlen, attoutflag, 
     .                        buff, val)

         since_T0 = MAX( INDEX(buff,'since'), INDEX(buff,'SINCE') )
         yyyymmdd_time = buff(1:15) .EQ. 'yyyymmddhhmmss'

         IF ( since_T0 .GT. 2 ) THEN
            linunits = buff(:since_T0-1)  ! just the "tunits" part 
         ELSEIF ( yyyymmdd_time ) THEN
            linunits = 'days'
	    dir_code = 'TI'
         ELSE
            IF (got_it) linunits = buff
         ENDIF

* EPIC time axes are encoded as 2 integers in variables TIME and TIME2
         IF ( vupcase .EQ. 'TIME' ) THEN
            CALL CD_GET_VAR_ID ( dset, 'time2', epic_t2var, status)
	    IF ( status .EQ. merr_ok ) linunits = 'Days'     ! arbitrary
         ENDIF
      ENDIF
      
* Skip this for DSG files.  Looking not just at axes but other vars
* which may be e.g. degrees heading.

      IF (full) THEN

* ... decode the units

      units = TM_UNIT_ID( linunits )
      istat = STR_UPCASE( vupcase, vname )


      IF (dir_code .EQ. 'NA' .OR. 
     .    dir_code .EQ. 'XX' .OR.
     .    dir_code .EQ. 'YY' .OR.
     .    dir_code .EQ. 'FF') THEN

      IF ( units .LT. 0 ) THEN
	 IF (dir_code .EQ. 'FF') THEN
	    dir_code = 'FI'  ! forecast time
	 ELSE
            dir_code = 'TI'  ! time
	 ENDIF

      ELSEIF ( units .EQ. 4 ) THEN     ! degrees
        IF  ( TM_HAS_STRING(linunits, 'NORTH')
     .   .OR. TM_HAS_STRING(linunits, 'SOUTH')
     .   .OR. TM_HAS_STRING(linunits, 'degreeN')   ! Check all CF_compliant 
     .   .OR. TM_HAS_STRING(linunits, 'degreesN')  ! options
     .   .OR. TM_HAS_STRING(linunits, 'degree_N')  ! Case independent
     .   .OR. TM_HAS_STRING(linunits, 'degrees_N')
c     .   .OR. TM_HAS_STRING(linunits, 'degreesS') ! But this matches all degrees
     .   .OR. TM_HAS_STRING(linunits, 'degree_S')
     .   .OR. TM_HAS_STRING(linunits, 'degrees_S')
     .   .OR. TM_HAS_STRING(linunits, 'LAT') ) THEN 
            dir_code = 'SN'
        ELSEIF ( TM_HAS_STRING(linunits, 'EAST')
     .   .OR. TM_HAS_STRING(linunits, 'WEST')
     .   .OR. TM_HAS_STRING(linunits, 'degreeE')
     .   .OR. TM_HAS_STRING(linunits, 'degreesE')
     .   .OR. TM_HAS_STRING(linunits, 'degree_E')
     .   .OR. TM_HAS_STRING(linunits, 'degrees_E')
     .   .OR. TM_HAS_STRING(linunits, 'degreeW')
     .   .OR. TM_HAS_STRING(linunits, 'degreesW')
     .   .OR. TM_HAS_STRING(linunits, 'degree_W')
     .   .OR. TM_HAS_STRING(linunits, 'degrees_W')
     .   .OR. TM_HAS_STRING(linunits, 'LON') ) THEN 
            dir_code = 'WE'
        ELSEIF ( dir_code .EQ. 'XX') THEN
                        dir_code = 'WE'
        ELSEIF ( dir_code .EQ. 'YY') THEN
                        dir_code = 'SN'
        ENDIF

      ELSEIF ( units .EQ. pun_millibars
     .   .OR.  units .EQ. pun_decibars ) THEN 
         dir_code = 'UD'
      ELSEIF ( units .EQ. 14
     .   .OR.  units .EQ. 15   ) THEN         ! layer or level
         dir_code = 'UD'
* ... then check stray clues
         ELSEIF ( TM_HAS_STRING(linunits, 'LAT') ) THEN
         dir_code = 'SN'
      ELSEIF ( TM_HAS_STRING(linunits, 'LON') ) THEN
         dir_code = 'WE'
      ENDIF
      ENDIF

      IF (units .EQ. 4) THEN    ! degrees
         IF (dir_code .EQ. 'NA' ) THEN
            IF ( vupcase(1:1) .EQ. 'Y'
     .      .OR. INDEX(vupcase, 'LAT') .GT. 0 ) THEN
               dir_code = 'SN'
            ELSEIF ( vupcase(1:1) .EQ. 'X'
     .      .OR. INDEX(vupcase, 'LON') .GT. 0 ) THEN
               dir_code = 'WE'
! 8/97: "Y" anywhere in name as a last ditch attempt to id it
            ELSEIF ( INDEX(vupcase, 'Y')   .GT. 0 ) THEN
               dir_code = 'SN'
            ELSE
               dir_code = 'WE'
            ENDIF
         ENDIF
	 IF (dir_code .EQ. 'WE') THEN
	    linunits = 'degrees_east'
	    slen = TM_LENSTR1(linunits)
	    dummy = 1.
	    CALL CD_REPLACE_ATTR (dset, ivar, 'units', NF_CHAR, slen,
     .            linunits, dummy, status)
	 ENDIF
	 IF (dir_code .EQ. 'SN') THEN
	    linunits = 'degrees_north'
	    slen = TM_LENSTR1(linunits)
	    dummy = 1.
	    CALL CD_REPLACE_ATTR (dset, ivar, 'units', NF_CHAR, slen,
     .            linunits, dummy, status)
	 ENDIF
      ENDIF

      ENDIF  ! full

! See if the long-name contains a string that indicates a z axis
      maxlen = bufflen
      got_it = NC_GET_ATTRIB( dset,ivar,'long_name',
     .           do_warn, vname(:vlen), maxlen, attlen, attoutflag, 
     .           buff, val)
      IF ( got_it .AND. dir_code .EQ. 'NA' ) THEN
         IF (STR_CASE_BLIND_COMPARE('elev',buff(:4)) .EQ. str_eq) 
     .           dir_code = 'DU'
         IF (STR_CASE_BLIND_COMPARE('height',buff(:6)) .EQ. str_eq) 
     .           dir_code = 'DU'
         IF (STR_CASE_BLIND_COMPARE('level',buff(:5)) .EQ. str_eq) 
     .           dir_code = 'DU'
         IF (STR_CASE_BLIND_COMPARE('layer',buff(:5)) .EQ. str_eq) 
     .           dir_code = 'DU'
         IF (STR_CASE_BLIND_COMPARE('depth',buff(:5)) .EQ. str_eq) 
     .           dir_code = 'UD'
         IF (STR_CASE_BLIND_COMPARE('latitude',buff(:8)) .EQ. str_eq) 
     .           dir_code = 'SN'
         IF (STR_CASE_BLIND_COMPARE('longitude',buff(:9)) .EQ. str_eq) 
     .           dir_code = 'WE'
      ENDIF

* the attribute "positive=down", etc. may help
      maxlen = bufflen
      got_it = NC_GET_ATTRIB( dset, ivar, 'positive', do_warn, 
     .                        vname(:vlen), maxlen, attlen, attoutflag, 
     .                        buff, val)
      IF ( got_it ) THEN
         IF (STR_CASE_BLIND_COMPARE('west',buff(:4))
     .                                     .EQ.str_eq) THEN
            dir_code = 'WE'
         ELSEIF( STR_CASE_BLIND_COMPARE('east',buff(:4))
     .                                     .EQ.str_eq) THEN
            dir_code = 'WE'
         ELSEIF( STR_CASE_BLIND_COMPARE('up',buff(:2))
     .                                     .EQ.str_eq) THEN
            dir_code = 'DU'
         ELSEIF( STR_CASE_BLIND_COMPARE('down',buff(:4))
     .                                     .EQ.str_eq) THEN
            dir_code = 'UD'
         ENDIF
      ENDIF


* Basic standard names
* Look for time, longitude, latitude, depth and height names.
* Forecast time axes have a standard-name attribute "forecast_reference_time"
* Ensemble axis may have a standard-name attribute "realization"

      maxlen = bufflen
      got_it = NC_GET_ATTRIB( dset, ivar, 'standard_name', do_warn, 
     .                        vname(:vlen), maxlen, attlen, attoutflag, 
     .                        buff, val)
      IF ( got_it ) THEN
         IF (STR_CASE_BLIND_COMPARE(
     .      'time',buff(:4))
     .                                     .EQ.str_eq) THEN
            dir_code = 'TI'
         ENDIF 
         IF (STR_CASE_BLIND_COMPARE(
     .      'longitude',buff(:9))          .EQ.str_eq .OR. 
     .      STR_CASE_BLIND_COMPARE(
     .      'grid_longitude',buff(:19))    .EQ.str_eq) THEN
            dir_code = 'WE'
         ENDIF 
         IF (STR_CASE_BLIND_COMPARE(
     .      'latitude',buff(:8))           .EQ.str_eq .OR.
     .      STR_CASE_BLIND_COMPARE(
     .      'grid_latitude',buff(:18))     .EQ.str_eq) THEN
            dir_code = 'SN'
         ENDIF 
	 
         IF (STR_CASE_BLIND_COMPARE(
     .      'depth',buff(:5))
     .                                     .EQ.str_eq) THEN
            dir_code = 'UD'
         ENDIF 
	 
         IF (STR_CASE_BLIND_COMPARE(
     .      'altitude',buff(:8))
     .                                     .EQ.str_eq) THEN
            dir_code = 'DU'
         ENDIF 
         IF (STR_CASE_BLIND_COMPARE(
     .      'forecast_reference_time',buff(:24))
     .                                     .EQ.str_eq) THEN
            dir_code = 'FI'
         ENDIF 
         IF (STR_CASE_BLIND_COMPARE(
     .      'realization',buff(:24))
     .                                     .EQ.str_eq) THEN
            dir_code = 'EE'
         ENDIF 
      ENDIF


* Standard_names indicating a sigma-axis in Z, and standard names 
* for geophysical axes, longitude, latitude, altitude etc. Note 
* that there seem to be some standard names related to sigma
* coordinates that may not indicate a vertical axis, so simply 
* matching sigma and coordinate is probably not safe.

      maxlen = bufflen
      got_it = NC_GET_ATTRIB( dset, ivar, 'standard_name', do_warn, 
     .                        vname(:vlen), maxlen, attlen, attoutflag, 
     .                        buff, val)
      IF ( got_it ) THEN
         IF (STR_CASE_BLIND_COMPARE(
     .      'atmosphere_hybrid_height_coordinate',buff(:35))
     .                                     .EQ.str_eq) THEN
            dir_code = 'DU'
         ENDIF 
         IF (STR_CASE_BLIND_COMPARE(
     .      'atmosphere_hybrid_sigma_pressure_coordinate',buff(:43))
     .                                     .EQ.str_eq) THEN
            dir_code = 'DU'
         ENDIF
         IF (STR_CASE_BLIND_COMPARE(
     .      'atmosphere_hybrid_height_coordinate',buff(:35))
     .                                     .EQ.str_eq) THEN
            dir_code = 'DU'
         ENDIF
         IF (STR_CASE_BLIND_COMPARE(
     .      'atmosphere_ln_pressure_coordinate',buff(:33))
     .                                     .EQ.str_eq) THEN
            dir_code = 'DU'
         ENDIF
         IF (STR_CASE_BLIND_COMPARE(
     .      'atmosphere_sigma_coordinate',buff(:27))
     .                                     .EQ.str_eq) THEN
            dir_code = 'DU'
         ENDIF
         IF (STR_CASE_BLIND_COMPARE(
     .      'atmosphere_sleve_coordinate',buff(:27))
     .                                     .EQ.str_eq) THEN
            dir_code = 'DU'
         ENDIF
         IF (STR_CASE_BLIND_COMPARE(
     .      'land_ice_sigma_coordinate',buff(:25))
     .                                     .EQ.str_eq) THEN
            dir_code = 'DU'
         ENDIF
         IF (STR_CASE_BLIND_COMPARE(
     .      'ocean_s_coordinate',buff(:19))
     .                                     .EQ.str_eq) THEN
            dir_code = 'UD'
         ENDIF
         IF (STR_CASE_BLIND_COMPARE(
     .      'ocean_sigma_coordinate',buff(:22))
     .                                     .EQ.str_eq) THEN
            dir_code = 'UD'
         ENDIF
         IF (STR_CASE_BLIND_COMPARE(
     .      'ocean_sigma_z_coordinate',buff(:24))
     .                                     .EQ.str_eq) THEN
            dir_code = 'UD'
         ENDIF
         IF (STR_CASE_BLIND_COMPARE(
     .      'ocean_double_sigma_coordinate',buff(:29))
     .                                     .EQ.str_eq) THEN
            dir_code = 'UD'
         ENDIF

      ENDIF  ! end of std names

* Still punting on axis names. If it's single-letter X, Y, Z, T,
* or starts with XAX, YAX, etc. then give it that direction.

      IF (dir_code .EQ. 'NA' .AND. vlen .EQ. 1) THEN
         IF ( vupcase(1:1) .EQ. 'X') dir_code = 'XX'
         IF ( vupcase(1:1) .EQ. 'Y') dir_code = 'YY'
         IF ( vupcase(1:1) .EQ. 'Z') dir_code = 'DU'
         IF ( vupcase(1:1) .EQ. 'T') dir_code = 'TI'
      ENDIF
      IF (dir_code.EQ.'NA') THEN
         IF (vupcase(1:3).EQ.'XAX') dir_code = 'XX'
         IF (vupcase(1:3).EQ.'YAX') dir_code = 'YY'
         IF (vupcase(1:3).EQ.'ZAX') dir_code = 'DU'
         IF (vupcase(1:3).EQ.'TAX') dir_code = 'TI'
      ENDIF

* If the name indicates a direction in obvious ways, use it.  But if there
* was a units string that did NOT dictate a direction, leave undefined.

      IF (full .AND. dir_code.EQ.'NA' .AND. units.EQ.no_units) THEN
         IF (vupcase(1:3).EQ.'LON') dir_code = 'WE'
         IF (vupcase(1:3).EQ.'LAT') dir_code = 'SN'
         IF (vupcase(1:4).EQ.'TIME') dir_code = 'TI'
         IF (vupcase(1:4).EQ.'DATE') dir_code = 'TI'
         IF (vupcase(1:3).EQ.'DEP') dir_code = 'UD'
         IF (vupcase(1:3).EQ.'ALT') dir_code = 'DU'
         IF (vupcase(1:4).EQ.'ELEV') dir_code = 'DU'
         IF (vupcase(1:3).EQ.'HEI') dir_code = 'DU'
         IF (vupcase(1:5).EQ.'LAYER') dir_code = 'DU'
         IF (vupcase(1:5).EQ.'LEVEL') dir_code = 'DU'
      ENDIF

 5000 RETURN
      END
