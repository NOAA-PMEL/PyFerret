      SUBROUTINE CD_GET_LINE_DIRECTION (dset, ivar, iaxis, 
     .     vname, do_warn)

*      include files: 
      
        IMPLICIT NONE
#include "netcdf.inc"
        include 'tmap_errors.parm'
        include 'tmap_dims.parm'
#include "tmap_dset.parm"
        include 'xtm_grid.cmn_text'
        external xgt_grid_data
        include 'xunits.cmn_text'
        external xunits_data
        include 'errmsg.parm'


* calling arguments
      LOGICAL do_warn
      INTEGER dset, ivar, iaxis, idir
      CHARACTER*(*) vname

* local declarations
      LOGICAL TM_HAS_STRING, NC_GET_ATTRIB, got_it
      CHARACTER*1 axis_dir*1, vupcase*128

      INTEGER STR_UPCASE, STR_CASE_BLIND_COMPARE, TM_LENSTR1, 
     .	      TM_UNIT_ID, units, attlen, attoutflag, attype, iatt,  
     .        vlen, slen,maxlen, istat, status

      REAL val, dummy

      INTEGER bufflen
      PARAMETER (bufflen = 256)
      CHARACTER*256 buff
      INTEGER     str_eq
      PARAMETER ( str_eq = 0 )


* Use attributes to try to determine the orientation and 
* direction of the axis

      line_direction(iaxis) = 'NA'

      vlen = TM_LENSTR1(vname)
      units = TM_UNIT_ID( line_units(iaxis) )
      istat = STR_UPCASE( vupcase, vname )


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
            line_direction(iaxis) = 'TI'  ! time
          ELSE IF (axis_dir.EQ.'X' .OR. axis_dir.EQ.'I' .OR.
     .             axis_dir.eq.'x' .or. axis_dir.eq.'i') THEN
            line_direction(iaxis) = 'XX'
         ELSE IF (axis_dir.EQ.'Y' .OR. axis_dir.EQ.'J' .OR.
     .            axis_dir.eq.'y' .or. axis_dir.eq.'j') THEN
            line_direction(iaxis) = 'YY'
         ELSE IF (axis_dir.EQ.'Z' .OR. axis_dir.EQ.'K' .OR.
     .            axis_dir.eq.'z' .or. axis_dir.eq.'k') THEN
            line_direction(iaxis) = 'DU'
         ELSE IF (axis_dir.EQ.'E' .OR. axis_dir.EQ.'M' .OR.
     .            axis_dir.eq.'e' .or. axis_dir.eq.'m') THEN
            line_direction(iaxis) = 'EE'
         ELSE IF (axis_dir.EQ.'F' .OR. axis_dir.EQ.'N' .OR.
     .            axis_dir.eq.'f' .or. axis_dir.eq.'n') THEN
            line_direction(iaxis) = 'FF'
         ENDIF
      ELSE
         axis_dir = char_init01
      ENDIF


* ... check for a CARTESIAN_AXIS attribute, if still undetermined

      IF (line_direction(iaxis) .EQ. 'NA') THEN

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
               line_direction(iaxis) = 'TI'  ! time
            ELSE IF (axis_dir.EQ.'X' .OR. axis_dir.EQ.'I' .OR.
     .               axis_dir.eq.'x' .or. axis_dir.eq.'i') THEN
            line_direction(iaxis) = 'WE'
            ELSE IF (axis_dir.EQ.'Y' .OR. axis_dir.EQ.'J' .OR.
     .               axis_dir.eq.'y' .or. axis_dir.eq.'j') THEN
             line_direction(iaxis) = 'SN'
            ELSE IF (axis_dir.EQ.'Z' .OR. axis_dir.EQ.'K' .OR.
     .               axis_dir.eq.'z' .or. axis_dir.eq.'k') THEN
               line_direction(iaxis) = 'DU'
            ELSE IF (axis_dir.EQ.'E' .OR. axis_dir.EQ.'M' .OR.
     .            axis_dir.eq.'e' .or. axis_dir.eq.'m') THEN
               line_direction(iaxis) = 'EE'
            ELSE IF (axis_dir.EQ.'F' .OR. axis_dir.EQ.'N' .OR.
     .            axis_dir.eq.'f' .or. axis_dir.eq.'n') THEN
               line_direction(iaxis) = 'FF'
            ENDIF
         ELSE
            axis_dir = char_init01
         ENDIF
      ENDIF


* ... next check units
               

      IF (line_direction(iaxis) .EQ. 'FF' .AND. 
     .     units .LT. 0 ) line_direction(iaxis) = 'FI'  ! forecast time

      IF (line_direction(iaxis) .EQ. 'NA' .OR. 
     .    line_direction(iaxis) .EQ. 'XX' .OR.
     .    line_direction(iaxis) .EQ. 'YY') THEN

      IF ( units .LT. 0 ) THEN
         line_direction(iaxis) = 'TI'  ! time
      ELSEIF ( units .EQ. 4 ) THEN     ! degrees
        IF  ( TM_HAS_STRING(line_units(iaxis), 'NORTH')
     .   .OR. TM_HAS_STRING(line_units(iaxis), 'SOUTH')
     .   .OR. TM_HAS_STRING(line_units(iaxis), 'degreeN')   ! Check all CF_compliant 
     .   .OR. TM_HAS_STRING(line_units(iaxis), 'degreesN')  ! options
     .   .OR. TM_HAS_STRING(line_units(iaxis), 'degree_N')  ! Case independent
     .   .OR. TM_HAS_STRING(line_units(iaxis), 'degrees_N')
c     .   .OR. TM_HAS_STRING(line_units(iaxis), 'degreesS') ! But this matches all degrees
     .   .OR. TM_HAS_STRING(line_units(iaxis), 'degree_S')
     .   .OR. TM_HAS_STRING(line_units(iaxis), 'degrees_S')
     .   .OR. TM_HAS_STRING(line_units(iaxis), 'LAT') ) THEN 
            line_direction(iaxis) = 'SN'
        ELSEIF ( TM_HAS_STRING(line_units(iaxis), 'EAST')
     .   .OR. TM_HAS_STRING(line_units(iaxis), 'WEST')
     .   .OR. TM_HAS_STRING(line_units(iaxis), 'degreeE')
     .   .OR. TM_HAS_STRING(line_units(iaxis), 'degreesE')
     .   .OR. TM_HAS_STRING(line_units(iaxis), 'degree_E')
     .   .OR. TM_HAS_STRING(line_units(iaxis), 'degrees_E')
     .   .OR. TM_HAS_STRING(line_units(iaxis), 'degreeW')
     .   .OR. TM_HAS_STRING(line_units(iaxis), 'degreesW')
     .   .OR. TM_HAS_STRING(line_units(iaxis), 'degree_W')
     .   .OR. TM_HAS_STRING(line_units(iaxis), 'degrees_W')
     .   .OR. TM_HAS_STRING(line_units(iaxis), 'LON') ) THEN 
            line_direction(iaxis) = 'WE'
        ELSEIF ( line_direction(iaxis) .EQ. 'XX') THEN
                        line_direction(iaxis) = 'WE'
        ELSEIF ( line_direction(iaxis) .EQ. 'YY') THEN
                        line_direction(iaxis) = 'SN'
        ENDIF

      ELSEIF ( units .EQ. pun_millibars
     .   .OR.  units .EQ. pun_decibars ) THEN 
         line_direction(iaxis) = 'UD'
      ELSEIF ( units .EQ. 14
     .   .OR.  units .EQ. 15   ) THEN         ! layer or level
         line_direction(iaxis) = 'UD'
* ... then check stray clues
         ELSEIF ( TM_HAS_STRING(line_units(iaxis), 'LAT') ) THEN
         line_direction(iaxis) = 'SN'
      ELSEIF ( TM_HAS_STRING(line_units(iaxis), 'LON') ) THEN
         line_direction(iaxis) = 'WE'
      ENDIF
      ENDIF

      IF (units .EQ. 4) THEN    ! degrees
         IF (line_direction(iaxis) .EQ. 'NA' ) THEN
            IF ( vupcase(1:1) .EQ. 'Y'
     .      .OR. INDEX(vupcase, 'LAT') .GT. 0 ) THEN
               line_direction(iaxis) = 'SN'
            ELSEIF ( vupcase(1:1) .EQ. 'X'
     .      .OR. INDEX(vupcase, 'LON') .GT. 0 ) THEN
               line_direction(iaxis) = 'WE'
! 8/97: "Y" anywhere in name as a last ditch attempt to id it
            ELSEIF ( INDEX(vupcase, 'Y')   .GT. 0 ) THEN
               line_direction(iaxis) = 'SN'
            ELSE
               line_direction(iaxis) = 'WE'
            ENDIF
         ENDIF
	 IF (line_direction(iaxis) .EQ. 'WE') THEN
	    line_units(iaxis) = 'degrees_east'
	    slen = TM_LENSTR1(line_units(iaxis))
	    dummy = 1.
	    CALL CD_REPLACE_ATTR (dset, ivar, 'units', NF_CHAR, slen,
     .            line_units(iaxis), dummy, status)
	 ENDIF
	 IF (line_direction(iaxis) .EQ. 'SN') THEN
	    line_units(iaxis) = 'degrees_north'
	    slen = TM_LENSTR1(line_units(iaxis))
	    dummy = 1.
	    CALL CD_REPLACE_ATTR (dset, ivar, 'units', NF_CHAR, slen,
     .            line_units(iaxis), dummy, status)
	 ENDIF
      ENDIF

! See if the long-name contains a string that indicates a z axis
      maxlen = bufflen
      got_it = NC_GET_ATTRIB( dset,ivar,'long_name',
     .           do_warn, vname(:vlen), maxlen, attlen, attoutflag, 
     .           buff, val)
      IF ( got_it .AND. line_direction(iaxis) .EQ. 'NA' ) THEN
         IF (STR_CASE_BLIND_COMPARE('elev',buff(:4)) .EQ. str_eq) 
     .           line_direction(iaxis) = 'DU'
         IF (STR_CASE_BLIND_COMPARE('height',buff(:6)) .EQ. str_eq) 
     .           line_direction(iaxis) = 'DU'
         IF (STR_CASE_BLIND_COMPARE('level',buff(:5)) .EQ. str_eq) 
     .           line_direction(iaxis) = 'DU'
         IF (STR_CASE_BLIND_COMPARE('layer',buff(:5)) .EQ. str_eq) 
     .           line_direction(iaxis) = 'DU'
         IF (STR_CASE_BLIND_COMPARE('depth',buff(:5)) .EQ. str_eq) 
     .           line_direction(iaxis) = 'UD'
      ENDIF

* the attribute "positive=down", etc. may help
      maxlen = bufflen
      got_it = NC_GET_ATTRIB( dset, ivar, 'positive', do_warn, 
     .                        vname(:vlen), maxlen, attlen, attoutflag, 
     .                        buff, val)
      IF ( got_it ) THEN
         IF (STR_CASE_BLIND_COMPARE('west',buff(:4))
     .                                     .EQ.str_eq) THEN
            line_direction(iaxis) = 'WE'
         ELSEIF( STR_CASE_BLIND_COMPARE('east',buff(:4))
     .                                     .EQ.str_eq) THEN
            line_direction(iaxis) = 'WE'
         ELSEIF( STR_CASE_BLIND_COMPARE('up',buff(:2))
     .                                     .EQ.str_eq) THEN
            line_direction(iaxis) = 'DU'
         ELSEIF( STR_CASE_BLIND_COMPARE('down',buff(:4))
     .                                     .EQ.str_eq) THEN
            line_direction(iaxis) = 'UD'
         ENDIF
      ENDIF

* standard_names indicating a sigma-axis in Z, and standard names 
* for geophysical axes, longitude, latitude, altitude etc. Note 
* that there seem to be some standard names related to sigma
* coordinates that may not indicate a vertical axis, so simply 
* matching sigma and coordinate is probably not safe.
* Forecast time axes have a standard-name attribute "forecast_reference_time"

      maxlen = bufflen
      got_it = NC_GET_ATTRIB( dset, ivar, 'standard_name', do_warn, 
     .                        vname(:vlen), maxlen, attlen, attoutflag, 
     .                        buff, val)
      IF ( got_it ) THEN
         IF (STR_CASE_BLIND_COMPARE(
     .      'forecast_reference_time',buff(:24))
     .                                     .EQ.str_eq) THEN
            line_direction(iaxis) = 'FI'
         ENDIF 
         IF (STR_CASE_BLIND_COMPARE(
     .      'atmosphere_hybrid_height_coordinate',buff(:35))
     .                                     .EQ.str_eq) THEN
            line_direction(iaxis) = 'DU'
         ENDIF 
         IF (STR_CASE_BLIND_COMPARE(
     .      'atmosphere_hybrid_sigma_pressure_coordinate',buff(:43))
     .                                     .EQ.str_eq) THEN
            line_direction(iaxis) = 'DU'
         ENDIF
         IF (STR_CASE_BLIND_COMPARE(
     .      'atmosphere_hybrid_height_coordinate',buff(:35))
     .                                     .EQ.str_eq) THEN
            line_direction(iaxis) = 'DU'
         ENDIF
         IF (STR_CASE_BLIND_COMPARE(
     .      'atmosphere_ln_pressure_coordinate',buff(:33))
     .                                     .EQ.str_eq) THEN
            line_direction(iaxis) = 'DU'
         ENDIF
         IF (STR_CASE_BLIND_COMPARE(
     .      'atmosphere_sigma_coordinate',buff(:27))
     .                                     .EQ.str_eq) THEN
            line_direction(iaxis) = 'DU'
         ENDIF
         IF (STR_CASE_BLIND_COMPARE(
     .      'atmosphere_sleve_coordinate',buff(:27))
     .                                     .EQ.str_eq) THEN
            line_direction(iaxis) = 'DU'
         ENDIF
         IF (STR_CASE_BLIND_COMPARE(
     .      'land_ice_sigma_coordinate',buff(:25))
     .                                     .EQ.str_eq) THEN
            line_direction(iaxis) = 'DU'
         ENDIF
         IF (STR_CASE_BLIND_COMPARE(
     .      'ocean_s_coordinate',buff(:19))
     .                                     .EQ.str_eq) THEN
            line_direction(iaxis) = 'UD'
         ENDIF
         IF (STR_CASE_BLIND_COMPARE(
     .      'ocean_sigma_coordinate',buff(:22))
     .                                     .EQ.str_eq) THEN
            line_direction(iaxis) = 'UD'
         ENDIF
         IF (STR_CASE_BLIND_COMPARE(
     .      'ocean_sigma_z_coordinate',buff(:24))
     .                                     .EQ.str_eq) THEN
            line_direction(iaxis) = 'UD'
         ENDIF
         IF (STR_CASE_BLIND_COMPARE(
     .      'ocean_double_sigma_coordinate',buff(:29))
     .                                     .EQ.str_eq) THEN
            line_direction(iaxis) = 'UD'
         ENDIF

      ENDIF  ! end of std names

      RETURN
      END
