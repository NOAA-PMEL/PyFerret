      SUBROUTINE CD_GET_MISSING_FLAG (cdfid, varid, vname, do_warn, 
     .        flag, status)

* include files

        include 'netcdf.decl'
#include "netcdf.inc"
	include 'tmap_errors.parm'

* Argument declarations
      LOGICAL do_warn
      INTEGER cdfid, varid, status
      REAL*8 flag
      CHARACTER*(*) vname

* Local declarations
      LOGICAL CD_GET_ATTVAL, got_bad, got_missing, scaled, got_it, tf2
      INTEGER TM_LENSTR1, vartyp, nvdim, vdims(6), nvatts, vlen, i,
     .        attype
     
      REAL missing_flag, bad_flag, scale_factor, add_offset

      status = merr_ok

* get the vital statistics for this variable
      CALL NCVINQ(cdfid, varid, vname, vartyp, nvdim, vdims,
     .                 nvatts, status)
      vlen = TM_LENSTR1(vname)

      got_it = CD_GET_ATTVAL( cdfid, varid, 'scale_factor',
     .             do_warn,vname(:vlen), scale_factor, 1, status )
      tf2    = CD_GET_ATTVAL( cdfid, varid, 'add_offset',
     .             do_warn,vname(:vlen), add_offset, 1, status )
      scaled = got_it .OR. tf2

* netCDF files may use "standard" attributes missing_value and/or
* _FillValue to flag bad data
* For the NOAA COOP standard the missing_value flag must be stored in the
* packed data type.  If so it must be unpacked. 

      got_bad = CD_GET_ATTVAL( cdfid, varid, 'missing_value',
     .         do_warn, vname(:vlen), bad_flag, 1, status )
      got_missing = CD_GET_ATTVAL( cdfid, varid, '_FillValue',
     .         do_warn, vname(:vlen), missing_flag, 1, status)

* ... need to unpack the missing_value flag?
*     (note that the netCDF library insists on _FillValue's data type)

      IF ( vartyp.LE.NCLONG .AND. scaled ) THEN
         IF ( got_bad ) THEN
	        CALL NCAINQ(cdfid, varid, 'missing_value', attype,
     .			    i, status )
	        IF ( attype .EQ. vartyp ) bad_flag =
     .			bad_flag*scale_factor
     .			+ add_offset
	     ENDIF

* ... *kob* Also need to see about unpacking _FillValue

         IF ( got_missing ) THEN
	        CALL NCAINQ(cdfid, varid, '_FillValue', attype,
     .			    i, status )
	        IF ( attype .EQ. vartyp ) missing_flag =
     .			missing_flag*scale_factor
     .			+ add_offset
	     ENDIF
      ENDIF

*  Return missing flag if it exists, else bad_flag

      IF (got_missing) THEN
         flag =  missing_flag
      ELSE 
         IF ( got_bad ) THEN
            flag =  bad_flag
         ELSE
            flag = 0.
            status = 0
         ENDIF
      ENDIF

      RETURN
      END
