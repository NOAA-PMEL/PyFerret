	LOGICAL FUNCTION CD_ISIT_EPIC ( dset, cdfid, status )

* determine if this netCDF file is an EPIC file
* the following approach is taken to determine if this is an EPIC file:
* 1) locate one of the axes, "time" or "lat"
* 2) check for the attribute "epic_code"

* the common variable cd_what_isit is modified only if it is EPIC

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 2/93

* argument definitions:
*	cdfid	- netCDF id number for already opened CDF file

* argument declarations
      INTEGER dset, cdfid, status

#ifdef unix
      include 'tmap_format/netcdf.decl'
      include 'tmap_format/netcdf.inc'
      include 'tmap_format/tmap_errors.parm'
      include 'tmap_format/tmap_dims.parm'
      include 'tmap_format/xdset_info.cmn_text'
      external xdset_info_data
      include 'tmap_format/xdset_info.cd_equiv'
#else
      INCLUDE 'TMAP_FORMAT:NETCDF.DECL'
      INCLUDE 'TMAP_FORMAT:NETCDF.INC'
      INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
#endif

* internal variable declarations:
      INTEGER NCVID, cdfstat, i, axid, attype, attlen
      INTEGER   axnamlen(2)
      CHARACTER*4 axname(2)

* internal data statements
      DATA axname(1)/'time'/, axnamlen(1)/4/, 
     .     axname(2)/'lat' /, axnamlen(2)/3/

* initialize
      status = merr_ok

* locate an axis that's in the file
      DO 100 i = 1, 2
         axid = NCVID( cdfid, axname(i)(1:axnamlen(i)), cdfstat )
         IF ( axid .GT. 0 ) GOTO 200
 100  CONTINUE

* axis not found - it must not be EPIC
      CD_ISIT_EPIC = .FALSE.
      RETURN

* check for "epic_code" attribute
 200  CALL NCAINQ( cdfid, axid, 'epic_code', attype, attlen, cdfstat )
      CD_ISIT_EPIC =  cdfstat .EQ. NCNOERR

* successful completion (always)
      RETURN
      END
