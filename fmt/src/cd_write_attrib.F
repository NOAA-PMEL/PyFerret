	SUBROUTINE CD_WRITE_ATTRIB(cdfid,vname,att,string,append,status)

* Write a character string attribute to a netCDF file
* If the variable is equal to the parameter pcdf_global then make it a
* global attribute.

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 3/92
*               10/92: set into define mode

* argument definitions:
*     cdfid   - netCDF id number for already opened CDF file
*     vname   - name of previously created netCDF variable
*     att     - name of attribute
*     string  - what the attribute is set equal to
*     append  - append to or replace the existing string, if any
*     status  - return status

* include files
#ifdef unix
        include 'tmap_format/netcdf.decl'
        include 'tmap_format/netcdf.inc'
	include 'tmap_format/tmap_errors.parm'
	include 'tmap_format/tmap_dims.parm'
#include "tmap_format/tmap_dset.parm"
	include 'tmap_format/cdf_tmap.parm'
	include 'tmap_format/gt_lib.parm'
	include 'tmap_format/cd_lib.parm'
        include 'tmap_format/xio.cmn_text'
        external xio_data
        include 'tmap_format/xrisc_buff.cmn'
#else
xx        include 'xxtmap_format/netcdf.decl'  ! untested
xx        include 'xxtmap_format/netcdf.inc'
        INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:CDF_TMAP.PARM'
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
	INCLUDE 'TMAP_FORMAT:CD_LIB.PARM'
        INCLUDE 'TMAP_FORMAT:XIO.CMN_TEXT'
        EXTERNAL XIO_DATA
        INCLUDE 'TMAP_FORMAT:XRISC_BUFF.CMN'
#endif

* argument declarations
      LOGICAL append
      INTEGER cdfid, status
      CHARACTER*(*) vname, att, string

* local variable declarations:
      INTEGER TM_LENSTR1, NCVID,
     .        vlen, alen, slen, cdfstat, varid, attype, attlen
      CHARACTER buff*256

      INTEGER tt  ! nice short name
      EQUIVALENCE (tt,lunit_errors)

* initialize
      vlen = TM_LENSTR1( vname )
      alen = TM_LENSTR1( att )
      slen = LEN( string )
      
* locate the variable id
      IF ( vname .EQ. pcdf_global ) THEN
         varid = NCGLOBAL
      ELSE
         varid = NCVID( cdfid, vname(:vlen), cdfstat )
         IF ( cdfstat .NE. NCNOERR ) GOTO 5100
      ENDIF

* does the attribute already exist ?
      CALL NCAINQ( cdfid, varid, att(:alen), attype, attlen, cdfstat )
      IF ( cdfstat .EQ. NCNOERR ) THEN
         IF ( attype.NE.NCBYTE .AND. attype.NE.NCCHAR ) GOTO 5200
         IF ( attlen+slen .GT. 256 ) THEN
            buff = att
            CALL TM_NOTE( 'CDF attribute '//buff(:alen)//
     .           ' is too long', tt )
            IF ( attlen .GT. 250 ) THEN
               CALL TM_NOTE( 'no text could be appended', tt )
               GOTO 1000
            ELSE
               CALL TM_NOTE( 'it may be truncated', tt )
            ENDIF
         ENDIF
      ENDIF

* put CDF into define mode so attributes can be written
      CALL CD_SET_MODE(cdfid, pcd_mode_define, status)
      IF ( status .NE. merr_ok ) RETURN

      IF ( cdfstat.EQ.NCNOERR .AND. append ) THEN
* append to the attribute
         CALL NCAGTC( cdfid, varid, att(:alen), buff, 256, cdfstat )
         risc_buff = string
         CALL NCAPTC( cdfid, varid, att(:alen), NCCHAR, attlen+slen,
     .             buff(:attlen)//risc_buff, cdfstat )
      ELSE
* write the attribute
         CALL NCAPTC( cdfid, varid, att(:alen), NCCHAR, slen,
     .             string, cdfstat )
      ENDIF
      IF ( cdfstat .NE. NCNOERR ) GOTO 5300

* successful completion
 1000 status = merr_ok
      RETURN

 5100 CALL TM_ERRMSG
     . ( merr_unkvar, status, 'CD_WRITE_ATTRIB', unspecified_int4,
     .   unspecified_int4,
     .  'variable doesnt exist in CDF file',
     .  vname(:vlen), *5900 )

 5200 CALL TM_ERRMSG
     . ( merr_gridpredef, status, 'CD_WRITE_ATTRIB', unspecified_int4,
     .   unspecified_int4,
     .  'incompatible data type of CDF attribute',
     .  att(:alen), *5900 )

 5300 CALL TM_ERRMSG
     . ( cdfstat+pcdferr, status, 'CD_WRITE_ATTRIB', unspecified_int4,
     . no_varid, 'cannot write to CDF output file',
     . no_errstring, *5900 )

* error exit
 5900   RETURN
	END
