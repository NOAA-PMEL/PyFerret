	SUBROUTINE CD_WRITE_ATTVAL_DP ( cdfid, vname, att, val, nval,
     .                                  type, status )

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
* Write a double precision attribute to a netCDF file
* If the variable is equal to the parameter pcdf_global then make it a
* global attribute.

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 3/92
*               10/92: set into define mode
* 10/2005 NOTE: do not change this for the attribute-access changes. 
*               This routine accesses the newly-created file using 
*               netcdf calls as always. The values sent in come from
*               the linked-list structure but output is all the same.
* V600  4/06 *acm* - additional declarations found by Solaris compiler

* argument definitions:
*     cdfid   - netCDF id number for already opened CDF file
*     vname   - name of previously created netCDF variable
*     att     - name of attribute
*     val     - value(s) for the attribute
*     nval    - number of attribute values
*     type    - data type of the attribute
*     status  - return status

* include files
#ifdef unix
        include 'netcdf.decl'
#include "netcdf.inc"
	include 'tmap_errors.parm'
	include 'tmap_dims.parm'
#include "tmap_dset.parm"
        include 'cdf_tmap.parm'
#include "gt_lib.parm"
	include 'cd_lib.parm'
#else
xx        include 'netcdf.decl'  ! untested
#include "netcdf.inc"
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
        INCLUDE 'TMAP_FORMAT:CD_LIB.PARM'
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
	INCLUDE 'TMAP_FORMAT:CD_LIB.PARM'
#endif

* argument declarations
      INTEGER cdfid, nval, type, i4(10), i, status
      INTEGER*2 i2(10)
      BYTE i1
      CHARACTER*(*) vname, att
      REAL*8 val(*)

* local variable declarations:
      INTEGER TM_LENSTR1, NC_PUT_ATT_FLOAT,
     .        vlen, alen, cdfstat, varid, attype, attlen, flen    
      CHARACTER*128 abuff     ! needed because of fussy compilers and // concatenation

      PARAMETER (flen=512)      ! size of variable name string

#ifdef sun
      BYTE      fhol(flen)      ! c-type Hollerith string buffer
#else
      INTEGER*1 fhol(flen)      ! c-type Hollerith string buffer
#endif 

* For error message
      CHARACTER*9 typnam(6)
      DATA typnam/'NC_BYTE', 'NC_CHAR', 'NC_SHORT', 'NC_INT', 
     .            'NC_FLOAT', 'NC_DOUBLE'/

* initialize
      vlen = TM_LENSTR1( vname )
      alen = TM_LENSTR1( att )
      
* locate the variable id
      IF ( vname .EQ. pcdf_global ) THEN
         varid = NCGLOBAL
      ELSE
         varid = NCVID( cdfid, vname(:vlen), cdfstat )
         IF ( cdfstat .NE. NCNOERR ) GOTO 5100
      ENDIF

* does the attribute already exist ?
      CALL NCAINQ( cdfid, varid, att(:alen), attype, attlen, cdfstat )
      IF ( cdfstat.EQ.NCNOERR .AND. attype.NE.type ) GOTO 5200

* put CDF into define mode so attributes can be written
      CALL CD_SET_MODE(cdfid, pcd_mode_define, status)
      IF ( status .NE. merr_ok ) RETURN

* Convert to the requested type and write the attribute
* nc_put_att_float converts types if the pointer points to a different
* type than that specified by type

      CALL TM_FTOC_STRNG( att(1:alen), fhol, flen )

      status = merr_ok
      CALL CD_WRITE_ATT_DP_SUB(cdfid, varid, fhol, type, nval, 
     .  val, status)

      IF ( status .EQ. -60) GOTO 5300

c*      IF (type .EQ. NCLONG) THEN
c*         DO i = 1,nval
c*           i4(i) = NINT(val(i))
c*         ENDDO
c*         CALL NCAPT( cdfid, varid, att(:alen), type, nval,
c*     .            i4, cdfstat )
c*      ELSE IF (type .EQ. NCSHORT) THEN
c*         DO i = 1,nval
c*           i2(i) = NINT(val(i))
c*         ENDDO
c*         CALL NCAPT( cdfid, varid, att(:alen), type, nval,
c*     .            i2, cdfstat )
c*c      ELSE IF (type .EQ. NCBYTE) THEN
c*c         i1 = INT(val)
c*c         CALL NCAPT( cdfid, varid, att(:alen), type, nval,
c*c     .            i1, cdfstat )
c*      ELSE
c*         CALL NCAPT( cdfid, varid, att(:alen), type, nval,
c*     .           val, cdfstat )
c*      ENDIF

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

 5300  abuff = att
       CALL TM_ERRMSG (status+pcdferr, status, 'CD_WRITE_ATTVAL',
     .     no_descfile, no_stepfile, 
     .    'data in attribute '//abuff(:alen)//
     .    ' not representable in output type '//typnam(type),
     .     no_errstring, *5900 )

* error exit
 5900   RETURN
	END
