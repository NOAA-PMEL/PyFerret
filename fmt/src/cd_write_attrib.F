	SUBROUTINE CD_WRITE_ATTRIB(cdfid,vname,att,string,append,status)

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
* Write a character string attribute to a netCDF file
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

* argument definitions:
*     cdfid   - netCDF id number for already opened CDF file
*     vname   - name of previously created netCDF variable
*     att     - name of attribute
*     string  - what the attribute is set equal to
*     append  - append to or replace the existing string, if any
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
        include 'xio.cmn_text'
        external xio_data
        include 'xrisc_buff.cmn'
#else
xx        include 'netcdf.decl'  ! untested
#include "netcdf.inc"
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
      INTEGER TM_LENSTR1, 
     .        vlen, alen, slen, cdfstat, varid, attype, attlen
      CHARACTER buff*1024

      INTEGER tt  ! nice short name
      EQUIVALENCE (tt,lunit_errors)

* initialize
      vlen = TM_LENSTR1( vname )
      alen = TM_LENSTR1( att )
      slen = LEN( string )
      attype = 0 
      attlen = 0 
      cdfstat = NCNOERR
      
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
         IF ( attlen+slen .GT. 1024 .AND. append ) THEN
            buff = att
            CALL TM_NOTE( 'CDF attribute '//buff(:alen)//
     .           ' is too long', tt )
            IF ( attlen .GT. 2046 ) THEN
               CALL TM_NOTE( 'no text could be appended', tt )
               GOTO 1000
            ELSE
               CALL TM_NOTE( 'it may be truncated', tt )
            ENDIF
         ENDIF
         IF ( attlen .GT. 1024 .AND. (.NOT.append) ) THEN
            buff = att
            CALL TM_NOTE( 'CDF attribute '//buff(:alen)//
     .           ' is too long', tt )
            CALL TM_NOTE( 'it may be truncated', tt )
         ENDIF
      ENDIF

* put CDF into define mode so attributes can be written
      CALL CD_SET_MODE(cdfid, pcd_mode_define, status)
      IF ( status .NE. merr_ok ) RETURN

      IF ( cdfstat.EQ.NCNOERR .AND. append ) THEN
* append to the attribute
         CALL NCAGTC( cdfid, varid, att(:alen), buff, 1024, cdfstat )
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

 5200 WRITE (risc_buff, *) 'incompatible data type of CDF attribute ',
     .                      att(:alen)
      alen = TM_LENSTR1(risc_buff)
      CALL TM_ERRMSG
     . ( merr_gridpredef, status, 'CD_WRITE_ATTRIB', unspecified_int4,
     .   unspecified_int4, risc_buff(:alen), 
     .   no_errstring, *5900 )

 5300 CALL TM_ERRMSG
     . ( cdfstat+pcdferr, status, 'CD_WRITE_ATTRIB', unspecified_int4,
     . no_varid, 'cannot write to CDF output file',
     . no_errstring, *5900 )

* error exit
 5900   RETURN
	END
