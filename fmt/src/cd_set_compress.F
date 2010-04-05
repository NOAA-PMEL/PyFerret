      SUBROUTINE CD_SET_COMPRESS (cdfid, varid, vname, dimid, deflate,
     .                            chunks, shuffle, endian_code, status)

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
* Set the netCDF compression for a variable

* Programmer Ansley Manke
* NOAA/PMEL, Seattle, WA - Thermal Modeling and Analysis Program
* Netcdf-4 with HDF chunking and compression
* revision 0.0 - 12/09

* argument definitions:
*   cdfid   - returned pointer to open CDF file
*   varid   - variable id
*   dimid   - id of dimensions
*   deflate - deflate level (level 1 is recommended)
*   chunks  - chunking corrresponding to dimid (if all 0 then no set chunking)
*   shuffle - 0 or 1 to turn on shuffle in NF_DEF_VAR_DEFLATE
*   endian_code - native=0, little=1, big=0 Values match nf_endian_*
*   status  - return status

* argument declarations:
	INTEGER cdfid, varid, dimid(*), deflate, chunks(*), 
     .          shuffle,  status
	CHARACTER*(*) vname

* include files
        include 'netcdf.decl'
#include "netcdf.inc"
        include 'cdf_tmap.parm'
        include 'tmap_errors.parm'

* Declare netcdf function names. 
* The ones without trailing underscore are declared in netcdf.inc
#include "netcdf_declareUnderscore.inc"


* local variable declarations:

      INTEGER cdfstat, idim, vlen
      INTEGER do_deflate, do_chunk, cdfformat

      CHARACTER*1     ww_dim_nam(4)
      DATA ww_dim_nam/'X', 'Y', 'Z', 'T'/

* Called from CD_MAKE_VAR. We are in define mode

      IF (deflate .GT. 0) do_deflate = 1

* Dataset needs to have been opened with netcdf4 mode, e.g.
* see cd_open_out NF_CREATE(cdfname, NF_NETCDF4, cdfid)
* Check this via a call to NF_INQ_FORMAT

#ifdef usingDODSf2cUnderscore 
           cdfstat = NF_INQ_FORMAT_(cdfid, cdfformat)
#else
           cdfstat = NF_INQ_FORMAT (cdfid, cdfformat) 
#endif

      IF (cdfformat .NE. NF_FORMAT_NETCDF4) GOTO 5200

* Has the user set chunking values?

      do_chunk = 0
      DO 100 idim = 1, 4
         do_chunk = MAX(do_chunk, chunks(idim) ) 
  100 CONTINUE

      IF (do_chunk .GT. 0) THEN
         cdfstat = NF_DEF_VAR_CHUNKING(cdfid, varid, NF_CHUNKED, chunks)
         IF (cdfstat .NE. NF_NOERR) GOTO 5100
      ENDIF

* Set output endian-ness. (default is whatever is native to
* writing machine, NF_ENDIAN_NATIVE).
      cdfstat = NF_DEF_VAR_ENDIAN(cdfid, varid, endian_code)
      IF (cdfstat .NE. NF_NOERR) GOTO 5100

      cdfstat = NF_DEF_VAR_DEFLATE(cdfid, varid, shuffle, 
     .          deflate, do_deflate)
      IF (cdfstat .NE. NF_NOERR) GOTO 5100

* Turn on fletcher32 checksum for the var
      cdfstat = NF_DEF_VAR_FLETCHER32(cdfid, varid, NF_FLETCHER32)
      IF (cdfstat .NE. NF_NOERR) GOTO 5100


* successful completion
      status = merr_ok
      RETURN

* error messages
 5100 CONTINUE
      CALL TM_ERRMSG
     .     ( status+pcdferr, status, 'CD_SET_COMPRESS', cdfid,
     .     no_varid, 'could not set compression', ' ', *5000 )

 5200 CONTINUE
      CALL TM_ERRMSG
     .     ( merr_syntax, status, 'CD_SET_COMPRESS', cdfid,no_varid, 
     .     'Cannot compress variable: dataset not netcdf4 format', 
     .     ' ', *5000 )

* error exit
 5000 RETURN
	END
