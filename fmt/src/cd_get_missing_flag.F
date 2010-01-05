      SUBROUTINE CD_GET_MISSING_FLAG (cdfid, varid, vname, do_warn, 
     .        flag, cdfstat)

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
* V65  *acm* 1/10  Change all netcdf2-style calls and parameter names to 
*                  netcdf3-style calls, which are also consistent with netcdf4.

* include files

        include 'netcdf.decl'
#include "netcdf.inc"
	include 'tmap_errors.parm'

* Argument declarations
      LOGICAL do_warn
      INTEGER cdfid, varid, cdfstat
      REAL*8 flag
      CHARACTER*(*) vname

* Local declarations
      LOGICAL CD_GET_ATTVAL, got_bad, got_missing, scaled, got_it, tf2
      INTEGER TM_LENSTR1, vartyp, nvdim, vdims(6), nvatts, vlen, i,
     .        attype
     
      REAL missing_flag, bad_flag, scale_factor, add_offset

      cdfstat = merr_ok

* get the vital statistics for this variable
      cdfstat = NF_INQ_VAR (cdfid, varid, vname, vartyp, nvdim, vdims,
     .                 nvatts)
      vlen = TM_LENSTR1(vname)

      got_it = CD_GET_ATTVAL( cdfid, varid, 'scale_factor',
     .             do_warn,vname(:vlen), scale_factor, 1, cdfstat )
      tf2    = CD_GET_ATTVAL( cdfid, varid, 'add_offset',
     .             do_warn,vname(:vlen), add_offset, 1, cdfstat )
      scaled = got_it .OR. tf2

* netCDF files may use "standard" attributes missing_value and/or
* _FillValue to flag bad data
* For the NOAA COOP standard the missing_value flag must be stored in the
* packed data type.  If so it must be unpacked. 

      got_bad = CD_GET_ATTVAL( cdfid, varid, 'missing_value',
     .         do_warn, vname(:vlen), bad_flag, 1, cdfstat )
      got_missing = CD_GET_ATTVAL( cdfid, varid, '_FillValue',
     .         do_warn, vname(:vlen), missing_flag, 1, cdfstat)

* ... need to unpack the missing_value flag?
*     (note that the netCDF library insists on _FillValue's data type)

      IF ( vartyp.LE.NF_INT .AND. scaled ) THEN
         IF ( got_bad ) THEN
	     cdfstat = NF_INQ_ATT (cdfid, varid, 'missing_value', 
     .             attype, i )
	        IF ( attype .EQ. vartyp ) bad_flag =
     .			bad_flag*scale_factor+ add_offset
	     ENDIF

* ... *kob* Also need to see about unpacking _FillValue

         IF ( got_missing ) THEN
	        cdfstat = NF_INQ_ATT(cdfid, varid, '_FillValue',
     .             attype, i )
	        IF ( attype .EQ. vartyp ) missing_flag =
     .			missing_flag*scale_factor+ add_offset
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
            cdfstat = 0
         ENDIF
      ENDIF

      RETURN
      END
