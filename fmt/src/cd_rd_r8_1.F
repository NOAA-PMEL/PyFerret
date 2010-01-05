	SUBROUTINE CD_RD_R8_1(cdfid,varid,index,dtype,errstr,dval,status)

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
* read a single value from a netCDF variable and return it as a double prec.
* The variable must be a simple array (e.g. an axis)

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 2/92

* argument definitions:
*	cdfid	 - netCDF id number for already opened CDF file
*       varid    - variable id in netCDF file
*       index    - subscript of array to return
*	dtype    - data type of variable
*       errstr   - identifying string to be printed with error messages
*       vname    - name of variable owning attribute (used in error msgs)
*       dval     - double precision value returned
*       status   - TMAP library return status

* argument declarations
      INTEGER cdfid, varid, index, dtype, status
      REAL*8  dval
      CHARACTER*(*) errstr

      include 'netcdf.decl'
#include "netcdf.inc"
      include 'tmap_errors.parm'

* internal variable declarations:
      INTEGER cdfstat

* read the value
* NF_ routines convert to desired output data type.

      cdfstat = NF_GET_VAR1_DOUBLE(cdfid, varid, index, dval)

      IF ( cdfstat .NE. NF_NOERR ) CALL TM_ERRMSG
     .     ( cdfstat+pcdferr, status, 'CD_RD_R8_1', cdfid, varid,
     .       errstr, no_errstring, *5000 )

      status = merr_ok

 5000 RETURN

      END
