	INTEGER FUNCTION CD_AXLEN ( cdfid, dimid, status )

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
* get the length of an axis (netCDF coordinate variable)
* the axis is presumed to exist and to be a 1D variable

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 2/92
* 4/94 *sh*,*kob* inserted return value for error exit case - avoid
*		  compiler error msg

* argument definitions:
*	cdfid	- netCDF id number for already opened CDF file
*       dimid   - dimension id of coordinate variable in netCDF file
*       status  - returned status code

* argument declarations
      INTEGER	    cdfid, dimid, status

#ifdef unix
      include 'tmap_format/netcdf.decl'
      include 'tmap_format/netcdf.inc'
      include 'tmap_format/tmap_errors.parm'
#else
      INCLUDE 'TMAP_FORMAT:NETCDF.DECL'
      INCLUDE 'TMAP_FORMAT:NETCDF.INC'
      INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
#endif

* internal variable declarations:
* *kob* explicit definitions not allowed on decl line in linux...moved it to parameter line
      LOGICAL CD_GET_ATTVAL, got_it, do_warn
      PARAMETER (do_warn = .TRUE.)
      INTEGER NCVID, TM_LENSTR1, cdfstat, nval, npts, nlen, axid
      CHARACTER*24 axname
      REAL rlen

* initialize in case of error exit (4/94)
      CD_AXLEN = 0

* get the basic information about this dimension
      CALL NCDINQ( cdfid, dimid, axname, npts, cdfstat )
      IF ( cdfstat .NE. NCNOERR ) GOTO 5100

* get the basics about the axis by the same name
      nlen = TM_LENSTR1( axname )
      axid = NCVID( cdfid, axname(:nlen), cdfstat )

* check "true_size" in case this is a condensed axis representation
      got_it = CD_GET_ATTVAL( cdfid, axid, 'true_size',
     .                        do_warn, axname(:nlen), rlen, 1, nval)
      IF (got_it) npts = rlen

* successful completion
      CD_AXLEN = npts
      status = merr_ok
      RETURN

* error exit
 5000 RETURN
 5100 CALL TM_ERRMSG( cdfstat+pcdferr, status, 'CD_AXLEN', cdfid,
     .                no_varid, no_errstring, no_errstring, *5000 )
      END
