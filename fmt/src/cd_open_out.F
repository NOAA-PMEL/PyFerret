	SUBROUTINE CD_OPEN_OUT(cdfname, append, cdfid, clobber, status)

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
* Open a netCDF file for output
* if appending then the file must already exist
* leave it in definition mode

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 3/92
*               10/92 - leave mode as opened (report via CD_SET_MODE)
* V450 11/96 - added argument "clobber" *sh*

* argument definitions:
*       cdfname - desired output file name
*       append  - logical control to create or append
*       cdfid   - returned pointer to open CDF file
*	clobber	- Boolean to say if existing file can be clobbered
*	status	- return status

* argument declarations:
        LOGICAL  append, clobber
        INTEGER	 cdfid, status
        CHARACTER*(*) cdfname

* include files
#ifdef unix
        include 'netcdf.decl'
#include "netcdf.inc"
        include 'cdf_tmap.parm'
        include 'tmap_errors.parm'
#include "gt_lib.parm"
#else
        INCLUDE 'TMAP_FORMAT:NETCDF.DECL'
#include "netcdf.inc"
        INCLUDE 'TMAP_FORMAT:CDF_TMAP.PARM'
        INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
        INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
#endif

* local variable declarations:
      LOGICAL file_exists, do_append
      INTEGER NCOPN, NCCRE, cdfstat, cstat, clobber_mode

* initialize netCDF
      CALL NCPOPT(0)  ! no error processing by netCDF

* does it exist already ?
      INQUIRE ( FILE = cdfname, EXIST = file_exists )
      do_append = append .AND. file_exists

* open or create the file
      IF ( do_append ) THEN
	 cdfid = NCOPN( cdfname, NCWRITE, cdfstat )
         IF ( cdfstat .NE. NCNOERR ) GOTO 5100
         CALL CD_SET_MODE( cdfid, pcd_reset_data, status )
         IF ( status .NE. merr_ok ) RETURN
      ELSE
	 IF (clobber) THEN
	   clobber_mode =   NCCLOB
	 ELSE
	   clobber_mode =   NCNOCLOB
	 ENDIF
* ... let netCDF detect the error if the file already exists
         cdfid = NCCRE( cdfname, clobber_mode, cdfstat )
         IF ( cdfstat .NE. NCNOERR ) GOTO 5100
         CALL CD_SET_MODE( cdfid, pcd_reset_define, status )
         IF ( status .NE. merr_ok ) RETURN
      ENDIF

* successful completion
	status = merr_ok
	RETURN

* error messages
 5100 CALL TM_ERRMSG
     .     ( cdfstat+pcdferr, status, 'CD_OPEN_OUT', unspecified_int4,
     .     no_varid, 'could not open CDF output file: ',
     .     cdfname, *5000 )

* error exit
 5000 RETURN
	END
