	SUBROUTINE CD_STAMP_OUT(append, cdfid, string, status)

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
* Put a "history" stamp into the current netCDF output file

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 3/92
* 10/96 *kob* - Linux port - had to move variable assignments off of
*		declaration line.
* V65  *acm* 1/10  Change all netcdf2-style calls and parameter names to 
*                  netcdf3-style calls, which are also consistent with netcdf4.

* argument definitions:
*       append  - logical control to create or append
*       cdfid   - returned pointer to open CDF file
*       string  - history string to add
*	status	- return status

* argument declarations:
        LOGICAL  append
	INTEGER	 cdfid, status
        CHARACTER*(*) string

* include files
        include 'netcdf.decl'
#include "netcdf.inc"
        include 'tmap_errors.parm'
        include 'cd_lib.parm'

* local variable declarations:
* *kob* explicit definitions not allowed on decl line in linux...
*       moved it to parameter line 10/96
      INTEGER TM_LENSTR1, STR_SAME, hlen, slen
      LOGICAL CD_GET_ATTRIB, got_it, do_warn, append_flag
      PARAMETER (do_warn = .TRUE. )
      CHARACTER*2048 history

* initialize
      slen = MIN( TM_LENSTR1(string), 120 )

* update (or create) history record
      got_it = CD_GET_ATTRIB( cdfid, NCGLOBAL, 'history',
     .                     .NOT.do_warn, ' ', history, hlen, 2048 )

* is the last history stamp the same as this ?
      IF ( hlen .GE. slen ) THEN
         IF (STR_SAME( history(hlen-slen+1:hlen), string(:slen) ) 
     .      .EQ. 0) RETURN
      ENDIF

* is the last history stamp just a FERRET Vxx  date?
* If so just replace with the current FERRET Vxxx ...
      append_flag = append
      IF ( STR_SAME (history(1:8), 'FERRET V') .EQ.0 .AND.
     .     hlen .LE. 30 ) append_flag = .FALSE.

* insert new signature
      CALL CD_WRITE_ATTRIB(cdfid, pcdf_global, 'history', string(:slen),
     .                     append_flag, status )
      IF ( status .NE. merr_ok ) GOTO 5000

* successful completion
      status = merr_ok
      RETURN

* error exit
 5000 RETURN
	END
