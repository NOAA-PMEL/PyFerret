	SUBROUTINE CD_CONVENTIONS_OUT(append, cdfid, string, status)

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
* Put a "conventions" attribute into the current netCDF output file

* Programmer Ansley Manke
* NOAA/PMEL, Seattle, WA - TMAP

* 7/2007 from CD_STAMP_OUT
* V65  *acm* 1/10  Change all netcdf2-style calls and parameter names to 
*                  netcdf3-style calls, which are also consistent with netcdf4.
* V6.74 3/12 *acm* cleanup ifdefs and unnecessary include files

* argument definitions:
*       append  - logical control to create or append
*       cdfid   - returned pointer to open CDF file
*       string  - conventions string to add
*	status	- return status

* argument declarations:
        LOGICAL  append
	INTEGER	 cdfid, status
        CHARACTER*(*) string

* include files
        include 'netcdf.inc'
        include 'tmap_errors.parm'
        include 'cd_lib.parm'

* local variable declarations:

      INTEGER TM_LENSTR1, STR_SAME, clen, slen
      LOGICAL CD_GET_ATTRIB, got_it, do_warn, append_flag
      PARAMETER (do_warn = .TRUE. )
      CHARACTER*132 buff
      CHARACTER*1 nl

* initialize
      slen = MIN( TM_LENSTR1(string), 120 )

* update (or create) conventions record
      got_it = CD_GET_ATTRIB( cdfid, NCGLOBAL, 'Conventions',
     .                     .NOT.do_warn, ' ', buff, clen, 132 )

* is the last conventions attribute the same as this ?
      IF ( clen .GE. slen ) THEN
         IF ( buff(clen-slen+1:clen) .EQ. string(:slen) ) RETURN
      ENDIF
      
* is the last conventions attribute just a "CF-xx?
* If so just replace with the current string  ...
      append_flag = append
      IF ( STR_SAME (buff(1:3), 'CF-') .EQ.0 .AND.
     .     clen .LE. 12 ) append_flag = .FALSE.

* Insert new CF conventions text
* If there is a conventions att, append current one.
* For conventions, just add it with a comma.

      IF (append_flag) THEN
c         nl = NEW_LINE('a')
c         CALL CD_WRITE_ATTRIB(cdfid, pcdf_global, 'Conventions', 
c     .                     ','//nl(:1)//' '//string(:slen),
c     .                     append_flag, status )
         CALL CD_WRITE_ATTRIB(cdfid, pcdf_global, 'Conventions', 
     .                     ', '//string(:slen),
     .                     append_flag, status )
      ELSE
         CALL CD_WRITE_ATTRIB(cdfid, pcdf_global, 'Conventions', string(:slen),
     .                     append_flag, status )
      ENDIF

      IF ( status .NE. merr_ok ) GOTO 5000

* successful completion
      status = merr_ok
*      RETURN

* error exit
 5000 RETURN
	END
