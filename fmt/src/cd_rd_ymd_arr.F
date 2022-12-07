	SUBROUTINE CD_RD_YMD_ARR(cdfid,t1id,t0,tunit,n,darr,status)

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
* read an array of time values from a "yymmddhhmmss" time axis
* if T0 is not yet known determine an appropriate value
* return time values as a double precision time relative to t0

* NOTE:  This routine could be grestly optimized for performance.
*     This version calls CD_RD_ARR_1 for each point.
*     Instead it could break apart T0 only once and use array reads to read
*     The double integers directly into the returned double precision array
*     The integers could then be re-arranged into pairs and converted to 
*     TMAP times in place.

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 2/95
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

* argument definitions:
*	cdfid	 - netCDF id number for already opened CDF file
*       t1id     - netCDF variable id of "time"
*       t0       - T0 string for the time axis
*       tunit    - number of seconds per axis time unit
*       n        - number of time values to return
*       darr     - double precision array returned
*       status   - TMAP library return status

* argument declarations
      INTEGER cdfid, t1id, n, status
      REAL tunit
      REAL*8  darr(n)
      CHARACTER*(*) t0

      include 'tmap_errors.parm'

* internal variable declarations:
      INTEGER i

      DO i = 1, n
         CALL CD_RD_YMD_1(cdfid,t1id,t0,tunit,i,darr(i),status)
      ENDDO

      RETURN

      END


