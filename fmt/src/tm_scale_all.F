	SUBROUTINE TM_SCALE_ALL ( factor, offset, array, vartyp, n  )

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
* rescale an array: multiply by a scale factor and add an offset

* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* revision 0.0 1/2010 for TMAP library
* Scale the bad data too... on input to netCDF we scale all the data,
* compute the missing flag for internal use as indat*inscale + inoff
*
* To write scaled data, scale all of it by ferretdat(1./inscale) - (inoff/inscale)
* the scaling to be used is sent into this routine.
* v6.8 3/14/2012 *acm* fix for scaled SHORT data and double-precision Ferret.

* include files
c        include 'netcdf.decl'
#include "netcdf.inc"

* calling argument declarations:
      INTEGER n, vartyp
      REAL factor, offset, array(*)

* internal variable declarations:
      LOGICAL TM_FPEQ_SNGL
      INTEGER i
      REAL*4 aa, sbad

* For short output datatype, the scaling and un-scaling
* can result in data outside the range of SHORT on output.
* FLOAT offset and scale factor are converted to DOUBLE, then
* 1./scale is computed, and re-applied to the data.
* This is a bit of a hack. 

      IF (vartyp .EQ. NF_SHORT) THEN
         sbad = 32767
         DO 100 i = 1, n
	    array(i) = array(i)*factor + offset
	    aa = array(i)
	    IF (TM_FPEQ_SNGL(ABS(aa),sbad) ) array(i) = sbad
 100     CONTINUE

      ELSE
         DO 200 i = 1, n
             array(i) = array(i)*factor + offset
 200     CONTINUE
      ENDIF

      RETURN
      END
