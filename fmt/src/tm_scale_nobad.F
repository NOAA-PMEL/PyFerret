	SUBROUTINE TM_SCALE_NOBAD( factor, offset, array, n, bad, vartyp )

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
* rescale an array: multiply by a scale factor and add an offset, skipping
*    any missing data

* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* from tm_scale.F  4/21/03
* v73+ *acm* 12/2017 Issue # 1846, writing user-variables with scale_factor, add_offset
*                    changes to check for data range when scaling as in tm_scale_all.

* include files
      INCLUDE 'netcdf.inc'

* calling argument declarations:
      INTEGER n, vartyp
      REAL factor, offset, array(n), bad

* internal variable declarations:
      LOGICAL TM_FPEQ_SNGL
      REAL*4 aa, sbad, sbadn
      INTEGER i

      sbad = 32767
      sbadn = -32768

      DO 100 i = 1, n
         IF (array(i) .NE. bad) THEN
            array(i) = array(i)*factor + offset
            IF (vartyp .EQ. NF_SHORT) THEN
               aa =  array(i) 
	       IF (TM_FPEQ_SNGL(aa,sbad) ) array(i) = sbad
	       IF (TM_FPEQ_SNGL(aa,sbadn) ) array(i) = sbadn
            ENDIF
         ENDIF
  100 CONTINUE

      RETURN
      END
