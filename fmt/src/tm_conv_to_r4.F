	SUBROUTINE TM_I4_TO_R4( input, vals, n)

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
* convert an array of INTEGER*4 values into REAL
* use backwards loop so input and output can be same array

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 2/92
* replaced "elif" syntax with
*	else
*	   if
*  for SGI port	 - kob 4/8/92
*
* argument declarations
      INTEGER n
      INTEGER*4 input(n)
      REAL  vals(n)

* local variable declarations
      INTEGER i

      DO 100 i = n, 1, -1
 100  vals(i) = input(i)

      RETURN
      END

***************************************************************

	SUBROUTINE TM_I2_TO_R4( input, vals, n)

* convert an array of INTEGER*2 values into REAL
* use backwards loop so input and output can be same array

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 2/92

* argument declarations
      INTEGER n
      INTEGER*2 input(n)
      REAL  vals(n)

* local variable declarations
      INTEGER i

      DO 100 i = n, 1, -1
 100  vals(i) = input(i)

      RETURN
      END

***************************************************************
	SUBROUTINE TM_I1_TO_R4( input, vals, n)

* convert an array of INTEGER*4 values into REAL
* use backwards loop so input and output can be same array

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 2/92

* argument declarations
      INTEGER n
      REAL  vals(n)
#ifdef sun
      BYTE input(n)
#else
#	if unix
      		INTEGER*1 input(n)
#	else
      		BYTE input(n)
#	endif
#endif

* local variable declarations
      INTEGER i

      DO 100 i = n, 1, -1
 100  vals(i) = input(i)

      RETURN
      END
