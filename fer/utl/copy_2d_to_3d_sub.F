	SUBROUTINE COPY_2D_TO_3D_SUB (
     .		src, slox, shix, sloy, shiy,             slot, shit,
     .		dst, dlox, dhix, dloy, dhiy, dloz, dhiz, dlot, dhit )

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
* support routine for COPY_2D_TO_3D with unsubscripted limits

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200: 10/12/89

#ifdef unix
	include 'ferret.parm'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
#endif

* calling argument declarations:
	INTEGER slox, shix, sloy, shiy,             slot, shit,
     .		dlox, dhix, dloy, dhiy, dloz, dhiz, dlot, dhit
	REAL    src( slox:shix,sloy:shiy,          slot:shit ),
     .		dst( dlox:dhix,dloy:dhiy,dloz:dhiz,dlot:dhit )

* internal variable declarations:
	INTEGER	i, j, k, l, k1

* --- end of introductory code ---

* copy one XY plane to the dstination grid
	k1 = dloz

	DO 100 l = dlot, dhit
	DO 100 j = dloy, dhiy
	DO 100 i = dlox, dhix

	   dst(i,j,k1,l) = src(i,j,l)

 100	CONTINUE

* (no check for bad data - same flag assumed on source and destination)

* now replicate along the Z axis
	DO 210 k = k1+1, dhiz

	   DO 200 l = dlot, dhit
	   DO 200 j = dloy, dhiy
	   DO 200 i = dlox, dhix
 200	   dst(i,j,k,l) = dst(i,j,k1,l)

 210	CONTINUE

	RETURN
	END 
