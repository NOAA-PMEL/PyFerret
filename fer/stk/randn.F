      REAL FUNCTION RANDN(jd)

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
* compute a quasi-random number with a standard normal distribution
* based on the routine "GASDEV" on p.203 of Numerical Recipes
* W.H.Press,B.P.Flannery,S.A.Teukolsky,W.T.Vetterling

* uses routine RANDU to supply normal deviates

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* V200:  3/29/90
* V6.8 for double-precision Ferret do the RANDU calculation
*      with a real*4 argument. Gives consistent results with single-p.
* 12/2022 *acm* Code cleanup: Argument to RANDU is of type integer

        IMPLICIT NONE
* calling argument declarations:
	INTEGER	jd

* internal variable declarations:
	LOGICAL saved
	REAL	RANDU, v1, v2, r, fac, rsave
	INTEGER izero

	izero = 0

* initialize the uniform and this random generator
	IF ( jd.NE.0 ) THEN
	   v1 = RANDU(jd)
	   saved = .FALSE.
	ENDIF

	IF ( saved ) THEN
* ... use the one saved last time
	   RANDN = rsave
	   saved = .FALSE.

	ELSE
* ... compute 2 normal from 2 uniforms, return 1, save 1
 10	   v1 = 2. * RANDU(izero) - 1.
	   v2 = 2. * RANDU(izero) - 1.
	   r  = v1**2 + v2**2
	   IF ( r .GE. 1 ) GOTO 10	! not inside unit circle ?
	   fac = SQRT(-2.*LOG(r)/r)	   
	   rsave = v1*fac
	   RANDN = v2*fac
	   saved = .TRUE.

	ENDIF

	RETURN
	END
