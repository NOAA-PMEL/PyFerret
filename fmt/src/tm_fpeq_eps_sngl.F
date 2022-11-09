	LOGICAL FUNCTION TM_FPEQ_EPS_SNGL ( eps, a, b )

*
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration''s (NOAA) Pacific Marine Environmental Lab(PMEL),
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
* test to see if REAL*4 floating point numbers are equal to within
* "reasonable" accuracy
* REAL*4 (F_floating) machine error on the VAX is "approximately one part in
* 2**23" or 1.2E-7.  Allow 4.E-7 as machine error after many typical roundoffs.
* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* programmer - Ansley Manke, from tm_fpeq.F
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* 9/2016 *acm* from TM_FPEQ_EPS
* Single-precision version of TM_FPEQ_EPS, to be called from PPLUS.
* V767 *acm*  6/22 PyFerret issue 116. A correction to diffeps when it
*                  is "small" was incorrect.

* calling arguments:
	REAL*4 eps, a, b

* local variables
        REAL*4 abmin, diffeps

* determine the maximum relative error
        abmin = MIN( ABS(a), ABS(b) )
        diffeps = abmin * ABS(eps)

        TM_FPEQ_EPS_SNGL = ( ABS(a-b) .LE. diffeps )

	RETURN
	END
