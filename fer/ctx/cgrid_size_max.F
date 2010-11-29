	INTEGER FUNCTION CGRID_SIZE_MAX ( cx )

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
* determine the number of values used on the grid
* of the specified context

* unlike CGRID_SIZE which uses "1" as the size of unspecified regions this
* one uses CAXIS_LEN - the maximum size.  A kludge in guessing this way.

* another kludge is needed for abstract variables -- since the abstract
* axis is deliberately defined to be "arbitrarily" long.  Use a length of
* 1 in this case

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V420: 11/95 - from CGRID_SIZE

#ifdef unix
	include 'ferret.parm'
	include 'xcontext.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
#endif

* calling argument declarations:
	INTEGER	cx

* internal variable declarations:
	INTEGER	CX_DIM_LEN, CAXIS_LEN, CGRID_AXIS, idim, i4d, axlen

* to accomodate the ABSTRACT axis
* (inefficient -- "mline_abstract" should be saved by DEFINE_SPECIAL_GRIDS)

* initialize
	i4d = 1

	DO 100 idim = 1, 4
	   IF (cx_lo_ss(cx,idim) .NE. unspecified_int4) THEN
	      i4d = i4d * CX_DIM_LEN(idim,cx)
	   ELSE
	      axlen = CAXIS_LEN(idim,cx)
	      IF ( axlen .EQ. 9999999 ) THEN
	         axlen = 1	! kludge trap for ABSTRACT axis length
	      ELSEIF ( axlen .GT. 5000 ) THEN
	         axlen = 5000	! kludge 
	      ENDIF
	      i4d = i4d * axlen
	   ENDIF
 100	CONTINUE

	CGRID_SIZE_MAX = i4d

	RETURN
	END
