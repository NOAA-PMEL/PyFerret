	SUBROUTINE TM_ADJUST_BOUNDS(iaxis, npts, modlen, axwwlen, ok)


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
* Addressing ticket 2504:
* If coordinates on an irregular axis lie within the modulo length but 
* their bounds make the axis too long, reset the bounds so the axis 
* is the modulo length. This is only called when Ferret is generating
* bounds on a DEFINE AXIS given without cell edges or from a netCDF file
* where bounds were not included in the file.
*
* V72 6/17 *acm* for trac enhancement #767 -- dynamic coordinate storage
*
* return axwwlen

#	include	"tmap_dset.parm"
	include	'tmap_dims.parm'
	include 'xtm_grid.cmn_text'

* Argument declarations
	LOGICAL ok
	INTEGER iaxis, npts
	REAL	modlen, axwwlen

* Local declarations
	LOGICAL TM_FPEQ
	INTEGER TM_LENSTR1, ipt1, ipte
	REAL	GET_LINE_COORD, TM_WW_AXLEN, 
     .		delta, pt1, ptn
	REAL*4	r4delta

* 
	ok = .TRUE.
	axwwlen = TM_WW_AXLEN(iaxis)

* Exit if the modulo length is 0. which indicates modulo length is axis length.
	IF (modlen .EQ. 0.) GOTO 5000

* Exit if the axis length is within the modulo length.
	IF (axwwlen .LE. modlen) GOTO 5000

* Exit if the line is regularly spaced
	IF ( line_regular(iaxis) ) GOTO 5000

	pt1 = GET_LINE_COORD (linemem(iaxis)%ptr,1)
	ptn = GET_LINE_COORD (linemem(iaxis)%ptr, npts)
	delta = modlen - (ptn  - pt1)
	
	CALL PUT_LINE_COORD ( lineedg(iaxis)%ptr, 1, pt1 - delta/2. )
	CALL PUT_LINE_COORD ( lineedg(iaxis)%ptr, npts+1, ptn + delta/2. )

* did the adjustment keep the length within the defined moudulo length?
* (probably always the case if we've got this far.)

	axwwlen = TM_WW_AXLEN(iaxis)
	
	r4delta = axwwlen ! single precision
	delta = r4delta
	ok = (TM_FPEQ(delta, modlen))
	IF (.NOT.ok) axwwlen = 0.
	
 5000	RETURN
	END
