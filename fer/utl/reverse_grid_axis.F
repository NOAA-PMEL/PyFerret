	SUBROUTINE REVERSE_GRID_AXIS ( axis, cx, mr, dat )

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
* reverse the ordering of the data along the indicated axis

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* V320:  2/16/95
* V541:  2/02 *sh* - added support for delta strides
* V560: 10/03 *acm* - fix bug 537, when use/order=x-y and strides, 
*                     need to reset the indices to apply to the data
*                     as it has been read, after the strides applied, 
*                     e.g. reset 4:360:3 to 1:119:1
* 560: *acm* 4/04 - stride/modulo fixes: use NINT to convert REAL*8 to integer

	include 'ferret.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'

* calling argument declarations:
	INTEGER axis, cx, mr
	REAL    dat( * )

* local variable declarations
	INTEGER idim, ilo(4), ihi(4), idel(4)

* get the delta values
	DO idim = 1, 4
	  ilo (idim) = cx_lo_ss(cx,idim)
	  ihi (idim) = cx_hi_ss(cx,idim)
	  IF (cx_delta(idim,cx) .EQ. unspecified_val8 ) THEN
	    idel(idim) = 1
	  ELSE
	    idel(idim) = NINT(cx_delta(idim,cx))
     
c	    ihi(idim) = 1 + (ihi(idim) - ilo(idim)) / idel(idim)
c	    ilo(idim) = 1
	    ihi(idim) = ilo(idim) + (ihi(idim) - ilo(idim)) / idel(idim)
            idel(idim) = 1

	  ENDIF
	ENDDO

	CALL REVERSE_GRID_AXIS_SUB( axis, dat,
     .				    ilo(axis), ihi(axis), idel(axis),
     .			            ilo(1), ihi(1),
     .			            ilo(2), ihi(2),
     .			            ilo(3), ihi(3),
     .			            ilo(4), ihi(4) )

	RETURN
	END 
