	SUBROUTINE CHECK_T_SCALE ( x_or_y, vallo, valhi)

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
* Apply scaling to LO, HI time ends on a 2d time overlay plot

* programmer - ACM 1/3/2018
* (Adapted from box_bounds)
* V73+ 1/18 *acm*  Changes for Issue 1009; precsision in time info sent to pplus for 2D plots
*             scaling for axis coords and box edges
*
	include 'xbox_bounds.cmn'

* calling argument declarations:
	REAL		vallo, valhi
	CHARACTER	x_or_y*1

* internal variable declarations:
	REAL		factr, offst

* initialize
	factr = 1.
	offst = 0.

	IF ( x_or_y .EQ. 'X' ) THEN
	   factr = bb_x_fac
	   offst = bb_x_off
	ELSEIF ( x_or_y .EQ. 'Y' ) THEN
	   factr = bb_y_fac
	   offst = bb_y_off
	ENDIF

	vallo = (vallo + offst) * factr 
	valhi = (valhi + offst) * factr 

	RETURN
	END
