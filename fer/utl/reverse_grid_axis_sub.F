	SUBROUTINE REVERSE_GRID_AXIS_SUB( axis, dat, ddlo, ddhi, ddel,
     .		dlox, dhix, dloy, dhiy, dloz, dhiz, dlot, dhit )

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

* reverse the grid indexing along the indicated axis
* the passed indices are correct for the *result* of the reversal

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* V320: 2/16/89
* V541:  2/02 *sh* - added support for delta strides

	include 'ferret.parm'

* calling argument declarations:
	INTEGER axis, ddlo, ddhi, ddel,
     .		dlox, dhix, dloy, dhiy, dloz, dhiz, dlot, dhit
	REAL    dat( dlox:dhix,dloy:dhiy,dloz:dhiz,dlot:dhit )

* internal variable declarations:
	INTEGER	i, j, k, l, turn_pt, hiplo
	REAL tmp

* --- end of introductory code ---

* initialize
	turn_pt = ddlo + (ddhi-ddlo-1)/2
	hiplo = ddlo + ddhi

* reverse the data along the appropriate axis

	IF ( axis .EQ. x_dim ) THEN
	   DO 100 l = dlot, dhit
	   DO 100 k = dloz, dhiz
	   DO 100 j = dloy, dhiy
	   DO 100 i = dlox, turn_pt

	      tmp = dat(i,j,k,l)
	      dat(      i,j,k,l) = dat(hiplo-i,j,k,l)
	      dat(hiplo-i,j,k,l) = tmp

 100	   CONTINUE

	ELSEIF ( axis .EQ. y_dim ) THEN
	   DO 200 l = dlot, dhit
	   DO 200 k = dloz, dhiz
	   DO 200 j = dloy, turn_pt
	   DO 200 i = dlox, dhix

	      tmp = dat(i,j,k,l)
	      dat(i,      j,k,l) = dat(i,hiplo-j,k,l)
	      dat(i,hiplo-j,k,l) = tmp

 200	   CONTINUE

	ELSEIF ( axis .EQ. z_dim ) THEN
	   DO 300 l = dlot, dhit
	   DO 300 k = dloz, turn_pt
	   DO 300 j = dloy, dhiy
	   DO 300 i = dlox, dhix

	      tmp = dat(i,j,k,l)
	      dat(i,j,      k,l) = dat(i,j,hiplo-k,l)
	      dat(i,j,hiplo-k,l) = tmp

 300	   CONTINUE

	ELSEIF ( axis .EQ. t_dim ) THEN
	   DO 400 l = dlot, turn_pt
	   DO 400 k = dloz, dhiz
	   DO 400 j = dloy, dhiy
	   DO 400 i = dlox, dhix

	      tmp = dat(i,j,k,l)
	      dat(i,j,k,      l) = dat(i,j,k,hiplo-l)
	      dat(i,j,k,hiplo-l) = tmp

 400	   CONTINUE

	ENDIF

	RETURN
	END 


