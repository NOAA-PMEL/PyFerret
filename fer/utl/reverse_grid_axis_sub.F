	SUBROUTINE REVERSE_GRID_AXIS_SUB( axis, dat, ddlo, ddhi, ddel,
     .		dlox, dhix, dloy, dhiy, dloz, dhiz, 
     .		dlot, dhit, dloe, dhie, dlof, dhif )

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
* 2/12 *acm* Add E and F dimensions (use nferdims in tmap_dims.parm)
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 
* 11/2022 *acm* Add reversal on E and F axes. 

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include 'ferret.parm'

* calling argument declarations:
	INTEGER axis, ddlo, ddhi, ddel,
     .		dlox, dhix, dloy, dhiy, dloz, dhiz, 
     .		dlot, dhit, dloe, dhie, dlof, dhif
	REAL    dat( dlox:dhix,dloy:dhiy,dloz:dhiz,dlot:dhit,dloe:dhie,dlof:dhif )

* internal variable declarations:
	INTEGER	i, j, k, l, m, n, turn_pt, hiplo
	REAL tmp

* --- end of introductory code ---

* initialize
	turn_pt = ddlo + (ddhi-ddlo-1)/2
	hiplo = ddlo + ddhi

* reverse the data along the appropriate axis

	IF ( axis .EQ. x_dim ) THEN
	   DO n = dlof, dhif
	   DO m = dloe, dhie
	   DO l = dlot, dhit
	   DO k = dloz, dhiz
	   DO j = dloy, dhiy
	   DO i = dlox, turn_pt

	      tmp = dat(i,j,k,l,m,n)
	      dat(      i,j,k,l,m,n) = dat(hiplo-i,j,k,l,m,n)
	      dat(hiplo-i,j,k,l,m,n) = tmp

	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

	ELSEIF ( axis .EQ. y_dim ) THEN
	   DO  n = dlof, dhif
	   DO  m = dloe, dhie
	   DO  l = dlot, dhit
	   DO  k = dloz, dhiz
	   DO  j = dloy, turn_pt
	   DO  i = dlox, dhix

	      tmp = dat(i,j,k,l,m,n)
	      dat(i,      j,k,l,m,n) = dat(i,hiplo-j,k,l,m,n)
	      dat(i,hiplo-j,k,l,m,n) = tmp

	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

	ELSEIF ( axis .EQ. z_dim ) THEN
	   DO n = dlof, dhif
	   DO m = dloe, dhie
	   DO l = dlot, dhit
	   DO k = dloz, turn_pt
	   DO j = dloy, dhiy
	   DO i = dlox, dhix

	      tmp = dat(i,j,k,l,m,n)
	      dat(i,j,      k,l,m,n) = dat(i,j,hiplo-k,l,m,n)
	      dat(i,j,hiplo-k,l,m,n) = tmp

	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

	ELSEIF ( axis .EQ. t_dim ) THEN
	   DO n = dlof, dhif
	   DO m = dloe, dhie
	   DO l = dlot, turn_pt
	   DO k = dloz, dhiz
	   DO j = dloy, dhiy
	   DO i = dlox, dhix

	      tmp = dat(i,j,k,l,m,n)
	      dat(i,j,k,      l,m,n) = dat(i,j,k,hiplo-l,m,n)
	      dat(i,j,k,hiplo-l,m,n) = tmp

	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO


	ELSEIF ( axis .EQ. e_dim ) THEN
	   DO n = dlof, dhif
	   DO m = dloe, turn_pt
	   DO l = dlot, dhit
	   DO k = dloz, dhiz
	   DO j = dloy, dhiy
	   DO i = dlox, dhix

	      tmp = dat(i,j,k,l,m,n)
	      dat(i,j,k,l,      m,n) = dat(i,j,k,l,hiplo-m,n)
	      dat(i,j,k,l,hiplo-m,n) = tmp

	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

	ELSEIF ( axis .EQ. f_dim ) THEN
	   DO n = dlof, turn_pt
	   DO m = dloe, dhie
	   DO l = dlot, dhit
	   DO k = dloz, dhiz
	   DO j = dloy, dhiy
	   DO i = dlox, dhix

	      tmp = dat(i,j,k,l,m,n)
	      dat(i,j,k,l,m,      n) = dat(i,j,k,l,m,hiplo-n)
	      dat(i,j,k,l,m,hiplo-n) = tmp

	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
* ALONG UNSUPPORTED AXIS
        ELSE
           STOP '**ERROR: REVERSE_GRID_AXIS_SUB on an unsupported axis'
        ENDIF

	RETURN
	END 


