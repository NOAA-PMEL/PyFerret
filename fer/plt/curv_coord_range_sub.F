	SUBROUTINE CURV_COORD_RANGE_SUB ( uc, xfield, yfield, ilo, ihi, 
     .              jlo, jhi, status )

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
* Compute the range within X and Y coordinates actually needed for the 
* hlimits and vlimits that were requested.
*
* V581:  1/2005 ACM 
* V600:  9/2005 acm Fix bug 1342, curvi data with 1-d coordinates.
*                   The loop 200 controls were wrong.
* V690:  4/2006 fix bug where if an axis is reversed (depth axis) the
*               comparisons also need to be reversed.


	include 'ferret.parm'
	include 'errmsg.parm'
	include	'xvariables.cmn'
	include	'xcurvilinear.cmn'

* calling argument declarations:
	INTEGER ilo, ihi, jlo, jhi, status
	REAL    uc(*),
     .		xfield(xpos_loss_dim1:xpos_hiss_dim1,
     .		       xpos_loss_dim2:xpos_hiss_dim2),
     .		yfield(ypos_loss_dim1:ypos_hiss_dim1,
     .		       ypos_loss_dim2:ypos_hiss_dim2)

* internal variable declarations:
	INTEGER i, j
        REAL xx, yy, up, dn

        ilo = MAX(xpos_hiss_dim1, xpos_loss_dim1)
        ihi = MIN(xpos_hiss_dim1, xpos_loss_dim1)
        up = MAX(uc(1), uc(2))
        dn = MIN(uc(1), uc(2))

	DO 100 i = xpos_loss_dim1, xpos_hiss_dim1
        DO 100 j = xpos_loss_dim2, xpos_hiss_dim2
           xx = xfield(i,j)
	   IF ( xx .EQ. badx) THEN
              GOTO 5100 
           ELSE
              IF ( xx .GE. dn.AND.  xx .LE. up ) THEN
                 ilo = MIN(i,ilo)
                 ihi = MAX(i,ihi)
              ENDIF
           ENDIF
  100   CONTINUE

        jlo = MAX(ypos_hiss_dim2, ypos_loss_dim2)
        jhi = MIN(ypos_hiss_dim2, ypos_loss_dim2)
        up = MAX(uc(3), uc(4))
        dn = MIN(uc(3), uc(4))

	DO 200 i = ypos_loss_dim1, ypos_hiss_dim1
        DO 200 j = ypos_loss_dim2, ypos_hiss_dim2
           yy = yfield(i,j)
	   IF ( yy .EQ. bady) THEN
              GOTO 5200 
           ELSE
              IF ( yy .GE. dn .AND. yy .LE. up ) THEN
                 jlo = MIN(j, jlo)
                 jhi = MAX(j, jhi)
              ENDIF
           ENDIF
  200   CONTINUE

* successful completion
	status = 0  ! (like ferr_ok but for PLOT+)
	RETURN

* error exit
 5000	RETURN
 5100	CALL ERRMSG( ferr_out_of_range, status,
     .		'X coordinates'//pCR//
     .		'missing value where val is needed', *5000)
 5200	CALL ERRMSG( ferr_out_of_range, status,
     .		'Y coordinates'//pCR//
     .		'missing value where val is needed', *5000)
	END 
