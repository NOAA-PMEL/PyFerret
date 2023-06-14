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
* V600:  4/2006 fix bug where if an axis is reversed (depth axis) the
*               comparisons also need to be reversed.
* V602:  2/2007 Fix bug 1485; when determining the range of indices to
*               use loop over the indices used in the projection; minss_dim1
*               minss_dim2 etc. 
* V68  *acm* 1/12  changes for double-precision ferret, single-precision pplus.
*       *acm* 3/12 cleanup ifdefs and unnecessary include files
*       *acm* 3/12 6D Ferret (common uses nferdims in tmap_dims.parm)
* V687 *acm* 3/14 tickets 2151 and 2147: allow missing coordinate data in 3-arg 
*                 and 4-arg plot commands. 
* V745 *acm* 3/19 Issue 1914: comparison of single-precision data in uc() and the
*                 coordinate data must be done in single precision using TM_FPEQ_SNGL
*               * Also clean up old code.
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'errmsg.parm'
	include	'xvariables.cmn'
	include	'xcurvilinear.cmn'

* calling argument declarations:
	INTEGER ilo, ihi, jlo, jhi, status
	REAL*4  uc(*)
	REAL*8	xfield(xpos_loss_dim1:xpos_hiss_dim1,
     .		       xpos_loss_dim2:xpos_hiss_dim2),
     .		yfield(ypos_loss_dim1:ypos_hiss_dim1,
     .		       ypos_loss_dim2:ypos_hiss_dim2)

* internal variable declarations:

	LOGICAL TM_FPEQ_SNGL 
	INTEGER i, j, i1, i2, j1, j2
        REAL*4 xx, yy, up, dn

        ilo = MAX(minss_dim1, maxss_dim1)
        ihi = MIN(minss_dim1, maxss_dim1)
        
        up = MAX(uc(1), uc(2))
        dn = MIN(uc(1), uc(2))

* Loop over the indices to be used for the projection.
* For map projections, a coord field may be 1D, in which case
* set any missing indices to the unspecified-int flag.

        i1 = minss_dim1
        IF (xpos_loss_dim1 .EQ. unspecified_int4) i1 = xpos_loss_dim1
        i2 = maxss_dim1
        IF (xpos_hiss_dim1 .EQ. unspecified_int4) i2 = xpos_hiss_dim1
        j1 = minss_dim2
        IF (xpos_loss_dim2 .EQ. unspecified_int4) j1 = xpos_loss_dim2
        j2 = maxss_dim2
        IF (xpos_hiss_dim2 .EQ. unspecified_int4) j2 = xpos_hiss_dim2

        DO i = i1, i2
        DO j = j1, j2
           xx = xfield(i,j)
	   IF ( xx .EQ. badxcurv) THEN
              CYCLE
           ELSE
              IF ( xx .GE. dn.AND.  xx .LE. up ) THEN
                 ilo = MIN(i,ilo)
                 ihi = MAX(i,ihi)
              ENDIF
	      IF (TM_FPEQ_SNGL(xx, dn)) ilo = MIN(i,ilo)
	      IF (TM_FPEQ_SNGL(xx, up)) ihi = MAX(i,ihi)
           ENDIF
	ENDDO
	ENDDO

        jlo = MAX(minss_dim2, maxss_dim2)
        jhi = MIN(minss_dim2, maxss_dim2)

        up = MAX(uc(3), uc(4))
        dn = MIN(uc(3), uc(4))

* Loop over the indices to be used for the projection.
* For map projections, a coord field may be 1D, in which case
* set any missing indices to the unspecified-int flag.

        i1 = minss_dim1
        IF (ypos_loss_dim1 .EQ. unspecified_int4) i1 = ypos_loss_dim1
        i2 = maxss_dim1
        IF (ypos_hiss_dim1 .EQ. unspecified_int4) i2 = ypos_hiss_dim1
        j1 = minss_dim2
        IF (ypos_loss_dim2 .EQ. unspecified_int4) j1 = ypos_loss_dim2
        j2 = maxss_dim2
        IF (ypos_hiss_dim2 .EQ. unspecified_int4) j2 = ypos_hiss_dim2

        DO i = i1, i2
        DO j = j1, j2
           yy = yfield(i,j)
	   IF ( yy .EQ. badycurv) THEN
              CYCLE
           ELSE
              IF ( yy .GE. dn .AND. yy .LE. up ) THEN
                 jlo = MIN(j, jlo)
                 jhi = MAX(j, jhi)
              ENDIF
	      IF (TM_FPEQ_SNGL(yy, dn)) jlo = MIN(j, jlo)
	      IF (TM_FPEQ_SNGL(yy, up)) jhi = MAX(j, jhi)
           ENDIF
        ENDDO
	ENDDO

* successful completion
	status = ferr_ok
	RETURN

	END 
