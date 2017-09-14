	LOGICAL FUNCTION TM_CHECK_BNDS (line, line_edges, npts, 
     .                         iaxis, doubletype, vname)

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
* Checks to make sure bounds are contiguous and the data points are 
* enclosed by its bounds on an axis

* 4/2004 ACM from tm_check_box_loc
* V600 *acm* 3/06  Fix bug 1400: Add more convenient syntax for user-specified bounds:
*                  Bounds can be either 2*N or N+1 values.
* V601 *acm* 9/19  Fix bug 1434; check bounds for regularly-spaced axes 
* V63  *acm* 10/09 Changes for gfortran build
* V65  *acm*  2/10 Test for overlapping bounds
* V66  *kms*  7/10 Initialize return value to .TRUE. (uninitialized bug for
*                  N+1 bounds) and no need to check for gaps for N+1 bounds
* V694  5/15 *acm* Match precision for bounds checks (as is done for coords).
* V696 10/15 *acm* Ticket 2321: Fix logic when detecting coords that match grid bounds
* V72 6/17 *acm*   For trac enhancement #767 -- dynamic coordinate storage. 

        INCLUDE 'xio.cmn_text'
	INCLUDE 'tmap_dims.parm'
        INCLUDE 'tmap_errors.parm'
	INCLUDE 'xtm_grid.cmn_text'
	EXTERNAL xgt_grid_data

* Arguments
	LOGICAL doubletype
	INTEGER iaxis, npts
	REAL	line(*), line_edges(*)
        CHARACTER*(*) vname

* Internal definitions
        LOGICAL TM_FPEQ_EPS, TM_DFPEQ_TOL, equal, equal_ends, overlapping
	INTEGER TM_LENSTR1, ipt1, ibx1, i, ngap, vlen, status
        REAL*8 pt, del, bhi, blo
	
	REAL epsilon_23, epsilon
	REAL*8 depsilon_46, depsilon, first_delta
        REAL blast
        CHARACTER TM_STRING*13, vbuff*128
	
        epsilon_23 =  2.**(-23) 
        depsilon_46 =  2.**(-46) 
        epsilon_23 =  2.**(-22) 
        depsilon_46 =  2.**(-45) 

	blast = line_edges(2)  ! First box-hi
        TM_CHECK_BNDS = .TRUE.
        ngap = 0
        overlapping = .FALSE.

        first_delta = line(2) - line(1)
        depsilon = depsilon_46 * 2.*(ABS(line(1)) / first_delta )
        epsilon = epsilon_23 * 2.*(ABS(line(1)) / first_delta )

        DO i = 1, npts
           pt = line(i)
           blo = line_edges(2*(i-1)+1)
           bhi = line_edges(2*(i-1)+2)
           IF ( (pt .LT. blo) .OR.  (pt .GT. bhi) ) THEN

              IF (doubletype) THEN
                 equal = TM_DFPEQ_TOL(pt,blo,depsilon) .OR. TM_DFPEQ_TOL(pt,bhi,depsilon)
              ELSE
                 equal = TM_FPEQ_EPS(epsilon,pt,blo) .OR. TM_FPEQ_EPS(epsilon,pt,bhi)
              ENDIF
              IF (.NOT. equal) THEN
                 TM_CHECK_BNDS = .FALSE.
                 GOTO 5000
              ENDIF

              GOTO 5000
           ENDIF
           IF (i .GT. 1) THEN
              IF (doubletype) THEN
                 equal_ends = TM_DFPEQ_TOL(blast, REAL(blo), depsilon)
              ELSE
                 equal_ends = TM_FPEQ_EPS(epsilon, blast, REAL(blo))
              ENDIF
              IF (.NOT.equal_ends .AND. blo.LT.blast) THEN 
                 overlapping = .TRUE.
                 GOTO 330
              ENDIF
              IF (.NOT. equal_ends ) THEN
                 ngap = ngap + 1
                 line_edges(2*(i-1)+1) = blast
              ENDIF
              blast = REAL(bhi)
	      ENDIF

	ENDDO

	IF (ngap .EQ. 0) THEN
	   TM_CHECK_BNDS = .TRUE.
	   GOTO 5000
	ELSE

	   vbuff = vname   
	   vlen = TM_LENSTR1(vbuff)
	   CALL WARN( 'Axis definition error on axis: ' 
     .               //vbuff(:vlen))
	   CALL WARN(
     .        'Bounds given describe non-contiguous axis cells ')
	   CALL WARN(
     .         'Using the LOWER bound specified for each axis cell')
	   TM_CHECK_BNDS = .TRUE.

	ENDIF

  330	IF (overlapping) THEN
	   vbuff = vname   
	   vlen = TM_LENSTR1(vbuff)
	   CALL WARN( 'Axis definition error on axis: '
     .        //vbuff(:vlen)//'. Bounds describe cells '//
     .        'that overlap one another ')
	   TM_CHECK_BNDS = .FALSE.
	ENDIF

 5000  RETURN

      END
