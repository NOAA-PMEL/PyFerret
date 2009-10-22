	LOGICAL FUNCTION TM_CHECK_BNDS (nline, nb, vname)

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
*
* Checks to make sure bounds are contiguous and the data points are 
* enclosed by its bounds on an axis

* 4/2004 ACM from tm_check_box_loc
* V600 *acm* 3/06  Fix bug 1400: Add more convenient syntax for user-specified bounds:
*                  Bounds can be either 2*N or N+1 values.
* V601 *acm* 9/19  Fix bug 1434; check bounds for regularly-spaced axes 
* V63  *acm* 10/09 Changes for gfortran build

        INCLUDE 'xio.cmn_text'
	INCLUDE 'tmap_dims.parm'
        INCLUDE 'tmap_errors.parm'
	INCLUDE 'xtm_grid.cmn_text'
	EXTERNAL xgt_grid_data

* Arguments
	INTEGER nline, nb
        CHARACTER*(*) vname

* Internal definitions
        LOGICAL TM_FPEQ, TM_DFPEQ
	INTEGER TM_LENSTR1, npts, ipt1, ibx1, i, ngap, vlen, status,
     .          nothing
        REAL*8 pt, del, bhi, blo
        REAL blast
        CHARACTER TM_STRING*13, vbuff*128

	ipt1 = line_subsc1(nline)
	npts = line_dim(nline)
	ibx1 = ipt1 + npts
        IF (ibx1 + nb .GT. maxlinestore) GOTO 5100

        blast = REAL(line_mem(ibx1+1))  ! First box_hi
        ngap = 0

        IF (nb .EQ. 2*npts) THEN

           IF (line_regular(nline)) THEN

              pt = line_start(nline)
              del = line_delta(nline)
	      DO 100 i = 0, npts-1
                 blo = line_mem(ibx1+2*i)
                 bhi = line_mem(ibx1+2*i+1)
	        IF ( (pt .LT. blo) .OR.  (pt .GT. bhi) ) THEN
		
                   IF (TM_DFPEQ(pt,blo) .OR. TM_DFPEQ(pt,bhi) ) THEN
                      nothing = 0
                   ELSE
                      TM_CHECK_BNDS = .FALSE.
		   ENDIF
                   RETURN
                ENDIF
	        pt = line_start(nline) + FLOAT(i+1)* del

  100           CONTINUE

            ELSE ! line_regular

	      DO 200 i = 0, npts-1
                pt = line_mem(ipt1+i)
                blo = line_mem(ibx1+2*i)
                bhi = line_mem(ibx1+2*i+1)
	        IF ( (pt .LT. blo) .OR.  (pt .GT. bhi) ) THEN
		
                   IF (TM_DFPEQ(pt,blo) .OR. TM_DFPEQ(pt,bhi) ) THEN
                      nothing = 0
                   ELSE
                      TM_CHECK_BNDS = .FALSE.
		   ENDIF

                   RETURN
	        ENDIF
                IF (i .GE. 1) THEN
                   IF (.NOT. TM_FPEQ(blast, REAL(blo)) ) THEN
                      ngap = ngap + 1
                      line_mem(ibx1+2*i) = blast
                   ENDIF
                   blast = REAL(bhi)
                ENDIF

  200        CONTINUE
           ENDIF ! line_regular

*  Change from N*2 storage of bounds to N+1 edges.

           DO 300 i = 0, npts-1
              bhi = line_mem(ibx1+2*i+1)
              line_mem(ibx1+i+1) = bhi
  300      CONTINUE

           IF (ngap .EQ. 0) THEN
              TM_CHECK_BNDS = .TRUE.
              RETURN
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

       ELSE  ! bounds are already specified as N+1 values

          IF (line_regular(nline)) THEN
             pt = line_start(nline)
             del = line_delta(nline)
             DO 400 i = 0, npts-1
                blo = line_mem(ibx1+i)
                bhi = line_mem(ibx1+i+1)
	        IF ( (pt .LT. blo) .OR.  (pt .GT. bhi) ) THEN
                   IF (TM_DFPEQ(pt,blo) .OR. TM_DFPEQ(pt,bhi) ) THEN
                      nothing = 0
                   ELSE
                      TM_CHECK_BNDS = .FALSE.
		   ENDIF
                   RETURN
	        ENDIF
                pt = line_start(nline) + FLOAT(i+1)* del

  400        CONTINUE

          ELSE ! line_regular
  
	     DO 500 i = 0, npts-1
                pt = line_mem(ipt1+i)
                blo = line_mem(ibx1+i)
                bhi = line_mem(ibx1+i+1)
	        IF ( (pt .LT. blo) .OR.  (pt .GT. bhi) ) THEN
                   IF (TM_DFPEQ(pt,blo) .OR. TM_DFPEQ(pt,bhi) ) THEN
                      nothing = 0
                   ELSE
                      TM_CHECK_BNDS = .FALSE.
		   ENDIF
                   RETURN
	        ENDIF
                IF (i .GE. 1) THEN
                   IF (.NOT. TM_FPEQ(blast, REAL(blo)) ) THEN
                      ngap = ngap + 1
                      line_mem(ibx1+i) = blast
                   ENDIF
                   blast = REAL(bhi)
                ENDIF

  500        CONTINUE
          ENDIF ! line_regular

       ENDIF

 5000  RETURN
 5100  TM_CHECK_BNDS = .FALSE.
       CALL TM_ERRMSG (merr_linstorlim, status, 'TM_CHECK_BNDS',
     .                  no_descfile, no_stepfile,
     .                  'MAX='//TM_STRING(DBLE(maxlinestore)),
     .                  no_errstring, *5000)

      END
