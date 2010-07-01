      SUBROUTINE set_levels_shade_fill (status)
	
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
* V630  *acm* 9/09 Introduction of syntax for variance-based and histogram levels
* V631  *acm*10/09 Fix bug 1704: LEVELS=xD was incorrect.
* V663  *acm* 5/10 Fix bug 1744: hang if /LEVELS=v and all missing data.
	
	include		'parampl5_dat.decl'
	include		'PARAMPL5.DAT'
	include		'axis_inc.decl'
	include		'AXIS.INC'
	include		'plt_inc.decl'
	include		'PLT.INC'
	include		'miss_inc.decl'
	include		'MISS.INC'
	include		'hd_inc.decl'
	include		'HD.INC'
	include		'cont_inc.decl'
	include		'CONT.INC'
        include         'errmsg.parm'
	
* argument declarations
	INTEGER		status

* local declarations
	REAL		dz, fm, znp, zmp, dzmax
	INTEGER		m, jj
        LOGICAL		TM_FPEQ

        EXTERNAL range

	status = ferr_ok
	lev_str = 'none'
C       
C       IAUTOC = 1 means: compute levels from input data and number
C                         of levels

	IF (IAUTOC.EQ.1) THEN

	   IF (ZMIN.EQ.ZMAX) THEN

C             NLEV2=0 is the code in pplmod_plotz
	      NLEV2=1
              zlev(1) = zmin

	   ELSE IF (ZMIN .GT. CMZGE .AND. ZMAX .LE. CMZLE) THEN  ! all missing

C             NLEV2=0 is the code in pplmod_plotz
	      NLEV2=1
              zlev(1) = zmin

	   ELSE
C       
C       If set_delta is .true. the delta value has been set
C       elsewhere and we should calculate the max and min values
C       of the range based on this delta.
C       (This portion stolen from RANGE)

	      if (set_delta) then 

		 dz = delta
C       
C       CALCULATE THE MINIMUM VALUE OF THE RANGE
C       
		 FM = (zmin) / DZ
		 M = IFIX(FM)
		 IF (FM .LT. 0.0) M = M - 1
		 znp = DZ * FLOAT(M)
C       
C       CALCULATE THE MAXIMUM VALUE OF THE RANGE.
C       
		 FM = (zmax) / DZ
		 M = IFIX(FM)
		 IF(FM .GT. 0.0)M = M + 1
		 zmp = DZ * FLOAT(M)
C
C       Set levels centered about zero if centered is .true.
C
	         if (centered) then
		    if (abs(znp) .gt. abs(zmp)) then
		       zmp = -znp
		    else
		       znp = -zmp
		    endif
	         endif

              else if (openlev) then
	         CALL set_open_levels

C       Else we should allow RANGE to determine the max and min range
C       values and also the appropriate delta value. (ZNP, ZMP, DZ)

	      else

	         znp = zmin
	         zmp = zmax
	         if (centered) then
		    if (abs(znp) .gt. abs(zmp)) then
		       zmp = -znp + zmean
		    else
		       znp = -zmp + zmean
		    endif
	         endif
		 CALL RANGE(znp,zmp,NLEV,ZNP,ZMP,DZ)

	      endif

              IF (.NOT.openlev) THEN
	      NLEV2=(ZMP-ZNP)/DZ+1 

c *acm* 4/08 Set jj as in plotz

              IF (TM_FPEQ((ZNP + DZ*(NLEV2-1)), ZMP) .EQV. .FALSE.) THEN
                 IF (ZNP + DZ*(NLEV2-1) .LT. ZMP) NLEV2 = NLEV2 + 1
	      ENDIF

	      IF (ZMP-ZNP .EQ. 0) THEN
                 JJ=NSIGC
              ELSE
                 JJ=NSIGC-IFIX(ALOG10(ABS(ZMP-ZNP))+.5)
              ENDIF
              IF (JJ.LE.0) JJ=-1

c *acm* 3/04 under g77 on linux this incorrectly uses
c            low-order bits to reset NLEV2 example:
c             yes?shade/lev=c {-1.1, 1.1}* ysequence({1,1})

c       4/04  But, dont just skip this, use TM_FPEQ to see if
c             values are equal... 
c       4/04  Also fix the negative-range case.

	      IF ( (ZNP + DZ*(NLEV2-1) .LT. ZMP) .AND. 
     .              .NOT. TM_FPEQ(ZNP + DZ*(NLEV2-1), ZMP) ) THEN
		 NLEV2 = NLEV2 + 1
	      ENDIF

              IF (NLEV2 .GT. ICNLEV) GO TO 5000

C acm 12/01  Set LWGT for future contour line overlays.

	      DO 100 I=1,NLEV2 

                 LWGT(I)=2
                 LDIG(I)=-1
                 ZLEV(I)=ZNP+DZ*FLOAT(I-1)
                 IF (ABS(ZLEV(I)).LT.1.E-5*DZ) GOTO 100
                 IF (ZLEV(I).GT.0.) THEN
                    LWGT(I)=1
                 ELSE
                    LWGT(I)=3
                 END IF
                 LDIG(I)=JJ


 100	      CONTINUE 

              endif  ! openlev

	   ENDIF

C       
C       Else IAUTOC != 1 means: use supplied levels
C       

C       
	ELSE  ! iautoc = 0

C	FIND MIN AND MAX as in lev.F from the Z levels.

	   DMIN=ZLEV(1)
	   IF (neginf) DMIN = ZLEV(2)
	   DMAX=DMIN
           DZMAX = 0.
	   DO 110 I=2,NLEV
	      IF(ZLEV(I).GT.DMAX)DMAX=ZLEV(I)
	      IF(ZLEV(I).LT.DMIN)DMIN=ZLEV(I)
              DZ = ZLEV(I)-ZLEV(I-1)
              IF (DZ .GT. DZMAX) DZMAX = DZ
110        CONTINUE

	   IF (DMIN.EQ.DMAX) THEN
c	      NLEV2=0
	      NLEV2=1
	   ELSE
	   
	   IF (openlev) THEN
	      CALL set_open_levels
	   ELSE
	      NLEV2=NLEV
              IF (neginf) zlev(1) = MIN( zlev(1), zmin-1. )
              IF (posinf) zlev(nlev) = MAX (zlev(nlev), zmax+1. )

C Compute the digits in the contour levels, and line characteristics
C these will not have been set before on, say a previous SHADE plot
C 8/07 ACM. Do this computation only for auto-level setting. If the user
C      has set levels, this is already done (and the values are different).

              IF (iautoc .EQ. 1) THEN
                 znp = dmin
                 zmp = dmax
                 dz = 0
                 IF (set_delta) THEN
                    dz = delta
	            IF (ZNP + DZMAX*(NLEV2-1) .LT. ZMP) NLEV2 = NLEV2 + 1
                 ENDIF

	         IF (ZMP-ZNP .EQ. 0) THEN
                    JJ=NSIGC
                 ELSE
                    JJ=NSIGC-IFIX(ALOG10(ABS(ZMP-ZNP))+.5)
                 ENDIF
                 IF (JJ.LE.0) JJ=-1

                 IF (NLEV2 .GT. ICNLEV) GO TO 5000

	         DO 106 I=1,NLEV2
		    LWGT(I)=2
		    LDIG(I)=-1
		    IF (ABS(ZLEV(I)).LT.1.E-5*DZMAX) GOTO 106
                    IF ( TM_FPEQ( ABS(ZLEV(I)) ,0.) ) GOTO 106
		    IF (ZLEV(I).GT.0.) THEN
		       LWGT(I)=1
		    ELSE
		       LWGT(I)=3
		    END IF
		    LDIG(I)=JJ
 106	         CONTINUE
   
	      ENDIF
	   ENDIF

        ENDIF  ! openlev
	END IF

* Now that the levels are set, create the LEVELS symbols

        CALL setlab_syms (ZLEV, MAX(NLEV2,1) )

	RETURN
 5000   status = ferr_out_of_range
        RETURN

	END
