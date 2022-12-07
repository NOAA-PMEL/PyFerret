
      SUBROUTINE STL2(X, Y, E, M, bad_y, bad_out, U, V, W, K, IP)


* 11/2022 *acm* Code cleanup: 
*              - Rework arithmetic IF statements 
*              - GOTO statements remain. Overall organization of the routine unchanged.

C PIECEWISE LINEAR APPROXIMATIONS OF FEWEST
C LINE SEGMENTS WITHIN GIVEN TOLERANCES.

C X,Y,E AND M CONTAIN INPUT DATA.
C U,V,K AND POSSIBLY W CONTAIN OUTPUT.

C IP IS A PARAMETER DETERMINING THE OPERATION
C  OF THE PROGRAM.

C X AND Y ARE INPUT DATA ARRAYS OF M ELEMENTS
C  X(I),Y(I) CONTAINS THE ITH DATA POINT.

C E MAY BE A SINGLE TOLERANCE OR A TABLE OF
C  TOLERANCES DEPENDING ON THE VALUE OF IP.
C IF E IS AN ARRAY, THEN E(I) IS THE TOLERANCE
C  ASSOCIATED WITH X(I),Y(I) AND E MUST CONTAIN
C  M NONNEGATIVE ELEMENTS.

C U AND V ARE OUTPUT ARRAYS OF K+1 ELEMENTS.
C  U IS A PARTITION OF THE INTERVAL (X(1),X(N))
C  WITH U(1)=X(1) AND U(K+1)=X(N).
C  V(I) IS AN ORDINATE TO BE ASSOCIATED WITH
C  U(I) IN THE APPROXIMATION.  (IF A CONTINUOUS
C  APPROXIMATION IS REQUESTED, THEN V(I) IS
C  'THE' ORDINATE TO BE ASSOCIATED WITH U(I).)

C IF A CONTINUOUS APPROXIMATION IS REQUESTED,
C  THEN W IS NOT USED.  IN THIS CASE THE ITH
C  APPROXIMATING SEGMENT IS THE STRAIGHT LINE
C  FROM U(I),V(I) TO U(I+1),V(I+1).

C IF A CONTINUOUS APPROXIMATION IS NOT
C  REQUESTED, THEN W IS A K-ELEMENT OUTPUT
C  ARRAY.  IN THIS CASE THE ITH APPROXIMATING
C  SEGMENT IS THE STRAIGHT LINE FROM
C  U(I),W(I) TO U(I+1),V(I+1), AND V(1) IS
C  SET EQUAL TO W(1).

C K IS THE NUMBER OF SEGMENTS IN THE PIECE-
C  WISE LINEAR APPROXIMATION GENERATED.  IN
C  CASE OF AN ERROR RETURN, K WILL BE SET TO
C  ZERO.

C THE CONTROL PARAMETER IP IS THE PRODUCT
C  OF THREE INDICATORS I1,I2 AND I3.
C  I1 INDICATES WHETHER OR NOT E IS AN
C     ARRAY OF TOLERANCES.
C     I1 = -1 INDICATES E IS AN ARRAY
C     I1 = +1 INDICATES E IS A SINGLE NUMBER.
C  I2 INDICATES WHETHER OR NOT THE
C     APPROXIMATION IS TO BE RESTRICTED TO
C     THE 'TOLERANCE BAND' ABOUT THE DATA.
C     I2 = 1 INDICATES NO BAND RESTRICTION
C     I2 = 2 INDICATES APPLY THIS RESTRICTION
C     (THE 'TOLERANCE BAND' IS A PIECEWISE
C     LINEAR BAND CENTERED AT THE DATA WHOSE
C     WIDTH IS DETERMINED BY THE TOLERANCES
C     AT THE DATA POINTS.)
C  I3 INDICATES WHETHER OR NOT THE
C     APPROXIMATION MUST BE CONTINUOUS.
C     I3 = 1 INDICATES CONTINUITY NOT REQUIRED
C     I3 = 3 INDICATES CONTINUITY IS REQUIRED

C CALL STL2 (X,Y,E,M,X,Y,E,M,IP) WILL NOT
C  CAUSE PROBLEMS PROVIDED THAT
C  EITHER A CONTINUOUS APPROXIMATION IS
C  REQUESTED, OR E IS A SUFFICIENTLY LARGE
C  ARRAY.

C THE PROGRAM PERFORMS THE FOLLOWING DATA
C  CHECKS.  ARE THE X-VALUES IN INCREASING
C  ORDER.  ARE THE TOLERANCE(S) NONNEGATIVE.
C  IS THE NUMBER OF DATA POINTS GREATER THAN
C  ONE.  IF ANY CHECK FAILS, THE PROGRAM
C  RETURNS WITH K SET EQUAL TO 0.  IN THIS
C  CASE NO FURTHER PROCESSING IS ATTEMPTED.

      IMPLICIT NONE
* Calling arguments
      REAL X(*), Y(*), E(*), U(*), V(*), W(*), bad_y, bad_out
      INTEGER M, K, IP

* Local declarations
      LOGICAL skipbad
      INTEGER N, ITCH, I, J, L, KEEP, INIT, INDC, IGRAZE,
     .        IDIOT, IT, IPIV, KP, ilo, ihi, icount
      REAL EPSLN, SGN, XEYE, YEYE, TEMP1, SMIN, DX, 
     .     SVX, SVY, SVMN, TEMP2, XINIT, YINIT, SLOPE, SMAX, 
     .     SVMX, TEST

* If there is a trivial amount of data in the input array then we are done
      ilo = 0
      ihi = 0
      icount = 0
      DO i = 1, m
         IF (y(i) .NE. bad_y) THEN
	    icount = icount + 1
	    IF (ilo .EQ. 0) ilo = i
	    ihi = i
	 ENDIF
      ENDDO

      IF (icount .LE. 2) THEN
         k = icount
	 IF (ilo .GT. 0) u(1) = ilo
	 IF (ihi .GT. 0) u(icount) = ihi
	 GOTO 600
      ENDIF

      N = M
      ITCH = IP
      J = 1
C ERROR CHECKS
      IF (N.LE.1) GO TO 400
      IF (E(1).LT.0.0) GO TO 400
      DO 10 L=2,N
        IF (X(L-1).GE.X(L)) GO TO 400
        IF (ITCH.GE.0) GO TO 10
        IF (E(L).LT.0.0) GO TO 400
   10 CONTINUE
C INITIALIZATION FOR ENTIRE PROGRAM
      EPSLN = E(1)
      SGN = 1.0
      KEEP = 1
      I = 1

      CALL PIECEWISE3_CHECKGAP (Y, bad_y, x, u, i, j, n, skipbad)
      IF (I.EQ.N) GOTO 260
      IF (skipbad) GOTO 30

      U(i) = X(i)
      J = 2
      INIT = 1
      INDC = 0
      GO TO 30
C INITIALIZATION FOR EACH SEGMENT
   20 CONTINUE
      J = J + 1
      INIT = I
      INDC = 0
      IF (IABS(ITCH).LT.3) KEEP = I
      IF (IABS(IABS(ITCH)-4).NE.2) GO TO 30
C RESTRICTED TO TOLERANCE BAND
      XEYE = U(J-1)
      YEYE = V(J-1)
      TEMP1 = EPSLN
      IF (ITCH.LT.0) TEMP1 = TEMP1 + (SGN*E(I-1)-EPSLN)*(X(I)-U(J-1)
     * )/(X(I)-X(I-1))
      YINIT = YEYE - TEMP1 - TEMP1
      GO TO 40
   30 CONTINUE
C NOT RESTRICTED TO TOLERANCE BAND
      XEYE = X(I)
      YEYE = Y(I) + EPSLN
      YINIT = Y(I) - EPSLN
      IF (IABS(ITCH).EQ.1 .OR. I.EQ.1) GO TO 40
      TEMP1 = EPSLN
      IF (ITCH.LT.0) TEMP1 = SGN*E(I+1)
      SMIN = (Y(I+1)-YEYE-TEMP1)/(X(I+1)-XEYE)
      IF (ITCH.LT.0) TEMP1 = SGN*E(I-1)
      SMAX = (YEYE-Y(I-1)+TEMP1)/(XEYE-X(I-1))

      IF (KEEP.EQ.I-1) GO TO 50
      IT = I - 2
      XINIT = XEYE
      IPIV = I
      IGRAZE = I
      I = I + 1

      CALL PIECEWISE3_CHECKGAP (Y, bad_y, x, u, i, j, n, skipbad)
      IF (skipbad) THEN
         IF (I.EQ.N) GOTO 260
	 GOTO 30
      ENDIF

      GO TO 150
   40 CONTINUE
      IF (XEYE.GE.X(I)) I = I + 1
      IF (ITCH.LT.0) EPSLN = SGN*E(I)
      DX = X(I) - XEYE
      SMAX = (Y(I)+EPSLN-YEYE)/DX
      SMIN = (Y(I)-EPSLN-YEYE)/DX
   50 CONTINUE
      XINIT = XEYE
      IPIV = I
      IGRAZE = I
C DETERMINATION OF INDIVIDUAL SEGMENT
   60 CONTINUE
      IF (I.EQ.N) GO TO 260
      I = I + 1
      
C New i, so does it point to a bad-val?

      CALL PIECEWISE3_CHECKGAP (Y, bad_y, x, u, i, j, n, skipbad)
      IF (skipbad) THEN
         IF (I.EQ.N) GOTO 260
	 GOTO 30
      ENDIF

   70 CONTINUE
 
C TEST FOR NEW *MAX* SLOPE
      DX = X(I) - XEYE
      IF (ITCH.LT.0) EPSLN = SGN*E(I)
      TEMP1 = (Y(I)+EPSLN-YEYE)/DX
      TEST = TEMP1 - SMAX
      IF (SGN.LE.0.0) TEST = -TEST
      IF (TEST .LT. 0.0) GOTO 80 
      IF (TEST .EQ. 0.0) GOTO 90
      IF (TEST .GT. 0.0) GOTO 100
   80 CONTINUE
C TEST FOR END OF CANDIDATE SEGMENT
      TEST = TEMP1 - SMIN
      IF (SGN.LE.0.0) TEST = -TEST
      IF (TEST.LT.0.0) GO TO 210
      SMAX = TEMP1
   90 CONTINUE
C TEST FOR NEW *MIN* SLOPE
      IPIV = I
  100 CONTINUE
      TEMP2 = (Y(I)-EPSLN-YEYE)/DX
      TEST = TEMP2 - SMAX
      IF (SGN.LE.0.0) TEST = -TEST
      IF (TEST .LT. 0.0) GOTO 110
      IF (TEST .EQ. 0.0) GOTO 120
      IF (TEST .GT. 0.0) GOTO 140
  110 CONTINUE
      TEST = SMIN - TEMP2
      IF (SGN.LE.0.0) TEST = -TEST
      IF (TEST .LT. 0.0) GOTO 120
      IF (TEST .EQ. 0.0) GOTO 130
      IF (TEST .GT. 0.0) GOTO 60
  120 CONTINUE
      SMIN = TEMP2
  130 CONTINUE
      IGRAZE = I
      GO TO 60
C CHECK FOR PIVOT AT NEW EYE POINT
  140 CONTINUE
      IF (XEYE.EQ.X(IPIV)) GO TO 220
      IF (ITCH.LT.0) EPSLN = SGN*E(IPIV)
      INDC = 1
      SVX = XEYE
      SVY = YEYE
      SVMN = SMIN
      SVMX = SMAX
      XEYE = X(IPIV)
      YEYE = Y(IPIV) + EPSLN
      SMIN = SMAX
      SMAX = (YINIT-YEYE)/(XINIT-XEYE)
      IF (KEEP.GE.IPIV) GO TO 170
      IT = IPIV - 1
  150 CONTINUE
      TEMP2 = YEYE + EPSLN
      DO 160 L=KEEP,IT
        IF (ITCH.LT.0) TEMP2 = YEYE + SGN*E(L)
        TEMP1 = (Y(L)-TEMP2)/(X(L)-XEYE)
        TEST = TEMP1 - SMAX
        IF (SGN.LE.0.0) TEST = -TEST
        IF (TEST.LT.0.0) SMAX = TEMP1
  160 CONTINUE
  170 CONTINUE
      IF (IPIV.GE.I-1) GO TO 70
      IT = I - 2
      TEMP2 = YEYE - EPSLN
      IDIOT = IPIV
      DO 200 L=IDIOT,IT
        DX = X(L+1) - XEYE
        IF (ITCH.LT.0) TEMP2 = YEYE - SGN*E(L+1)
        TEMP1 = (Y(L+1)-TEMP2)/DX
        TEST = TEMP1 - SMAX
        IF (SGN.LE.0.0) TEST = -TEST
        IF (TEST .LT. 0.0) GOTO 180
        IF (TEST .EQ. 0.0) GOTO 190
        IF (TEST .GT. 0.0) GOTO 200
  180   CONTINUE
        SMAX = TEMP1
  190   CONTINUE
        IPIV = L + 1
  200 CONTINUE
      GO TO 70
C END OF CURRENT SEGMENT
  210 CONTINUE
      TEMP2 = SMIN
      IF (I.EQ.N) GO TO 240
      KEEP = IGRAZE
      GO TO 250
  220 CONTINUE
      TEMP2 = SMAX
      IF (I.EQ.N) GO TO 230
      SGN = -SGN
      EPSLN = -EPSLN
      KEEP = IPIV
      GO TO 250
  230 CONTINUE
      IF (INDC.EQ.0 .OR. XEYE.NE.X(N-1)) GO TO 240
      XEYE = SVX
      YEYE = SVY
      SMIN = SVMN
      SMAX = SVMX
  240 CONTINUE
      U(J) = X(N-1)
      YINIT = Y(N-1)
      GO TO 270
  250 CONTINUE
      IF (IABS(IABS(ITCH)-4).NE.2) GO TO 300
C DETERMINE KNOT ON EDGE OF TOLERANCE BAND
      TEMP1 = 0.0
      IF (ITCH.LT.0) TEMP1 = EPSLN - SGN*E(I-1)
      TEMP1 = (Y(I)-Y(I-1)+TEMP1)/(X(I)-X(I-1))
      
      U(J) = (Y(I)+EPSLN-YEYE-TEMP1*X(I)+TEMP2*XEYE)/(TEMP2-TEMP1)

      GO TO 310
  260 CONTINUE
      U(J) = X(N)
      YINIT = Y(N)
  270 CONTINUE
C CONTINUITY CHECK FOR LAST SEGMENT
      IF (IABS(ITCH).GE.3 .OR. INIT.EQ.1) GO TO 290
      IT = INIT - 1
      SVMX = SMAX + SGN
      TEMP2 = YEYE + EPSLN
      DO 280 L=KP,IT
        IF (ITCH.LT.0) TEMP2 = YEYE + SGN*E(L)
        TEMP1 = (Y(L)-TEMP2)/(X(L)-XEYE)
        TEST = TEMP1 - SVMX
        IF (SGN.LE.0.0) TEST = -TEST
        IF (TEST.LT.0.0) SVMX = TEMP1
  280 CONTINUE
      IF (ABS(SVMX-SMAX+SVMX-SMIN).LE.ABS(SMAX-SMIN)) SMAX = SVMX
  290 CONTINUE
C NEARNESS CHECK FOR LAST SEGMENT
      TEMP2 = SMAX
      TEMP1 = YEYE + SMAX*(U(J)-XEYE)
      TEST = YINIT - TEMP1
      IF (SGN.LT.0.0) TEST = -TEST
      IF (TEST.GT.0.0) GO TO 310
      TEMP2 = SMIN
      TEMP1 = YEYE + SMIN*(U(J)-XEYE)
      TEST = YINIT - TEMP1
      IF (SGN.LT.0.0) TEST = -TEST
      IF (TEST.LT.0.0) GO TO 310
      TEMP2 = (YINIT-YEYE)/(U(J)-XEYE)
      V(J) = YINIT
      GO TO 320
  300 CONTINUE
      IF (IABS(ITCH).GE.3) GO TO 330
      U(J) = 0.5*(X(I)+X(I-1))
  310 CONTINUE
      V(J) = YEYE + TEMP2*(U(J)-XEYE)
  320 CONTINUE
      IF (XEYE.NE.XINIT) GO TO 330
      IF (IABS(ITCH).EQ.2) GO TO 360
      IF (IABS(ITCH).NE.6) GO TO 330
      IF (J.LE.2) GO TO 380
      GO TO 390
  330 CONTINUE
C RECOMPUTATION OF KNOT FOR CONTINUITY
      IF (J.LE.2) GO TO 370
      IF (SLOPE.EQ.TEMP2) GO TO 360
      YINIT = V(J-2)
      IF (IABS(ITCH).LT.3) YINIT = W(J-2)
      TEMP1 = (XEYE*TEMP2-U(J-2)*SLOPE+YINIT-YEYE)/(TEMP2-SLOPE)
      IF (IABS(ITCH).GE.3) GO TO 350
      IF (TEMP1.GT.XINIT) GO TO 360
      TEST = ABS(EPSLN)
      IDIOT = INIT - KP
      DO 340 L=1,IDIOT
        IT = INIT - L
        IF (TEMP1.GE.X(IT)) GO TO 350
        DX = Y(IT) - YEYE - TEMP2*(X(IT)-XEYE)
        IF (ITCH.LT.0) TEST = E(IT)
        IF (ABS(DX).GT.TEST) GO TO 360
  340 CONTINUE
  350 CONTINUE
      U(J-1) = TEMP1
      V(J-1) = YEYE + TEMP2*(U(J-1)-XEYE)
      IF (IABS(ITCH).LT.3) W(J-1) = V(J-1)
      GO TO 390
  360 CONTINUE
      W(J-1) = YEYE + TEMP2*(U(J-1)-XEYE)
      GO TO 390
  370 CONTINUE
      IF (IABS(ITCH).LT.3) GO TO 360
  380 CONTINUE
      V(1) = YEYE + TEMP2*(U(1)-XEYE)
  390 CONTINUE
      SLOPE = TEMP2
      KP = KEEP
      IF (I.LT.N) GO TO 20
      IF (X(N).EQ.U(J)) GO TO 400
      IF (IABS(ITCH).LT.3) W(J) = V(J)
      J = J + 1
      U(J) = X(N)
      V(J) = Y(N)
  400 CONTINUE
      IF (J.GE.2 .AND. IABS(ITCH).LT.3) V(1) = W(1)
      K = J

* Truncate to integer values: indices at which to sample

      DO 500 I = 1, K
         u(i) = INT(u(i))
  500 CONTINUE

  600 CONTINUE
      RETURN
      END

      SUBROUTINE PIECEWISE3_CHECKGAP (Y, bad_y, x, u, i, j, 
     .   n, skipbad)
      REAL Y(*), bad_y, x(*), u(*)
      INTEGER i, j, n
      LOGICAL skipbad

      skipbad = .FALSE.
	 
      IF ( Y(i).EQ.bad_y ) THEN
	 IF (i.GT.2) THEN
	    u(j) = x(i-1)
	       
	    IF ( j .gt. 1 .and. INT(u(j)) .LT. INT(u(j-1)) ) then
	       j = j - 1
	       DO WHILE (INT(u(j)) .LE. INT(u(j-1)) )
		     j = j - 1
	       ENDDO	  
	       u(j) = x(i-1)
	    ENDIF
            IF ( j .gt. 1 .and. INT(u(j)) .EQ. INT(u(j-1)) ) j = j - 1

            j = j + 1
	 ENDIF
         DO WHILE ( Y(i).EQ.bad_y .AND. i.LT.n)
            i = i + 1
         ENDDO
         
         IF (i.GT.1) u(j) = x(i-1)
         IF (i.GE.n) THEN
	    i = n
	    u(j) = x(n)
	 ENDIF
         IF ( j .gt. 1 .and. INT(u(j)) .EQ. INT(u(j-1)) ) j = j - 1

         j = j + 1
         u(j) = x(i)
         IF ( j .gt. 1 .and. INT(u(j)) .EQ. INT(u(j-1)) ) j = j - 1

         j = j + 1
         skipbad = .TRUE.
      ENDIF

      RETURN
      END


      subroutine merge(a,na,b,nb,c,nout,nmax)
c http://en.wikibooks.org/wiki/Algorithm_Implementation/Sorting/Merge_sort#Fortran
! Normal usage: NA+NB = NC
! B overlays C(NA+1:NC)

! modified to return nout instead of assuming treating NC, length of C as known.

      integer nout, nmax
      integer  NA,NB,NC         
      REAL  A(*), B(*), C(*)
 
      integer I,J,K
 
      I = 1
      J = 1
      K = 1
      do while(I .LE. NA .and. J .LE. NB)
         if (A(I) .LE. B(J)) then
            C(K) = A(I)
            I = I+1
	    IF (I.GT.nmax) GOTO 100
         else
            C(K) = B(J)
            J = J+1
         endif
         K = K + 1
	 IF (K.GT.nmax) GOTO 100
      enddo
      do while (I .LE. NA)
         C(K) = A(I)
         I = I + 1
         K = K + 1
	 IF (K.GT.nmax) GOTO 100
      enddo
      do while (J .LE. NB)
         C(K) = B(J)
         J = J+1
         K = K + 1
	 IF (K.GT.nmax) GOTO 100
      enddo

* remove duplicates

 100  CONTINUE
      j = 1
      do i = 2, k
         IF (c(i) .GT. c(i-1)) then
	    j = j + 1
	    IF (j.GT.nmax) THEN
	       j = nmax
	       GOTO 200
	    ENDIF
	    c(j) = c(i)
	 endif
      enddo
 200  nout = j

      return
 
      end ! subroutine merge
