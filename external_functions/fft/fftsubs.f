
      SUBROUTINE FOUR_RE (ND, X, CREAL, CIMAG, WFT)
      REAL X(*), WFT(*) 
      REAL CREAL(*), CIMAG(*), CR, CI, CN, XN
      INTEGER ND, NF, I, J

c   uses NCAR FFT code

c   written by Jim Larsen
c   revised 11/19/96

C  Version by Ansley Manke that replaces complex array C with 
C  real arrays CREAL and CIMAG.

c   Calls: RFFTF

      NF = ND/ 2

c   save X

      DO I = 1, NF
        J = I + I
        CREAL(I) = X(J-1)
        CIMAG(I) = X(J)
      ENDDO

      CALL RFFTF (ND, X, WFT)   !  -> 2 Cn exp(iwt)

c      XN = 1.0/ REAL(ND)	! normalization.  Not using this.
      XN = 1.0
      XN = 0.5			! we're returning half the spectrum. scale by .5

c   restore X and compute complex Cn

      J = 0
      CN = 0.5* X(2) 

      DO I = 1, NF-1
        J = J + 2
        CR = CREAL(I)
        CI = CIMAG(I)
        X(J-1) = CR
        X(J)   = CI
        CREAL(I) = xn* X(J+1)
        CIMAG(I) = xn* X(J+2)
      ENDDO
      CR = CREAL(NF)
      CI = CIMAG(NF)
      X(ND-1) = CR
      X(ND)   = CI
      CREAL(NF) = CN
      CIMAG(NF) = 0.0

      RETURN 
      END
      SUBROUTINE CFFTB (N,C,WSAVE)
C***BEGIN PROLOGUE  CFFTB
c   revised by Jim larsen 4/2/99
C***CATEGORY NO.  D6
C***KEYWORDS FFT,FAST FOURIER TRANSFORM,INVERSE,INVERSE FFT,COMPLEX
C***DATE WRITTEN  FEBRUARY 1978
C***AUTHOR  SWARZTRAUBER P.N. (NCAR)
C***PURPOSE
C INVERSE FFT OF A COMPLEX PERIODIC SEQUENCE
C***DESCRIPTION
C
C     SUBROUTINE CFFTB COMPUTES THE BACKWARD COMPLEX DISCRETE FOURIER
C     TRANSFORM (THE FOURIER SYNTHESIS). EQUIVALENTLY , CFFTB COMPUTES
C     A COMPLEX PERIODIC SEQUENCE FROM ITS FOURIER COEFFICIENTS.
C     THE TRANSFORM IS DEFINED BELOW AT OUTPUT PARAMETER C.
C
C     A CALL OF CFFTF FOLLOWED BY A CALL OF CFFTB WILL MULTIPLY THE
C     SEQUENCE BY N.
C
C     THE ARRAY WSAVE WHICH IS USED BY SUBROUTINE CFFTB MUST BE
C     INITIALIZED BY CALLING SUBROUTINE CFFTI(N,WSAVE).
C
C     INPUT PARAMETERS
C
C
C     N      THE LENGTH OF THE COMPLEX SEQUENCE C. THE METHOD IS
C            MORE EFFICIENT WHEN N IS THE PRODUCT OF SMALL PRIMES.
C
C     C      A COMPLEX ARRAY OF LENGTH N WHICH CONTAINS THE SEQUENCE
C
C     WSAVE   A REAL WORK ARRAY WHICH MUST BE DIMENSIONED AT LEAST 4N+1
C             IN THE PROGRAM THAT CALLS CFFTB. THE WSAVE ARRAY MUST BE
C             INITIALIZED BY CALLING SUBROUTINE CFFTI(N,WSAVE) AND A
C             DIFFERENT WSAVE ARRAY MUST BE USED FOR EACH DIFFERENT
C             VALUE OF N. THIS INITIALIZATION DOES NOT HAVE TO BE
C             REPEATED SO LONG AS N REMAINS UNCHANGED THUS SUBSEQUENT
C             TRANSFORMS CAN BE OBTAINED FASTER THAN THE FIRST.
C             THE SAME WSAVE ARRAY CAN BE USED BY CFFTF AND CFFTB.
C
C     OUTPUT PARAMETERS
C
C     C      FOR J=1,...,N
C
C                C(J)=THE SUM FROM K=1,...,N OF
C
C                      C(K)*EXP(I*J*K*2*PI/N)
C
C                            WHERE I=SQRT(-1)
C
C     WSAVE   CONTAINS INITIALIZATION CALCULATIONS WHICH MUST NOT BE
C             DESTROYED BETWEEN CALLS OF SUBROUTINE CFFTF OR CFFTB
C***REFERENCES
C***ROUTINES CALLED  CFFTB1
C***END PROLOGUE  CFFTB
      DIMENSION       C(*)       ,WSAVE(*)
C***FIRST EXECUTABLE STATEMENT  CFFTB
      IF (N .EQ. 1) RETURN
      N2 = N+N
      IW1 = 1+N2
      IW2 = IW1+N2
      CALL CFFTB1 (N,C,WSAVE,WSAVE(IW1),WSAVE(IW2))
      RETURN
      END
      SUBROUTINE CFFTB1 (N,C,CH,WA,IFAC)
C***BEGIN PROLOGUE  CFFTB1
c   revised by Jim larsen 4/2/99
C***REFER TO CFFTB
C***ROUTINES CALLED  PASSB,PASSB5,PASSB3,PASSB2,PASSB4
C***END PROLOGUE  CFFTB1
      DIMENSION       CH(*)      ,C(*)       ,WA(*)      ,IFAC(*)
C***FIRST EXECUTABLE STATEMENT  CFFTB1
      NF = IFAC(2)
      NA = 0
      L1 = 1
      IW = 1
      DO K1=3,NF+2
        NAC = 1
        IP = IFAC(K1)
        L2 = IP*L1
        IDO = N/L2
        IDOT = IDO+IDO
        IDL1 = IDOT*L1
        IF (IP .EQ. 2) THEN
          IF (NA .EQ. 0) THEN
            CALL PASSB2 (IDOT,L1,C,CH,WA(IW))
          ELSE
            CALL PASSB2 (IDOT,L1,CH,C,WA(IW))
          ENDIF
        ELSE IF (IP .EQ. 3) THEN
          IX2 = IW+IDOT
          IF (NA .EQ. 0) THEN
            CALL PASSB3 (IDOT,L1,C,CH,WA(IW),WA(IX2))
          ELSE
            CALL PASSB3 (IDOT,L1,CH,C,WA(IW),WA(IX2))
          ENDIF
        ELSE IF (IP .EQ. 4) THEN
          IX2 = IW+IDOT
          IX3 = IX2+IDOT
          IF (NA .EQ. 0) THEN
            CALL PASSB4 (IDOT,L1,C,CH,WA(IW),WA(IX2),WA(IX3))
          ELSE
            CALL PASSB4 (IDOT,L1,CH,C,WA(IW),WA(IX2),WA(IX3))
          ENDIF
        ELSE IF (IP .EQ. 5) THEN
          IX2 = IW+IDOT
          IX3 = IX2+IDOT
          IX4 = IX3+IDOT
          IF (NA .EQ. 0) THEN
            CALL PASSB5 (IDOT,L1,C,CH,WA(IW),WA(IX2),WA(IX3),WA(IX4))
          ELSE
            CALL PASSB5 (IDOT,L1,CH,C,WA(IW),WA(IX2),WA(IX3),WA(IX4))
          ENDIF
        ELSE
          IF (NA .EQ. 0) THEN
            CALL PASSB (NAC,IDOT,IP,L1,IDL1,C,C,C,CH,CH,WA(IW))
          ELSE
            CALL PASSB (NAC,IDOT,IP,L1,IDL1,CH,CH,CH,C,C,WA(IW))
          ENDIF
        ENDIF
        IF (NAC .NE. 0) NA = 1-NA
        L1 = L2
        IW = IW+(IP-1)*IDOT
      ENDDO
      IF (NA .EQ. 0) RETURN
      N2 = N+N
      DO I=1,N2
        C(I) = CH(I)
      ENDDO
      RETURN
      END
      SUBROUTINE CFFTF (N,C,WSAVE)
C***BEGIN PROLOGUE  CFFTF
c   revised by Jim larsen 4/2/99
C***CATEGORY NO.  D6
C***KEYWORDS FFT,FAST FOURIER TRANSFORM,COMPLEX
C***DATE WRITTEN  FEBRUARY 1978
C***AUTHOR  SWARZTRAUBER P.N. (NCAR)
C***PURPOSE
C  FORWARD FFT OF A COMPLEX PERIODIC SEQUENCE
C***DESCRIPTION
C
C     SUBROUTINE CFFTF COMPUTES THE FORWARD COMPLEX DISCRETE FOURIER
C     TRANSFORM (THE FOURIER ANALYSIS). EQUIVALENTLY , CFFTF COMPUTES
C     THE FOURIER COEFFICIENTS OF A COMPLEX PERIODIC SEQUENCE.
C     THE TRANSFORM IS DEFINED BELOW AT OUTPUT PARAMETER C.
C
C     THE TRANSFORM IS NOT NORMALIZED. TO OBTAIN A NORMALIZED TRANSFORM
C     THE OUTPUT MUST BE DIVIDED BY N. OTHERWISE A CALL OF CFFTF
C     FOLLOWED BY A CALL OF CFFTB WILL MULTIPLY THE SEQUENCE BY N.
C
C     THE ARRAY WSAVE WHICH IS USED BY SUBROUTINE CFFTF MUST BE
C     INITIALIZED BY CALLING SUBROUTINE CFFTI(N,WSAVE).
C
C     INPUT PARAMETERS
C
C
C     N      THE LENGTH OF THE COMPLEX SEQUENCE C. THE METHOD IS
C            MORE EFFICIENT WHEN N IS THE PRODUCT OF SMALL PRIMES. N
C
C     C      A COMPLEX ARRAY OF LENGTH N WHICH CONTAINS THE SEQUENCE
C
C     WSAVE   A REAL WORK ARRAY WHICH MUST BE DIMENSIONED AT LEAST 4N+1
C             IN THE PROGRAM THAT CALLS CFFTF. THE WSAVE ARRAY MUST BE
C             INITIALIZED BY CALLING SUBROUTINE CFFTI(N,WSAVE) AND A
C             DIFFERENT WSAVE ARRAY MUST BE USED FOR EACH DIFFERENT
C             VALUE OF N. THIS INITIALIZATION DOES NOT HAVE TO BE
C             REPEATED SO LONG AS N REMAINS UNCHANGED THUS SUBSEQUENT
C             TRANSFORMS CAN BE OBTAINED FASTER THAN THE FIRST.
C             THE SAME WSAVE ARRAY CAN BE USED BY CFFTF AND CFFTB.
C
C     OUTPUT PARAMETERS
C
C     C      FOR J=1,...,N
C
C                C(J)=THE SUM FROM K=1,...,N OF
C
C                      C(K)*EXP(-I*J*K*2*PI/N)
C
C                            WHERE I=SQRT(-1)
C
C     WSAVE   CONTAINS INITIALIZATION CALCULATIONS WHICH MUST NOT BE
C             DESTROYED BETWEEN CALLS OF SUBROUTINE CFFTF OR CFFTB
C
C***REFERENCES
C***ROUTINES CALLED  CFFTF1
C***END PROLOGUE  CFFTF
      DIMENSION       C(*)       ,WSAVE(*)
C***FIRST EXECUTABLE STATEMENT  CFFTF
      IF (N .EQ. 1) RETURN
      N2 = N+N
      IW1 = 1+N2
      IW2 = IW1+N2
      CALL CFFTF1 (N,C,WSAVE,WSAVE(IW1),WSAVE(IW2))
      RETURN
      END
      SUBROUTINE CFFTF1 (N,C,CH,WA,IFAC)
C***BEGIN PROLOGUE  CFFTF1
c   revised by Jim larsen 4/6/99
C***REFER TO CFFTF
C***ROUTINES CALLED  PASSF,PASSF5,PASSF3,PASSF2,PASSF4
C***END PROLOGUE  CFFTF1
      DIMENSION       CH(*)      ,C(*)       ,WA(*)      ,IFAC(*)
C***FIRST EXECUTABLE STATEMENT  CFFTF1
      NF = IFAC(2)
      NA = 0
      L1 = 1
      IW = 1
      DO K1=3,NF+2
        NAC = 1
        IP = IFAC(K1)
        L2 = IP*L1
        IDO = N/L2
        IDOT = IDO+IDO
        IDL1 = IDOT*L1
        IF (IP .EQ. 2) THEN
          IF (NA .EQ. 0) THEN
            CALL PASSF2 (IDOT,L1,C,CH,WA(IW))
          ELSE
            CALL PASSF2 (IDOT,L1,CH,C,WA(IW))
          ENDIF
        ELSE IF (IP .EQ. 3) THEN
          IX2 = IW+IDOT
          IF (NA .EQ. 0) THEN
            CALL PASSF3 (IDOT,L1,C,CH,WA(IW),WA(IX2))
          ELSE
            CALL PASSF3 (IDOT,L1,CH,C,WA(IW),WA(IX2))
          ENDIF
        ELSE IF (IP .EQ. 4) THEN
          IX2 = IW+IDOT
          IX3 = IX2+IDOT
          IF (NA .EQ. 0) THEN
            CALL PASSF4 (IDOT,L1,C,CH,WA(IW),WA(IX2),WA(IX3))
          ELSE
            CALL PASSF4 (IDOT,L1,CH,C,WA(IW),WA(IX2),WA(IX3))
          ENDIF
        ELSE IF (IP .EQ. 5) THEN
          IX2 = IW+IDOT
          IX3 = IX2+IDOT
          IX4 = IX3+IDOT
          IF (NA .EQ. 0) THEN
            CALL PASSF5 (IDOT,L1,C,CH,WA(IW),WA(IX2),WA(IX3),WA(IX4))
          ELSE
            CALL PASSF5 (IDOT,L1,CH,C,WA(IW),WA(IX2),WA(IX3),WA(IX4))
          ENDIF
        ELSE
          IF (NA .EQ. 0) THEN
            CALL PASSF (NAC,IDOT,IP,L1,IDL1,C,C,C,CH,CH,WA(IW))
          ELSE
            CALL PASSF (NAC,IDOT,IP,L1,IDL1,CH,CH,CH,C,C,WA(IW))
          ENDIF
        ENDIF
        IF (NAC .NE. 0) NA = 1-NA
        L1 = L2
        IW = IW+(IP-1)*IDOT
      ENDDO
      IF (NA .EQ. 0) RETURN
      N2 = N+N
      DO I=1,N2
        C(I) = CH(I)
      ENDDO
      RETURN
      END
      SUBROUTINE CFFTI (N,WSAVE)
C***BEGIN PROLOGUE  CFFTI
c   revised by Jim larsen 4/2/99
C***CATEGORY NO.  D6
C***KEYWORDS FFT,FAST FOURIER TRANSFORM,FOURIER TRANSFORM,COMPLEX
C***DATE WRITTEN  FEBRUARY 1978
C***AUTHOR  SWARZTRAUBER P.N. (NCAR)
C***PURPOSE
C   INITIALIZE FOR CFFTF AND CFFTB
C***DESCRIPTION
C
C     SUBROUTINE CFFTI INITIALIZES THE ARRAY WSAVE WHICH IS USED IN
C     BOTH CFFTF AND CFFTB. THE PRIME FACTORIZATION OF N TOGETHER WITH
C     A TABULATION OF THE TRIGONOMETRIC FUNCTIONS ARE COMPUTED AND
C     STORED IN WSAVE.
C
C     INPUT PARAMETER
C
C     N       THE LENGTH OF THE SEQUENCE TO BE TRANSFORMED
C
C     OUTPUT PARAMETER
C
C     WSAVE   A WORK ARRAY WHICH MUST BE DIMENSIONED AT LEAST 4*N+15
C             THE SAME WORK ARRAY CAN BE USED FOR BOTH CFFTF AND CFFTB
C             AS LONG AS N REMAINS UNCHANGED. DIFFERENT WSAVE ARRAYS
C             ARE REQUIRED FOR DIFFERENT VALUES OF N. THE CONTENTS OF
C             WSAVE MUST NOT BE CHANGED BETWEEN CALLS OF CFFTF OR CFFTB
C
C***REFERENCES
C***ROUTINES CALLED  CFFTI1
C***END PROLOGUE  CFFTI
      DIMENSION       WSAVE(*)
C***FIRST EXECUTABLE STATEMENT  CFFTI
      IF (N .EQ. 1) RETURN
      N2 = N+N
      IW1 = 1+N2
      IW2 = IW1+N2
      CALL CFFTI1 (N,WSAVE(IW1),WSAVE(IW2))
      RETURN
      END
      SUBROUTINE CFFTI1 (N,WA,IFAC)
C***BEGIN PROLOGUE  CFFTI1
c   revised by Jim larsen 4/7/99
C***REFER TO CFFTI
C***ROUTINES CALLED  (NONE)
C***END PROLOGUE  CFFTI1
      DIMENSION       WA(*)      ,IFAC(*)    ,NTRYH(4)
      DATA NTRYH(1),NTRYH(2),NTRYH(3),NTRYH(4)/3,4,2,5/
C***FIRST EXECUTABLE STATEMENT  CFFTI1
      TPI = 2.0* 3.141592654
      NL = N
      NF = 0
      J  = 1
      NTRY = 3
      DO WHILE (NL .NE. 1)
        NQ = NL/NTRY
        NR = NL-NTRY*NQ
        DO WHILE (NR .NE. 0) 
          J = J + 1
          IF (J .LE. 4) THEN
            NTRY = NTRYH(J)
          ELSE
            NTRY = NTRY+2
          ENDIF
          NQ = NL/NTRY
          NR = NL-NTRY*NQ
        ENDDO
        IF (NR .EQ. 0) THEN
          NF = NF + 1
          IFAC(NF+2) = NTRY
          NL = NQ
          IF (NTRY .EQ. 2 .AND. NF .NE. 1) THEN
            DO IB = NF+1,3,-1
              IFAC(IB+1) = IFAC(IB)
            ENDDO
            IFAC(3) = 2
          ENDIF
        ENDIF
      ENDDO
      IFAC(1) = N
      IFAC(2) = NF
      ARGH = TPI/REAL(N)
      I = 2
      L1 = 1
      DO K1=3,NF+2
        IP = IFAC(K1)
        L2 = L1*IP
        IDO = N/L2
        IDOT = IDO+IDO+2
        IPM = IP-1
        WLDC = 1.
        WLDS = 0.
        ARG = REAL(L1)*ARGH
        DL1C = COS(ARG)
        DL1S = SIN(ARG)
        DO J=1,IPM
          I1 = I
          WA(I-1) = 1.
          WA(I) = 0.
          WLDCH = WLDC
          WLDC = DL1C* WLDC  - DL1S* WLDS
          WLDS = DL1S* WLDCH + DL1C* WLDS
          DO II=4,IDOT,2
            W1 = WA(I-1)
            W2 = WA(I)
            I  = I+2
            WA(I-1) = WLDC* W1 - WLDS* W2
            WA(I)   = WLDS* W1 + WLDC* W2
          ENDDO
          IF (IP .GT. 5) THEN
            WA(I1-1) = WA(I-1)
            WA(I1)   = WA(I)
          ENDIF
        ENDDO
        L1 = L2
      ENDDO
      RETURN
      END
      SUBROUTINE COSWINDOW (W, ND, NP)

c   Cosine tapers end values for NP/100 percent of ND values for start and end
c   values

c   written by Jim Larsen
c   revised 1/24/96

c   Calls: no subroutines

      DIMENSION W(*)
      REAL*8 PI, XM, CC, S1, S2, S3

      DO I = 1, ND
        W(I) = 1.0
      ENDDO

      PI = 3.1415926535897932384

      MD = 0.01* NP* ND + 0.5
      XM = PI/ (MD* 2)

      W(1)  = 0.0
      W(ND) = 0.0

      CC = 2.D0* COS (XM)
      S1 = 0.D0
      S2 = SIN (XM)

      J = ND
      DO I = 2, MD
        J = J - 1
        W(I) = S2
        W(J) = S2 
        S3 = CC* S2 - S1
        S1 = S2
        S2 = S3
      ENDDO

      S = 0.0
      DO I = 1, ND
        S = S + W(I)**2
      ENDDO
      S = SQRT (ND/ S)
      DO I = 1, ND
        W(I) = S* W(I)
      ENDDO

      RETURN
      END
*DECK EZFFTB
      SUBROUTINE EZFFTB (N,R,AZERO,A,B,WSAVE)
C***BEGIN PROLOGUE  EZFFTB
C***REVISION DATE  811015   (YYMMDD)
C***CATEGORY NO.  D6
C***KEYWORDS FFT,FAST FOURIER TRANSFORM,INVERSE,INVERSE FFT
C***DATE WRITTEN  FEBRUARY 1978
C***AUTHOR  SWARZTRAUBER P.N. (NCAR)
C***PURPOSE
C SIMPLIFIED VERSION OF INVERSE FFT OF A REAL PERIODIC SEQUENCE
C***DESCRIPTION
C     *****************************************************************
C
C     SUBROUTINE EZFFTB(N,R,AZERO,A,B,WSAVE)
C
C     *****************************************************************
C
C     SUBROUTINE EZFFTB COMPUTES A REAL PERODIC SEQUENCE FROM ITS
C     FOURIER COEFFICIENTS (FOURIER SYNTHESIS). THE TRANSFORM IS
C     DEFINED BELOW AT OUTPUT PARAMETER R. EZFFTB IS A SIMPLIFIED
C     VERSION OF RFFTB. IT IS NOT AS FAST AS RFFTB SINCE SCALING AND
C     INITIALIZATION ARE COMPUTED FOR EACH TRANSFORM. THE REPEATED
C     INITIALIZATION CAN BE SUPPRESSED BY REMOVEING THE STATMENT
C     ( CALL EZFFTI(N,WSAVE) ) FROM BOTH EZFFTF AND EZFFTB AND INSERTING
C     IT AT THE APPROPRIATE PLACE IN YOUR PROGRAM.
C
C     INPUT PARAMETERS
C
C     N       THE LENGTH OF THE ARRAY R. EZFFTB IS ABOUT TWICE AS FAST
C             FOR EVEN N AS IT IS FOR ODD N. ALSO EZFFTB IS MORE
C             EFFICIENT WHEN N IS A PRODUCT OF SMALL PRIMES.
C
C     AZERO   THE CONSTANT FOURIER COEFFICIENT
C
C     A,B     ARRAYS WHICH CONTAIN THE REMAINING FOURIER COEFFICIENTS
C             THESE ARRAYS ARE NOT DESTROYED.
C
C             THE LENGTH OF THESE ARRAYS DEPENDS ON WHETHER N IS EVEN OR
C             ODD.
C
C             IF N IS EVEN N/2    LOCATIONS ARE REQUIRED
C             IF N IS ODD (N-1)/2 LOCATIONS ARE REQUIRED
C
C     WSAVE   A WORK ARRAY WHOSE LENGTH DEPENDS ON WHETHER N IS EVEN OR
C             ODD.
C
C                  IF N IS EVEN 3.5*N+15 LOCATIONS ARE REQUIRED
C                  IF N IS ODD  6*N+15   LOCATIONS ARE REQUIRED
C
C
C     OUTPUT PARAMETERS
C
C     R       IF N IS EVEN DEFINE KMAX=N/2
C             IF N IS ODD  DEFINE KMAX=(N-1)/2
C
C             THEN FOR I=1,...,N
C
C                  R(I)=AZERO PLUS THE SUM FROM K=1 TO K=KMAX OF
C
C                  A(K)*COS(K*(I-1)*2*PI/N)+B(K)*SIN(K*(I-1)*2*PI/N)
C
C     ********************* COMPLEX NOTATION **************************
C
C             FOR J=1,...,N
C
C             R(J) EQUALS THE SUM FROM K=-KMAX TO K=KMAX OF
C
C                  C(K)*EXP(I*K*(J-1)*2*PI/N)
C
C             WHERE
C
C                  C(K) = .5*CMPLX(A(K),-B(K))   FOR K=1,...,KMAX
C
C                  C(-K) = CONJG(C(K))
C
C                  C(0) = AZERO
C
C                       AND I=SQRT(-1)
C
C     *************** AMPLITUDE - PHASE NOTATION **********************
C
C             FOR I=1,...,N
C
C             R(I) EQUALS AZERO PLUS THE SUM FROM K=1 TO K=KMAX OF
C
C                  ALPHA(K)*COS(K*(I-1)*2*PI/N+BETA(K))
C
C             WHERE
C
C                  ALPHA(K) = SQRT(A(K)*A(K)+B(K)*B(K))
C
C                  COS(BETA(K))=A(K)/ALPHA(K)
C
C                  SIN(BETA(K))=-B(K)/ALPHA(K)
C
C***REFERENCES
C***ROUTINES CALLED  CFFTB,RFFTB,EZFFTI
C***END PROLOGUE  EZFFTB
      DIMENSION       R(*)       ,A(*)       ,B(*)       ,WSAVE(*)
C***FIRST EXECUTABLE STATEMENT  EZFFTB
      IF (N .GT. 1) GO TO 101
      R(1) = AZERO
      RETURN
  101 IF (N .GT. 2) GO TO 102
      R(1) = AZERO+A(1)
      R(2) = AZERO-A(1)
      RETURN
  102 NS2 = N/2
C
C     TO SUPRESS REPEATED INITIALIZATION REMOVE THE FOLLOWING STATMENT
C     ( CALL EZFFTI(N,WSAVE)) FROM BOTH THIS PROGRAM AND EZFFTF AND
C     INSERT IT AT THE APPROPRIATE PLACE IN YOUR PROGRAM.
C
      CALL EZFFTI (N,WSAVE)
C
      MODN = N-NS2-NS2
      IF (MODN .NE. 0) GO TO 104
      IW1 = N+1
      NS2M = NS2-1
      DO 103 I=1,NS2M
         R(2*I+1) = .5*A(I)
         R(2*I+2) = .5*B(I)
  103 CONTINUE
      R(1) = AZERO
      R(2) = A(NS2)
      CALL RFFTB (N,R,WSAVE(IW1))
      RETURN
  104 IW1 = N+N+1
      NM1S2 = (N-1)/2
      DO 105 I=1,NM1S2
         IC = N-I
         WSAVE(2*I+1) = A(I)
         WSAVE(2*I+2) = -B(I)
         WSAVE(2*IC+1) = A(I)
         WSAVE(2*IC+2) = B(I)
  105 CONTINUE
      WSAVE(1) = AZERO+AZERO
      WSAVE(2) = 0.
      CALL CFFTB (N,WSAVE,WSAVE(IW1))
      DO 106 I=1,N
         R(I) = .5*WSAVE(2*I-1)
  106 CONTINUE
      RETURN
      END
*DECK EZFFTF
      SUBROUTINE EZFFTF (N,R,AZERO,A,B,WSAVE)
C***BEGIN PROLOGUE  EZFFTF
C***REVISION DATE  811015   (YYMMDD)
C***CATEGORY NO.  D6
C***KEYWORDS FFT,FAST FOURIER TRANSFORM,FOURIER TRANSFORM
C***DATE WRITTEN  FEBRUARY 1978
C***AUTHOR  SWARZTRAUBER P.N. (NCAR)
C***PURPOSE
C  FORWARD FFT OF A REAL PERIODIC SEQUENCE (SIMPLIFED VERSION)
C***DESCRIPTION
C     *****************************************************************
C
C     SUBROUTINE EZFFTF(N,R,AZERO,A,B,WSAVE)
C
C     *****************************************************************
C
C     SUBROUTINE EZFFTF COMPUTES THE FOURIER COEFFICIENTS OF A REAL
C     PERODIC SEQUENCE (FOURIER ANALYSIS). THE TRANSFORM IS DEFINED
C     BELOW AT OUTPUT PARAMETERS AZERO,A AND B. EZFFTF IS A SIMPLIFIED
C     VERSION OF RFFTF. IT IS NOT AS FAST AS RFFTF SINCE SCALING
C     AND INITIALIZATION ARE COMPUTED FOR EACH TRANSFORM. THE REPEATED
C     INITIALIZATION CAN BE SUPPRESSED BY REMOVEING THE STATMENT
C     ( CALL EZFFTI(N,WSAVE) ) FROM BOTH EZFFTF AND EZFFTB AND INSERTING
C     IT AT THE APPROPRIATE PLACE IN YOUR PROGRAM.
C
C     INPUT PARAMETERS
C
C     N       THE LENGTH OF THE ARRAY R. EZFFTF IS ABOUT TWICE AS FAST
C             FOR EVEN N AS IT IS FOR ODD N. ALSO EZFFTF IS MORE
C             EFFICIENT WHEN N IS A PRODUCT OF SMALL PRIMES.
C
C     R       A REAL ARRAY OF LENGTH N WHICH CONTAINS THE SEQUENCE
C             TO BE TRANSFORMED. R IS NOT DESTROYED.
C
C     WSAVE   A WORK ARRAY WHOSE LENGTH DEPENDS ON WHETHER N IS EVEN OR
C             ODD.
C
C                  IF N IS EVEN 3.5*N+15 LOCATIONS ARE REQUIRED
C                  IF N IS ODD  6*N+15   LOCATIONS ARE REQUIRED
C
C
C     OUTPUT PARAMETERS
C
C     AZERO   THE SUM FROM I=1 TO I=N OF R(I)/N
C
C     A,B     FOR N EVEN B(N/2)=0. AND A(N/2) IS THE SUM FROM I=1 TO
C             I=N OF (-1)**(I-1)*R(I)/N
C
C             FOR N EVEN DEFINE KMAX=N/2-1
C             FOR N ODD  DEFINE KMAX=(N-1)/2
C
C             THEN FOR  K=1,...,KMAX
C
C                  A(K) EQUALS THE SUM FROM I=1 TO I=N OF
C
C                       2./N*R(I)*COS(K*(I-1)*2*PI/N)
C
C                  B(K) EQUALS THE SUM FROM I=1 TO I=N OF
C
C                       2./N*R(I)*SIN(K*(I-1)*2*PI/N)
C
C
C***REFERENCES
C***ROUTINES CALLED  CFFTF,RFFTF,EZFFTI
C***END PROLOGUE  EZFFTF
      DIMENSION       R(*)       ,A(*)       ,B(*)       ,WSAVE(*)
C***FIRST EXECUTABLE STATEMENT  EZFFTF
      IF (N .GT. 1) GO TO 101
      AZERO = R(1)
      RETURN
  101 IF (N .GT. 2) GO TO 102
      AZERO = .5*(R(1)+R(2))
      A(1) = .5*(R(1)-R(2))
      RETURN
  102 NS2 = N/2
C
C     TO SUPRESS REPEATED INITIALIZATION REMOVE THE FOLLOWING STATMENT
C     ( CALL EZFFTI(N,WSAVE)) FROM BOTH THIS PROGRAM AND EZFFTB AND
C     INSERT IT AT THE APPROPRIATE PLACE IN YOUR PROGRAM.
C
      CALL EZFFTI (N,WSAVE)
C
      MODN = N-NS2-NS2
      IF (MODN .NE. 0) GO TO 105
      IW1 = N+1
      DO 103 I=1,N
         WSAVE(I) = R(I)
  103 CONTINUE
      CALL RFFTF (N,WSAVE,WSAVE(IW1))
      CF = 1./FLOAT(N)
      AZERO = .5*CF*WSAVE(1)
      A(NS2) = .5*CF*WSAVE(2)
      B(NS2) = 0.
      NS2M = NS2-1
      DO 104 I=1,NS2M
         A(I) = CF*WSAVE(2*I+1)
         B(I) = CF*WSAVE(2*I+2)
  104 CONTINUE
      RETURN
  105 IW1 = N+N+1
      DO 106 I=1,N
         WSAVE(2*I-1) = R(I)
         WSAVE(2*I) = 0.
  106 CONTINUE
      CALL CFFTF (N,WSAVE,WSAVE(IW1))
      CF = 2./FLOAT(N)
      AZERO = .5*CF*WSAVE(1)
      NM1S2 = (N-1)/2
      DO 107 I=1,NM1S2
         A(I) = CF*WSAVE(2*I+1)
         B(I) = -CF*WSAVE(2*I+2)
  107 CONTINUE
      RETURN
      END
*DECK EZFFTI
      SUBROUTINE EZFFTI(N,WSAVE)
C***BEGIN PROLOGUE  EZFFTI
C***REVISION DATE  811015   (YYMMDD)
C***CATEGORY NO.  D6
C***KEYWORDS FFT,FAST FOURIER TRANSFORM,FOURIER TRANSFORM
C***DATE WRITTEN  FEBRUARY 1978
C***AUTHOR  SWARZTRAUBER P.N. (NCAR)
C***PURPOSE
C   INITIALIZE FOR EZFFTF AND EZFFTB
C***DESCRIPTION
C     *****************************************************************
C
C     SUBROUTINE EZFFTI(N,WSAVE)
C
C     *****************************************************************
C
C     SUBROUTINE EZFFTI INITIALIZES THE ARRAY WSAVE WHICH IS USED IN
C     BOTH EZFFTF AND EZFFTB. THE PRIME FACTORIZATION OF N TOGETHER WITH
C     A TABULATION OF THE TRIGONOMETRIC FUNCTIONS ARE COMPUTED AND
C     STORED IN WSAVE.
C
C     INPUT PARAMETER
C
C     N       THE LENGTH OF THE SEQUENCE TO BE TRANSFORMED.
C
C     OUTPUT PARAMETER
C
C     WSAVE   A WORK ARRAY WHICH MUST BE DIMENSIONED AT LEAST 4*N+15
C             THE SAME WORK ARRAY CAN BE USED FOR BOTH EZFFTF AND EZFFTB
C             AS LONG AS N REMAINS UNCHANGED. DIFFERENT WSAVE ARRAYS
C             ARE REQUIRED FOR DIFFERENT VALUES OF N. THE CONTENTS OF
C             WSAVE MUST NOT BE CHANGED BETWEEN CALLS OF EZFFTF OR EZFFT
C
C***REFERENCES
C***ROUTINES CALLED  CFFTI,RFFTI
C***END PROLOGUE  EZFFTI
      DIMENSION       WSAVE(*)
C***FIRST EXECUTABLE STATEMENT  EZFFTI
      IF (N .LE. 2) RETURN
      NS2 = N/2
      MODN = N-NS2-NS2
      IF (MODN .NE. 0) GO TO 101
      IW1 = N+1
      CALL RFFTI (N,WSAVE(IW1))
      RETURN
  101 IW1 = N+N+1
      CALL CFFTI (N,WSAVE(IW1))
      RETURN
      END
*DECK FFTDOC
      SUBROUTINE FFTDOC
C***BEGIN PROLOGUE  FFTDOC
C***DATE WRITTEN   780201   (YYMMDD)
C***REVISION DATE  811015   (YYMMDD)
C***PURPOSE  DOCUMENTATION FOR FFT PACKAGE
C***DESCRIPTION
C     * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
C
C                       VERSION 2  FEBRUARY 1978
C
C          A PACKAGE OF FORTRAN SUBPROGRAMS FOR THE FAST FOURIER
C           TRANSFORM OF PERIODIC AND OTHER SYMMETRIC SEQUENCES
C
C                              BY
C
C                       PAUL N SWARZTRAUBER
C
C       NATIONAL CENTER FOR ATMOSPHERIC RESEARCH  BOULDER,COLORADO 8030
C
C        WHICH IS SPONSORED BY THE NATIONAL SCIENCE FOUNDATION
C
C     * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
C
C
C     THIS PACKAGE CONSISTS OF PROGRAMS WHICH PERFORM FAST FOURIER
C     TRANSFORMS FOR BOTH COMPLEX AND REAL PERIODIC SEQUENCES AND
C     CERTAIN OTHER SYMMETRIC SEQUENCES THAT ARE LISTED BELOW.
C
C     1.   RFFTI     INITIALIZE  RFFTF AND RFFTB
C     2.   RFFTF     FORWARD TRANSFORM OF A REAL PERIODIC SEQUENCE
C     3.   RFFTB     BACKWARD TRANSFORM OF A REAL COEFFICIENT ARRAY
C
C     4.   EZFFTF    A SIMPLIFIED REAL PERIODIC FORWARD TRANSFORM
C     5.   EZFFTB    A SIMPLIFIED REAL PERIODIC BACKWARD TRANSFORM
C
C     6.   SINTI     INITIALIZE SINT
C     7.   SINT      SINE TRANSFORM OF A REAL ODD SEQUENCE
C
C     8.   COSTI     INITIALIZE COST
C     9.   COST      COSINE TRANSFORM OF A REAL EVEN SEQUENCE
C
C     10.  SINQI     INITIALIZE SINQF AND SINQB
C     11.  SINQF     FORWARD SINE TRANSFORM WITH ODD WAVE NUMBERS
C     12.  SINQB     UNNORMALIZED INVERSE OF SINQF
C
C     13.  COSQI     INITIALIZE COSQF AND COSQB
C     14.  COSQF     FORWARD COSINE TRANSFORM WITH ODD WAVE NUMBERS
C     15.  COSQB     UNNORMALIZED INVERSE OF COSQF
C
C     16.  CFFTI     INITIALIZE CFFTF AND CFFTB
C     17.  CFFTF     FORWARD TRANSFORM OF A COMPLEX PERIODIC SEQUENCE
C     18.  CFFTB     UNNORMALIZED INVERSE OF CFFTF
C
C***ROUTINES CALLED  (NONE)
C***END PROLOGUE  FFTDOC
C***FIRST EXECUTABLE STATEMENT  FFTDOC
       RETURN
       END

      SUBROUTINE PASSB (NAC,IDO,IP,L1,IDL1,CC,C1,C2,CH,CH2,WA)
C***BEGIN PROLOGUE  PASSB
c   revised by Jim larsen 4/6/99
C***REFER TO CFFTB
C***ROUTINES CALLED  (NONE)
C***END PROLOGUE  PASSB
      DIMENSION CH(IDO,L1,IP), CC(IDO,IP,L1), C1(IDO,L1,IP), WA(*),
     .          C2(IDL1,IP), CH2(IDL1,IP)
C***FIRST EXECUTABLE STATEMENT  PASSB
      IDOT = IDO/2
      IPP2 = IP+2
      IPPH = (IP+1)/2
      IDP = IP*IDO

      IF (IDO .GE. L1) THEN
        DO J=2,IPPH
          JC = IPP2-J
          DO  K=1,L1
            DO I=1,IDO
              X1 = CC(I,J,K)
              X2 = CC(I,JC,K)
              CH(I,K,J)  = X1 + X2
              CH(I,K,JC) = X1 - X2
            ENDDO
          ENDDO
        ENDDO
        DO K=1,L1
          DO I=1,IDO
            CH(I,K,1) = CC(I,1,K)
          ENDDO
        ENDDO
      ELSE
        DO J=2,IPPH
          JC = IPP2-J
          DO I=1,IDO
            DO K=1,L1
              X1 = CC(I,J,K)
              X2 = CC(I,JC,K)
              CH(I,K,J)  = X1 + X2 
              CH(I,K,JC) = X1 - X2 
            ENDDO
          ENDDO
        ENDDO
        DO I=1,IDO
          DO K=1,L1
            CH(I,K,1) = CC(I,1,K)
          ENDDO
        ENDDO
      ENDIF
      IDL = 2-IDO
      INC = 0
      DO L=2,IPPH
        LC = IPP2-L
        IDL = IDL+IDO
        W1 = WA(IDL)
        W2 = WA(IDL-1)
        DO IK=1,IDL1
          C2(IK,L)  = CH2(IK,1) + W2* CH2(IK,2)
          C2(IK,LC) = W1* CH2(IK,IP)
        ENDDO
        IDLJ = IDL
        INC = INC+IDO
        DO J=3,IPPH
          JC = IPP2-J
          IDLJ = IDLJ+INC
          IF (IDLJ .GT. IDP) IDLJ = IDLJ-IDP
          WAR = WA(IDLJ-1)
          WAI = WA(IDLJ)
          DO IK=1,IDL1
            C2(IK,L)  = C2(IK,L)  + WAR* CH2(IK,J)
            C2(IK,LC) = C2(IK,LC) + WAI* CH2(IK,JC)
          ENDDO
        ENDDO
      ENDDO
      DO J=2,IPPH
        DO IK=1,IDL1
          CH2(IK,1) = CH2(IK,1) + CH2(IK,J)
        ENDDO
      ENDDO
      DO J=2,IPPH
        JC = IPP2-J
        DO IK=2,IDL1,2
          IK1 = IK-1
          X1 = C2(IK1,J)
          X2 = C2(IK1,JC)
          X3 = C2(IK,J)
          X4 = C2(IK,JC)
          CH2(IK1,J)  =  X1 - X4
          CH2(IK1,JC) =  X1 + X4
          CH2(IK,J)   =  X3 + X2
          CH2(IK,JC)  =  X3 - X2
        ENDDO
      ENDDO
      NAC = 1
      IF (IDO .EQ. 2) RETURN
      NAC = 0
      DO IK=1,IDL1
        C2(IK,1) = CH2(IK,1)
      ENDDO
      DO J=2,IP
        DO K=1,L1
          C1(1,K,J) = CH(1,K,J)
          C1(2,K,J) = CH(2,K,J)
        ENDDO
      ENDDO
      IF (IDOT .LE. L1) THEN
        IDIJ = 0
        DO J=2,IP
          IDIJ = IDIJ+2
          DO I=4,IDO,2
            IDIJ = IDIJ+2
            I1 = I-1
            W1 = WA(IDIJ-1)
            W2 = WA(IDIJ)
            DO K=1,L1
              X1 = CH(I1,K,J)
              X2 = CH(I,K,J)
              C1(I1,K,J) = W1* X1 - W2* X2
              C1(I,K,J)  = W1* X2 + W2* X1
            ENDDO
          ENDDO
        ENDDO
        RETURN
      ELSE
        IDJ = 2-IDO
        DO J=2,IP
          IDJ = IDJ+IDO
          DO K=1,L1
            IDIJ = IDJ
            DO I=4,IDO,2
              I1 = I-1
              IDIJ = IDIJ+2
              W1 = WA(IDIJ-1)
              W2 = WA(IDIJ) 
              X1 = CH(I1,K,J)
              X2 = CH(I,K,J)
              C1(I1,K,J) = W1* X1 - W2* X2
              C1(I,K,J)  = W1* X2 + W2* X1
            ENDDO
          ENDDO
        ENDDO
      ENDIF
      RETURN
      END
      SUBROUTINE PASSB2 (IDO,L1,CC,CH,WA1)
C***BEGIN PROLOGUE  PASSB2
c   revised by Jim larsen 4/9/99
C***REFER TO CFFTB
C***ROUTINES CALLED  (NONE)
C***END PROLOGUE  PASSB2
      DIMENSION CC(IDO,2,L1),CH(IDO,L1,2),WA1(*)
C***FIRST EXECUTABLE STATEMENT  PASSB2
      IF (IDO .LE. 2) THEN
        DO K=1,L1
          X11 = CC(1,1,K) 
          X12 = CC(1,2,K) 
          X21 = CC(2,1,K) 
          X22 = CC(2,2,K) 
          CH(1,K,1) = X11 + X12
          CH(1,K,2) = X11 - X12
          CH(2,K,1) = X21 + X22
          CH(2,K,2) = X21 - X22
        ENDDO
      ELSE
        DO K=1,L1
          DO I=2,IDO,2
            J = I-1
            W1 = WA1(J)
            W2 = WA1(I)
            XJ1 = CC(J,1,K) 
            XJ2 = CC(J,2,K) 
            XI1 = CC(I,1,K) 
            XI2 = CC(I,2,K) 
            TR2 = XJ1 - XJ2
            TI2 = XI1 - XI2
            CH(J,K,1) = XJ1 + XJ2
            CH(I,K,1) = XI1 + XI2
            CH(J,K,2) = W1* TR2 - W2* TI2
            CH(I,K,2) = W1* TI2 + W2* TR2
          ENDDO
        ENDDO
      ENDIF
      RETURN
      END
      SUBROUTINE PASSB3 (IDO,L1,CC,CH,WA1,WA2)
C***BEGIN PROLOGUE  PASSB3
c   revised by Jim larsen 4/9/99
C***REFER TO CFFTB
C***ROUTINES CALLED  (NONE)
C***END PROLOGUE  PASSB3
      DIMENSION CC(IDO,3,L1),CH(IDO,L1,3),WA1(*),WA2(*)
C***FIRST EXECUTABLE STATEMENT  PASSB3
      TAUR = -.5
      TAUI =  .866025403784439
      IF (IDO .EQ. 2) THEN
        DO K=1,L1
          X11 = CC(1,1,K) 
          X12 = CC(1,2,K) 
          X13 = CC(1,3,K) 
          X21 = CC(2,1,K) 
          X22 = CC(2,2,K) 
          X23 = CC(2,3,K) 
          TR2 = X12 + X13
          TI2 = X22 + X23
          CR2 = X11 + TAUR* TR2
          CI2 = X21 + TAUR* TI2
          CR3 = TAUI* (X12 - X13)
          CI3 = TAUI* (X22 - X23)
          CH(1,K,1) = X11 + TR2
          CH(2,K,1) = X21 + TI2
          CH(1,K,2) = CR2 - CI3
          CH(1,K,3) = CR2 + CI3
          CH(2,K,2) = CI2 + CR3
          CH(2,K,3) = CI2 - CR3
        ENDDO
      ELSE
        DO K=1,L1
          DO I=2,IDO,2
            J = I-1
            W1 = WA1(J)
            W2 = WA1(I)
            W3 = WA2(J)
            W4 = WA2(I)
            XJ1 = CC(J,1,K) 
            XJ2 = CC(J,2,K) 
            XJ3 = CC(J,3,K) 
            XI1 = CC(I,1,K) 
            XI2 = CC(I,2,K) 
            XI3 = CC(I,3,K)
            TR2 = XJ2 + XJ3
            TI2 = XI2 + XI3
            CR2 = XJ1 + TAUR* TR2
            CI2 = XI1 + TAUR* TI2
            CR3 = TAUI* (XJ2 - XJ3)
            CI3 = TAUI* (XI2 - XI3)
            DR2 = CR2 - CI3
            DR3 = CR2 + CI3
            DI2 = CI2 + CR3
            DI3 = CI2 - CR3
            CH(J,K,1) = XJ1 + TR2
            CH(I,K,1) = XI1 + TI2
            CH(I,K,2) = W1* DI2 + W2* DR2
            CH(J,K,2) = W1* DR2 - W2* DI2
            CH(I,K,3) = W3* DI3 + W4* DR3
            CH(J,K,3) = W3* DR3 - W4* DI3
          ENDDO
        ENDDO
      ENDIF
      RETURN
      END
      SUBROUTINE PASSB4 (IDO,L1,CC,CH,WA1,WA2,WA3)
C***BEGIN PROLOGUE  PASSB4
c   revised by Jim larsen 4/9/99
C***REFER TO CFFTB
C***ROUTINES CALLED (NONE)
C***END PROLOGUE  PASSB4
      DIMENSION CC(IDO,4,L1),CH(IDO,L1,4),WA1(*),WA2(*),WA3(*)
C***FIRST EXECUTABLE STATEMENT  PASSB4
      IF (IDO .EQ. 2) THEN
        DO K=1,L1
          X11 = CC(1,1,K)
          X12 = CC(1,2,K)
          X13 = CC(1,3,K)
          X14 = CC(1,4,K)
          X21 = CC(2,1,K)
          X22 = CC(2,2,K)
          X23 = CC(2,3,K)
          X24 = CC(2,4,K)
          TI1 = X21 - X23
          TI2 = X21 + X23
          TR4 = X24 - X22
          TI3 = X24 + X22
          TR1 = X11 - X13
          TR2 = X11 + X13
          TI4 = X12 - X14
          TR3 = X12 + X14
          CH(1,K,1) = TR2 + TR3
          CH(1,K,3) = TR2 - TR3
          CH(2,K,1) = TI2 + TI3
          CH(2,K,3) = TI2 - TI3
          CH(1,K,2) = TR1 + TR4
          CH(1,K,4) = TR1 - TR4
          CH(2,K,2) = TI1 + TI4
          CH(2,K,4) = TI1 - TI4
        ENDDO
      ELSE
        DO K=1,L1
          DO I=2,IDO,2
            J = I-1
            W1 = WA1(J)
            W2 = WA1(I)
            W3 = WA2(J)
            W4 = WA2(I)
            W5 = WA3(J)
            W6 = WA3(I)
            XJ1 = CC(J,1,K)
            XJ2 = CC(J,2,K)
            XJ3 = CC(J,3,K)
            XJ4 = CC(J,4,K)
            XI1 = CC(I,1,K)
            XI2 = CC(I,2,K)
            XI3 = CC(I,3,K)
            XI4 = CC(I,4,K)
            TI1 = XI1 - XI3
            TI2 = XI1 + XI3
            TR4 = XI4 - XI2
            TI3 = XI4 + XI2
            TR1 = XJ1 - XJ3
            TR2 = XJ1 + XJ3
            TI4 = XJ2 - XJ4
            TR3 = XJ2 + XJ4
            CR3 = TR2 - TR3
            CI3 = TI2 - TI3
            CR2 = TR1 + TR4
            CR4 = TR1 - TR4
            CI2 = TI1 + TI4
            CI4 = TI1 - TI4
            CH(J,K,1) = TR2 + TR3
            CH(I,K,1) = TI2 + TI3
            CH(J,K,2) = W1* CR2 - W2* CI2
            CH(I,K,2) = W1* CI2 + W2* CR2
            CH(J,K,3) = W3* CR3 - W4* CI3
            CH(I,K,3) = W3* CI3 + W4* CR3
            CH(J,K,4) = W5* CR4 - W6* CI4
            CH(I,K,4) = W5* CI4 + W6* CR4
          ENDDO
        ENDDO
      ENDIF
      RETURN
      END
      SUBROUTINE PASSB5 (IDO,L1,CC,CH,WA1,WA2,WA3,WA4)
C***BEGIN PROLOGUE  PASSB5
c   revised by Jim larsen 4/9/99
C***REFER TO CFFTB
C***ROUTINES CALLED  (NONE)
C***END PROLOGUE  PASSB5
      DIMENSION CC(IDO,5,L1),CH(IDO,L1,5),WA1(*),WA2(*),WA3(*),WA4(*)
C***FIRST EXECUTABLE STATEMENT  PASSB5
      TR11 =  .309016994374947
      TI11 =  .951056516295154
      TR12 = -.809016994374947
      TI12 =  .587785252292473
      IF (IDO .EQ. 2) THEN
        DO K=1,L1
          X11 = CC(1,1,K)
          X12 = CC(1,2,K)
          X13 = CC(1,3,K)
          X14 = CC(1,4,K)
          X15 = CC(1,5,K)
          X21 = CC(2,1,K)
          X22 = CC(2,2,K)
          X23 = CC(2,3,K)
          X24 = CC(2,4,K)
          X25 = CC(2,5,K)
          TI5 = X22 - X25 
          TI2 = X22 + X25 
          TI4 = X23 - X24
          TI3 = X23 + X24
          TR5 = X12 - X15
          TR2 = X12 + X15
          TR4 = X13 - X14 
          TR3 = X13 + X14 
          CR2 = X11 + TR11* TR2 + TR12* TR3
          CI2 = X21 + TR11* TI2 + TR12* TI3
          CR3 = X11 + TR12* TR2 + TR11* TR3
          CI3 = X21 + TR12* TI2 + TR11* TI3
          CR5 = TI11* TR5 + TI12* TR4
          CI5 = TI11* TI5 + TI12* TI4
          CR4 = TI12* TR5 - TI11* TR4
          CI4 = TI12* TI5 - TI11* TI4
          CH(1,K,1) = X11 + TR2 + TR3
          CH(2,K,1) = X21 + TI2 + TI3
          CH(1,K,2) = CR2 - CI5
          CH(1,K,5) = CR2 + CI5
          CH(2,K,2) = CI2 + CR5
          CH(2,K,3) = CI3 + CR4
          CH(1,K,3) = CR3 - CI4
          CH(1,K,4) = CR3 + CI4
          CH(2,K,4) = CI3 - CR4
          CH(2,K,5) = CI2 - CR5
        ENDDO
      ELSE
        DO K=1,L1
          DO I=2,IDO,2
            J = I-1
            W1 = WA1(J)
            W2 = WA1(I)
            W3 = WA2(J)
            W4 = WA2(I)
            W5 = WA3(J)
            W6 = WA3(I)
            W7 = WA4(J)
            W8 = WA4(I)
            XJ1 = CC(J,1,K)
            XJ2 = CC(J,2,K)
            XJ3 = CC(J,3,K)
            XJ4 = CC(J,4,K)
            XJ5 = CC(J,5,K)
            XI1 = CC(I,1,K)
            XI2 = CC(I,2,K)
            XI3 = CC(I,3,K)
            XI4 = CC(I,4,K)
            XI5 = CC(I,5,K)
            TI5 = XI2 - XI5 
            TI2 = XI2 + XI5 
            TI4 = XI3 - XI4
            TI3 = XI3 + XI4
            TR5 = XJ2 - XJ5
            TR2 = XJ2 + XJ5
            TR4 = XJ3 - XJ4
            TR3 = XJ3 + XJ4
            CR2 = XJ1 + TR11* TR2 + TR12* TR3
            CI2 = XI1 + TR11* TI2 + TR12* TI3
            CR3 = XJ1 + TR12* TR2 + TR11* TR3
            CI3 = XI1 + TR12* TI2 + TR11* TI3
            CR5 = TI11* TR5 + TI12* TR4
            CI5 = TI11* TI5 + TI12* TI4
            CR4 = TI12* TR5 - TI11* TR4
            CI4 = TI12* TI5 - TI11* TI4
            DR3 = CR3 - CI4
            DR4 = CR3 + CI4
            DI3 = CI3 + CR4
            DI4 = CI3 - CR4
            DR5 = CR2 + CI5
            DR2 = CR2 - CI5
            DI5 = CI2 - CR5
            DI2 = CI2 + CR5
            CH(J,K,1) = XJ1 + TR2 + TR3
            CH(I,K,1) = XI1 + TI2 + TI3
            CH(J,K,2) = W1* DR2 - W2* DI2
            CH(I,K,2) = W1* DI2 + W2* DR2
            CH(J,K,3) = W3* DR3 - W4* DI3
            CH(I,K,3) = W3* DI3 + W4* DR3
            CH(J,K,4) = W5* DR4 - W6* DI4
            CH(I,K,4) = W5* DI4 + W6* DR4
            CH(J,K,5) = W7* DR5 - W8* DI5
            CH(I,K,5) = W7* DI5 + W8* DR5
          ENDDO
        ENDDO
      ENDIF
      RETURN
      END
      SUBROUTINE PASSF (NAC,IDO,IP,L1,IDL1,CC,C1,C2,CH,CH2,WA)
C***BEGIN PROLOGUE  PASSF
c   revised by Jim larsen 4/7/99
C***REFER TO CFFTF
C***ROUTINES CALLED  (NONE)
C***END PROLOGUE  PASSF
      DIMENSION CH(IDO,L1,IP), CC(IDO,IP,L1), C1(IDO,L1,IP), WA(*),
     .          C2(IDL1,IP), CH2(IDL1,IP)
C***FIRST EXECUTABLE STATEMENT  PASSF
      IDOT = IDO/2
      IPP2 = IP+2
      IPPH = (IP+1)/2
      IDP = IP*IDO

      IF (IDO .GE. L1) THEN
        DO J=2,IPPH
          JC = IPP2-J
          DO K=1,L1
            DO I=1,IDO
              X1 = CC(I,J,K)
              X2 = CC(I,JC,K)
              CH(I,K,J)  = X1 + X2
              CH(I,K,JC) = X1 - X2
            ENDDO
          ENDDO
        ENDDO
        DO K=1,L1
          DO I=1,IDO
            CH(I,K,1) = CC(I,1,K)
          ENDDO
        ENDDO
      ELSE
        DO J=2,IPPH
          JC = IPP2-J
          DO I=1,IDO
            DO K=1,L1
              X1 = CC(I,J,K)
              X2 = CC(I,JC,K)
              CH(I,K,J)  = X1 + X2 
              CH(I,K,JC) = X1 - X2 
            ENDDO
          ENDDO
        ENDDO
        DO I=1,IDO
          DO K=1,L1
            CH(I,K,1) = CC(I,1,K)
          ENDDO
        ENDDO
      ENDIF
      IDL = 2-IDO
      INC = 0
      DO L=2,IPPH
        LC = IPP2-L
        IDL = IDL+IDO
        W1 = WA(IDL)
        W2 = WA(IDL-1)
        DO IK=1,IDL1
          C2(IK,L)  = CH2(IK,1) + W2* CH2(IK,2)
          C2(IK,LC) = -W1* CH2(IK,IP)
        ENDDO
        IDLJ = IDL
        INC = INC+IDO
        DO J=3,IPPH
          JC = IPP2-J
          IDLJ = IDLJ+INC
          IF (IDLJ .GT. IDP) IDLJ = IDLJ-IDP
          WAR = WA(IDLJ-1)
          WAI = WA(IDLJ)
          DO IK=1,IDL1
            C2(IK,L)  = C2(IK,L)  + WAR* CH2(IK,J)
            C2(IK,LC) = C2(IK,LC) - WAI* CH2(IK,JC)
          ENDDO
        ENDDO
      ENDDO
      DO J=2,IPPH
        DO IK=1,IDL1
          CH2(IK,1) = CH2(IK,1)+CH2(IK,J)
        ENDDO
      ENDDO
      DO J=2,IPPH
        JC = IPP2-J
        DO IK=2,IDL1,2
          IK1 = IK-1
          X1 = C2(IK1,J)
          X2 = C2(IK1,JC)
          X3 = C2(IK,J)
          X4 = C2(IK,JC)
          CH2(IK1,J)  = X1 - X4
          CH2(IK1,JC) = X1 + X4
          CH2(IK,J)   = X3 + X2
          CH2(IK,JC)  = X3 - X2
        ENDDO
      ENDDO
      NAC = 1
      IF (IDO .EQ. 2) RETURN
      NAC = 0
      DO IK=1,IDL1
        C2(IK,1) = CH2(IK,1)
      ENDDO
      DO J=2,IP
        DO K=1,L1
          C1(1,K,J) = CH(1,K,J)
          C1(2,K,J) = CH(2,K,J)
        ENDDO
      ENDDO
      IF (IDOT .LE. L1) THEN
        IDIJ = 0
        DO J=2,IP
          IDIJ = IDIJ+2
          DO I=4,IDO,2
            I1 = I-1
            IDIJ = IDIJ+2
            W1 = WA(IDIJ-1)
            W2 = WA(IDIJ)
            DO  K=1,L1
              X1 = CH(I1,K,J)
              X2 = CH(I,K,J)
              C1(I1,K,J) = W1* X1 + W2* X2
              C1(I,K,J)  = W1* X2 - W2* X1
            ENDDO
          ENDDO
        ENDDO
        RETURN
      ELSE
        IDJ = 2-IDO
        DO J=2,IP
          IDJ = IDJ+IDO
          DO K=1,L1
            IDIJ = IDJ
            DO I=4,IDO,2
              I1 = I-1
              IDIJ = IDIJ+2
              W1 = WA(IDIJ-1)
              W2 = WA(IDIJ) 
              X1 = CH(I1,K,J)
              X2 = CH(I,K,J)
              C1(I1,K,J) = W1* X1 + W2* X2
              C1(I,K,J)  = W1* X2 - W2* X1
            ENDDO
          ENDDO
        ENDDO
      ENDIF
      RETURN
      END
      SUBROUTINE PASSF2 (IDO,L1,CC,CH,WA1)
C***BEGIN PROLOGUE  PASSF2
c   revised by Jim larsen 4/9/99
C***REFER TO CFFTF
C***ROUTINES CALLED  (NONE)
C***END PROLOGUE  PASSF2
      DIMENSION CC(IDO,2,L1),CH(IDO,L1,2),WA1(*)
C***FIRST EXECUTABLE STATEMENT  PASSF2
      IF (IDO .LE. 2) THEN
        DO K=1,L1
          X11 = CC(1,1,K) 
          X12 = CC(1,2,K) 
          X21 = CC(2,1,K) 
          X22 = CC(2,2,K) 
          CH(1,K,1) = X11 + X12
          CH(1,K,2) = X11 - X12
          CH(2,K,1) = X21 + X22
          CH(2,K,2) = X21 - X22
        ENDDO
      ELSE
        DO K=1,L1
          DO I=2,IDO,2
            J = I-1
            W1 = WA1(J)
            W2 = WA1(I)
            XJ1 = CC(J,1,K) 
            XJ2 = CC(J,2,K) 
            XI1 = CC(I,1,K) 
            XI2 = CC(I,2,K) 
            TR2 = XJ1 - XJ2
            TI2 = XI1 - XI2
            CH(J,K,1) = XJ1 + XJ2
            CH(I,K,1) = XI1 + XI2
            CH(J,K,2) = W1* TR2 + W2* TI2
            CH(I,K,2) = W1* TI2 - W2* TR2
          ENDDO
        ENDDO
      ENDIF
      RETURN
      END
      SUBROUTINE PASSF3 (IDO,L1,CC,CH,WA1,WA2)
C***BEGIN PROLOGUE  PASSF3
c   revised by Jim larsen 4/9/99
C***REFER TO CFFTF
C***ROUTINES CALLED  (NONE)
C***END PROLOGUE  PASSF3
      DIMENSION CC(IDO,3,L1),CH(IDO,L1,3),WA1(*),WA2(*)
C***FIRST EXECUTABLE STATEMENT  PASSF3
      TAUR = -.5
      TAUI = -.86602540378443
      IF (IDO .EQ. 2) THEN
        DO K=1,L1
          X11 = CC(1,1,K) 
          X12 = CC(1,2,K) 
          X13 = CC(1,3,K) 
          X21 = CC(2,1,K) 
          X22 = CC(2,2,K) 
          X23 = CC(2,3,K) 
          TR2 = X12 + X13
          TI2 = X22 + X23
          CR2 = X11 + TAUR* TR2
          CI2 = X21 + TAUR* TI2
          CR3 = TAUI* (X12 - X13)
          CI3 = TAUI* (X22 - X23)
          CH(1,K,1) = X11 + TR2
          CH(2,K,1) = X21 + TI2
          CH(1,K,2) = CR2 - CI3
          CH(1,K,3) = CR2 + CI3
          CH(2,K,2) = CI2 + CR3
          CH(2,K,3) = CI2 - CR3
        ENDDO
      ELSE
        DO K=1,L1
          DO I=2,IDO,2
            J = I-1
            W1 = WA1(J)
            W2 = WA1(I)
            W3 = WA2(J)
            W4 = WA2(I)
            XJ1 = CC(J,1,K) 
            XJ2 = CC(J,2,K) 
            XJ3 = CC(J,3,K) 
            XI1 = CC(I,1,K) 
            XI2 = CC(I,2,K) 
            XI3 = CC(I,3,K)
            TR2 = XJ2 + XJ3
            TI2 = XI2 + XI3
            CR2 = XJ1 + TAUR* TR2
            CI2 = XI1 + TAUR* TI2
            CR3 = TAUI* (XJ2 - XJ3)
            CI3 = TAUI* (XI2 - XI3)
            DR2 = CR2 - CI3
            DR3 = CR2 + CI3
            DI2 = CI2 + CR3
            DI3 = CI2 - CR3
            CH(J,K,1) = XJ1 + TR2
            CH(I,K,1) = XI1 + TI2
            CH(I,K,2) = W1* DI2 - W2* DR2
            CH(J,K,2) = W1* DR2 + W2* DI2
            CH(I,K,3) = W3* DI3 - W4* DR3
            CH(J,K,3) = W3* DR3 + W4* DI3
          ENDDO
        ENDDO
      ENDIF
      RETURN
      END
      SUBROUTINE PASSF4 (IDO,L1,CC,CH,WA1,WA2,WA3)
C***BEGIN PROLOGUE  PASSF4
c   revised by Jim larsen 4/9/99
C***REFER TO CFFTF
C***ROUTINES CALLED  (NONE)
C***END PROLOGUE  PASSF4
      DIMENSION CC(IDO,4,L1),CH(IDO,L1,4),WA1(*),WA2(*),WA3(*)
C***FIRST EXECUTABLE STATEMENT  PASSF4
      IF (IDO .EQ. 2) THEN
        DO K=1,L1
          X11 = CC(1,1,K)
          X12 = CC(1,2,K)
          X13 = CC(1,3,K)
          X14 = CC(1,4,K)
          X21 = CC(2,1,K)
          X22 = CC(2,2,K)
          X23 = CC(2,3,K)
          X24 = CC(2,4,K)
          TI1 = X21 - X23
          TI2 = X21 + X23
          TR4 = X22 - X24
          TI3 = X22 + X24
          TR1 = X11 - X13
          TR2 = X11 + X13
          TI4 = X14 - X12
          TR3 = X14 + X12
          CH(1,K,1) = TR2 + TR3
          CH(1,K,3) = TR2 - TR3
          CH(2,K,1) = TI2 + TI3
          CH(2,K,3) = TI2 - TI3
          CH(1,K,2) = TR1 + TR4
          CH(1,K,4) = TR1 - TR4
          CH(2,K,2) = TI1 + TI4
          CH(2,K,4) = TI1 - TI4
        ENDDO
      ELSE
        DO K=1,L1
          DO I=2,IDO,2
            J = I-1
            W1 = WA1(J)
            W2 = WA1(I)
            W3 = WA2(J)
            W4 = WA2(I)
            W5 = WA3(J)
            W6 = WA3(I)
            XJ1 = CC(J,1,K)
            XJ2 = CC(J,2,K)
            XJ3 = CC(J,3,K)
            XJ4 = CC(J,4,K)
            XI1 = CC(I,1,K)
            XI2 = CC(I,2,K)
            XI3 = CC(I,3,K)
            XI4 = CC(I,4,K)
            TI1 = XI1 - XI3
            TI2 = XI1 + XI3
            TR4 = XI2 - XI4
            TI3 = XI2 + XI4
            TR1 = XJ1 - XJ3
            TR2 = XJ1 + XJ3
            TI4 = XJ4 - XJ2
            TR3 = XJ4 + XJ2
            CR3 = TR2 - TR3
            CI3 = TI2 - TI3
            CR2 = TR1 + TR4
            CR4 = TR1 - TR4
            CI2 = TI1 + TI4
            CI4 = TI1 - TI4
            CH(J,K,1) = TR2 + TR3
            CH(I,K,1) = TI2 + TI3
            CH(J,K,2) = W1* CR2 + W2* CI2
            CH(I,K,2) = W1* CI2 - W2* CR2
            CH(J,K,3) = W3* CR3 + W4* CI3
            CH(I,K,3) = W3* CI3 - W4* CR3
            CH(J,K,4) = W5* CR4 + W6* CI4
            CH(I,K,4) = W5* CI4 - W6* CR4
          ENDDO
        ENDDO
      ENDIF
      RETURN
      END
      SUBROUTINE PASSF5 (IDO,L1,CC,CH,WA1,WA2,WA3,WA4)
C***BEGIN PROLOGUE  PASSF5
c   revised by Jim larsen 4/9/99
C***REFER TO CFFTF
C***ROUTINES CALLED  (NONE)
C***END PROLOGUE  PASSF5
      DIMENSION CC(IDO,5,L1),CH(IDO,L1,5),WA1(*),WA2(*),WA3(*),WA4(*)
C***FIRST EXECUTABLE STATEMENT  PASSF5
      TR11 =  .309016994374947
      TI11 = -.951056516295154 
      TR12 = -.809016994374947 
      TI12 = -.587785252292473 
      IF (IDO .EQ. 2) THEN
        DO K=1,L1
          X11 = CC(1,1,K)
          X12 = CC(1,2,K)
          X13 = CC(1,3,K)
          X14 = CC(1,4,K)
          X15 = CC(1,5,K)
          X21 = CC(2,1,K)
          X22 = CC(2,2,K)
          X23 = CC(2,3,K)
          X24 = CC(2,4,K)
          X25 = CC(2,5,K)
          TI5 = X22 - X25
          TI2 = X22 + X25
          TI4 = X23 - X24
          TI3 = X23 + X24
          TR5 = X12 - X15
          TR2 = X12 + X15
          TR4 = X13 - X14 
          TR3 = X13 + X14 
          CR2 = X11 + TR11* TR2 + TR12* TR3
          CI2 = X21 + TR11* TI2 + TR12* TI3
          CR3 = X11 + TR12* TR2 + TR11* TR3
          CI3 = X21 + TR12* TI2 + TR11* TI3
          CR5 = TI11* TR5 + TI12* TR4
          CI5 = TI11* TI5 + TI12* TI4
          CR4 = TI12* TR5 - TI11* TR4
          CI4 = TI12* TI5 - TI11* TI4
          CH(1,K,1) = X11 + TR2 + TR3
          CH(2,K,1) = X21 + TI2 + TI3
          CH(1,K,2) = CR2 - CI5
          CH(1,K,5) = CR2 + CI5
          CH(2,K,2) = CI2 + CR5
          CH(2,K,3) = CI3 + CR4
          CH(1,K,3) = CR3 - CI4
          CH(1,K,4) = CR3 + CI4
          CH(2,K,4) = CI3 - CR4
          CH(2,K,5) = CI2 - CR5
        ENDDO
      ELSE
        DO K=1,L1
          DO I=2,IDO,2
            J = I-1
            W1 = WA1(J)
            W2 = WA1(I)
            W3 = WA2(J)
            W4 = WA2(I)
            W5 = WA3(J)
            W6 = WA3(I)
            W7 = WA4(J)
            W8 = WA4(I)
            XJ1 = CC(J,1,K)
            XJ2 = CC(J,2,K)
            XJ3 = CC(J,3,K)
            XJ4 = CC(J,4,K)
            XJ5 = CC(J,5,K)
            XI1 = CC(I,1,K)
            XI2 = CC(I,2,K)
            XI3 = CC(I,3,K)
            XI4 = CC(I,4,K)
            XI5 = CC(I,5,K)
            TI5 = XI2 - XI5 
            TI2 = XI2 + XI5 
            TI4 = XI3 - XI4
            TI3 = XI3 + XI4
            TR5 = XJ2 - XJ5
            TR2 = XJ2 + XJ5
            TR4 = XJ3 - XJ4
            TR3 = XJ3 + XJ4
            CR2 = XJ1 + TR11* TR2 + TR12* TR3
            CI2 = XI1 + TR11* TI2 + TR12* TI3
            CR3 = XJ1 + TR12* TR2 + TR11* TR3
            CI3 = XI1 + TR12* TI2 + TR11* TI3
            CR5 = TI11* TR5 + TI12* TR4
            CI5 = TI11* TI5 + TI12* TI4
            CR4 = TI12* TR5 - TI11* TR4
            CI4 = TI12* TI5 - TI11* TI4
            DR3 = CR3 - CI4
            DR4 = CR3 + CI4
            DI3 = CI3 + CR4
            DI4 = CI3 - CR4
            DR5 = CR2 + CI5
            DR2 = CR2 - CI5
            DI5 = CI2 - CR5
            DI2 = CI2 + CR5
            CH(J,K,1) = XJ1 + TR2 + TR3
            CH(I,K,1) = XI1 + TI2 + TI3
            CH(J,K,2) = W1* DR2 + W2* DI2
            CH(I,K,2) = W1* DI2 - W2* DR2
            CH(J,K,3) = W3* DR3 + W4* DI3
            CH(I,K,3) = W3* DI3 - W4* DR3
            CH(J,K,4) = W5* DR4 + W6* DI4
            CH(I,K,4) = W5* DI4 - W6* DR4
            CH(J,K,5) = W7* DR5 + W8* DI5
            CH(I,K,5) = W7* DI5 - W8* DR5
          ENDDO
        ENDDO
      ENDIF
      RETURN
      END
      SUBROUTINE PROLATE (W, ND, ISW)
C  CALCULATES PROLATE SPHEROIDAL WAVEFUNCTION DATA WINDOW FOR
C  HIGH RESOLUTION FOURIER ANALYSIS
C  REF: D.J.THOMSON, BELL SYST. TECH. J. 56,1769-1815 (1977)
C
C  W IS A SINGLE PRECISON REAL ARRAY OF DATA WINDOW VALUES
C  ND IS THE NUMBER OF POINTS IN W
C  ISW IS A SWITCH--ISW=4 MEANS USE A 4-PI WINDOW, ISW=1 MEANS USE
C      THE HIGHER RESOLUTION PI WINDOW

C  SCALE FACTORS=INTEGRAL(BOXCAR)/INTEGRAL(PROLATE WINDOW) ARE:
C    4 PI PROLATE WINDOW--1.425658520238489
C    PI PROLATE WINDOW--1.057568010371401
C  THESE ARE THE NUMBERS TO MULTIPLY THE SPECTRUM BY FOR COMPARISON
C  WITH OTHER WINDOWS

c     revise 4/20/98

      REAL W(*)
      REAL*8 UN, D, U, WI
      UN = DBLE(4)/(DBLE(ND-1)* DBLE(ND-1))
      IF (ISW .EQ. 4) THEN
      	D = SQRT (DBLE(2)/ 0.508125548147497D0)
       	DO I = 1, ND
      	  U  = DBLE(ND-I)* DBLE(I-1)* UN
      	  WI = D*(((((((((((((((((((((
     $        2.6197747176990866D-11*U+2.9812025862125737D-10)*U+
     $        3.0793023552299688D-9)*U+2.8727486379692354D-8)*U+
     $        2.4073904863499725D-7)*U+1.8011359410323110D-6)*U+
     $        1.1948784162527709D-5)*U+6.9746276641509466D-5)*U+
     $        3.5507361197109845D-4)*U+1.5607376779150113D-3)*U+
     $        5.8542015072142441D-3)*U+1.8482388295519675D-2)*U+
     $        4.8315671140720506D-2)*U+1.0252816895203814D-1)*U+
     $        1.7233583271499150D-1)*U+2.2242525852102708D-1)*U+
     $        2.1163435697968192D-1)*U+1.4041394473085307D-1)*U+
     $        5.9923940532892353D-2)*U+1.4476509897632850D-2)*U+
     $        1.5672417352380246D-3)*U+4.2904633140034110D-5)
          W(I) = WI
      	ENDDO
      	RETURN
      ELSE
      	D = SQRT (DBLE(2))
      	DO I = 1, ND
      	  U  = DBLE(ND-I)* DBLE(I-1)* UN
      	  WI = D*((((((((((
     $        5.3476939016920851D-11*U+2.2654256220146656D-9)*U+
     $        7.8075102004229667D-8)*U+2.1373409644281953D-6)*U+
     $        4.5094847544714943D-5)*U+7.0498957221483167D-4)*U+
     $        7.7412693304064753D-3)*U+5.5280627452077586D-2)*U+
     $        2.2753754228751827D-1)*U+4.3433904277546202D-1)*U+
     $        2.2902051859068017D-1)
          W(I) = WI
        ENDDO
        RETURN
      ENDIF
      END
      SUBROUTINE RFFTB (N,R,WSAVE)
C***BEGIN PROLOGUE  RFFTB
c   revised by Jim larsen 4/2/99
C***CATEGORY NO.  D6
C***KEYWORDS FFT,FAST FOURIER TRANSFORM,INVERSE,INVERSE FFT
C***DATE WRITTEN  FEBRUARY 1978
C***AUTHOR  SWARZTRAUBER P.N. (NCAR)
C***PURPOSE
C  INVERSE FFT OF A REAL PERIODIC SEQUENCE
C***DESCRIPTION
C
C     SUBROUTINE RFFTB COMPUTES THE REAL PERODIC SEQUENCE FROM ITS
C     FOURIER COEFFICIENTS (FOURIER SYNTHESIS). THE TRANSFORM IS DEFINED
C     BELOW AT OUTPUT PARAMETER R.
C
C     INPUT PARAMETERS
C
C     N       THE LENGTH OF THE ARRAY R. N MUST BE EVEN AND THE METHOD
C             IS MOST EFFICIENT WHEN N IS A PRODUCT OF SMALL PRIMES.
C             N MAY CHANGE SO LONG AS DIFFERENT WORK ARRAYS ARE PROVIDED
C
C     R       A REAL ARRAY OF LENGTH N WHICH CONTAINS THE SEQUENCE
C             TO BE TRANSFORMED
C
C     WSAVE   A WORK ARRAY WHICH MUST BE DIMENSIONED AT LEAST 2.5*N+15
C             IN THE PROGRAM THAT CALLS RFFTB. THE WSAVE ARRAY MUST BE
C             INITIALIZED BY CALLING SUBROUTINE RFFTI(N,WSAVE) AND A
C             DIFFERENT WSAVE ARRAY MUST BE USED FOR EACH DIFFERENT
C             VALUE OF N. THIS INITIALIZATION DOES NOT HAVE TO BE
C             REPEATED SO LONG AS N REMAINS UNCHANGED THUS SUBSEQUENT
C             TRANSFORMS CAN BE OBTAINED FASTER THAN THE FIRST.
C             THE SAME WSAVE ARRAY CAN BE USED BY RFFTF AND RFFTB.
C
C
C     OUTPUT PARAMETERS
C
C     R       FOR I=1,...,N
C
C                  R(I)=X(1)+(-1)**(I+1)*X(2)
C
C                       PLUS THE SUM FROM K=2 TO K=N/2 OF
C
C                         2*R(2K-1)*COS((K-1)*(I-1)*2*PI/N)
C
C                        +2*R(2K)*SIN((K-1)*(I-1)*2*PI/N)
C
C      *****  NOTE
C                  THIS TRANSFORM IS UNNORMALIZED SINCE A CALL OF RFFTF
C                  FOLLOWED BY A CALL OF RFFTB WILL MULTIPLY THE INPUT
C                  SEQUENCE BY 2*N.
C
C     WSAVE   CONTAINS RESULTS WHICH MUST NOT BE DESTROYED BETWEEN
C             CALLS OF RFFTB OR RFFTF.
C
C
C***REFERENCES
C***ROUTINES CALLED  RFFTB1,XERROR
C***END PROLOGUE  RFFTB
      DIMENSION       R(2,*)     ,WSAVE(*)
C***FIRST EXECUTABLE STATEMENT  RFFTB
      IF(MOD(N,2).NE.0)PRINT *, '23HRFFTB  ---N IS NOT EVEN,23,2,1'
      IF (N .LE. 2) THEN
        X1 = R(1,1)
        X2 = R(2,1)
        R(1,1) = X1 + X2
        R(2,1) = X1 - X2
        RETURN
      ENDIF
      IW1 = N/2+1
      CALL RFFTB1 (N,R,WSAVE(IW1),WSAVE)
      RETURN
      END
      SUBROUTINE RFFTB1 (N,X,XH,W)
C***BEGIN PROLOGUE  RFFTB1
c   revised by Jim larsen 4/9/99
C***REFER TO RFFTB
C***ROUTINES CALLED  CFFTB
C***END PROLOGUE  RFFTB1
      DIMENSION       X(2,*)     ,XH(2,*)    ,W(*)
C***FIRST EXECUTABLE STATEMENT  RFFTB1
      NS2 = N/2
      NS2P2 = NS2+2
      NQ = NS2/2
      IPAR = NS2-NQ-NQ
      NQM = NQ
      IF (IPAR .EQ. 0) NQM = NQM-1
      NQP = NQM+1
      X1 = X(1,1)
      X2 = X(2,1)
      X(1,1) = X1 + X2
      X(2,1) = X1 - X2
      IF (IPAR .EQ. 0) THEN
        NQP1 = NQP+1
        X1 = X(1,NQP1)
        X2 = X(2,NQP1)
        X(1,NQP1) = X1 + X1
        X(2,NQP1) = X2 + X2
      ENDIF
      IF (NQP .GE. 2) THEN
        DO K=2,NQP
          KC = NS2P2-K
          W1 = W(K-1)
          W2 = W(KC-1)
          X1 = X(1,K)
          X2 = X(2,K)
          X3 = X(1,KC)
          X4 = X(2,KC)
          S1 =  X1 - X3
          S2 = -X2 - X4
          Y1 = X1 + X3
          Y2 = X4 - X2
          Y3 = W1* S1 - W2* S2
          Y4 = W1* S2 + W2* S1
          XH(1,K)  = Y1
          XH(2,K)  = Y2
          XH(1,KC) = Y3
          XH(2,KC) = Y4
          X(1,K)  = Y1 - Y4 
          X(2,K)  = Y3 + Y2
          X(1,KC) = Y1 + Y4 
          X(2,KC) = Y3 - Y2
        ENDDO
      ENDIF
      CALL CFFTB (NS2,X,XH)
      RETURN
      END
      SUBROUTINE RFFTF (N,R,WSAVE)
C***BEGIN PROLOGUE  RFFTF
c   revised by Jim larsen 4/2/99
C***CATEGORY NO.  D6
C***KEYWORDS FFT,FAST FOURIER TRANSFORM,FOURIER TRANSFORM
C***DATE WRITTEN  FEBRUARY 1978
C***AUTHOR  SWARZTRAUBER P.N. (NCAR)
C***PURPOSE
C  FORWARD FFT OF A REAL PERIODIC SEQUENCE
C***DESCRIPTION
C
C     SUBROUTINE RFFTF COMPUTES THE FOURIER COEFFICIENTS OF A REAL
C     PERODIC SEQUENCE (FOURIER ANALYSIS). THE TRANSFORM IS DEFINED
C     BELOW AT OUTPUT PARAMETER R.
C
C     INPUT PARAMETERS
C
C     N       THE LENGTH OF THE ARRAY R. N MUST BE EVEN AND THE METHOD
C             IS MOST EFFICIENT WHEN N IS A PRODUCT OF SMALL PRIMES. N
C             MAY CHANGE SO LONG AS DIFFERENT WORK ARRAYS ARE PROVIDED
C
C     R       A REAL ARRAY OF LENGTH N WHICH CONTAINS THE SEQUENCE
C             TO BE TRANSFORMED
C
C     WSAVE   A WORK ARRAY WHICH MUST BE DIMENSIONED AT LEAST 2.5*N+15
C             IN THE PROGRAM THAT CALLS RFFTF. THE WSAVE ARRAY MUST BE
C             INITIALIZED BY CALLING SUBROUTINE RFFTI(N,WSAVE) AND A
C             DIFFERENT WSAVE ARRAY MUST BE USED FOR EACH DIFFERENT
C             VALUE OF N. THIS INITIALIZATION DOES NOT HAVE TO BE
C             REPEATED SO LONG AS N REMAINS UNCHANGED THUS SUBSEQUENT
C             TRANSFORMS CAN BE OBTAINED FASTER THAN THE FIRST.
C             THE SAME WSAVE ARRAY CAN BE USED BY RFFTF AND RFFTB.
C
C
C     OUTPUT PARAMETERS
C
C     R       FOR K=2,...,N/2
C
C                  R(2*K-1)= THE SUM FROM I=1 TO I=N OF
C
C                       2.*R(I)*COS((K-1)*(I-1)*2*PI/N)
C
C                  R(2*K)= THE SUM FROM I=1 TO I=N OF
C
C                       2.*R(I)*SIN((K-1)*(I-1)*2*PI/N)
C
C             ALSO
C
C                  R(1)= THE SUM FROM I=1 TO I=N OF 2.*R(I)
C
C                  R(2)= THE SUM FROM I=1 TO I=N OF 2.*(-1)**(I+1)*R(I)
C
C      *****  NOTE
C                  THIS TRANSFORM IS UNNORMALIZED SINCE A CALL OF RFFTF
C                  FOLLOWED BY A CALL OF RFFTB WILL MULTIPLY THE INPUT
C                  SEQUENCE BY 2*N.
C
C     WSAVE   CONTAINS RESULTS WHICH MUST NOT BE DESTROYED BETWEEN
C             CALLS OF RFFTF OR RFFTB.
C
C
C***REFERENCES
C***ROUTINES CALLED  RFFTF1,XERROR
C***END PROLOGUE  RFFTF
      DIMENSION       R(2,*)     ,WSAVE(*)
C***FIRST EXECUTABLE STATEMENT  RFFTF
      IF(MOD(N,2).NE.0)PRINT *, '23HRFFTF  ---N IS NOT EVEN,23,2,1'
      IF (N .LE. 2) THEN
        X1 = R(1,1)
        X2 = R(2,1)
        X1 = X1 + X1
        X2 = X2 + X2
        R(1,1) = X1 + X2
        R(2,1) = X1 - X2
        RETURN
      ENDIF
      IW1 = N/2+1
      CALL RFFTF1 (N,R,WSAVE(IW1),WSAVE)
      RETURN
      END
      SUBROUTINE RFFTF1 (N,X,XH,W)
C***BEGIN PROLOGUE  RFFTF1
c   revised by Jim larsen 4/9/99
C***REFER TO RFFTF
C***ROUTINES CALLED  CFFTF
C***END PROLOGUE  RFFTF1
      DIMENSION       X(2,*)     ,XH(2,*)    ,W(*)
C***FIRST EXECUTABLE STATEMENT  RFFTF1
      NS2 = N/2
      NS2P2 = NS2+2
      NQ = NS2/2
      IPAR = NS2-NQ-NQ
      NQM = NQ
      IF (IPAR .EQ. 0) NQM = NQM-1
      NQP = NQM+1
      CALL CFFTF (NS2,X,XH)
      IF (NQP .GE. 2) THEN
        DO K=2,NQP
          KC = NS2P2-K
          W1 = W(K-1)
          W2 = W(KC-1)
          X1 = X(1,K)
          X2 = X(2,K)
          X3 = X(1,KC)
          X4 = X(2,KC)
          S1 = X2 + X4
          S2 = X3 - X1
          Y1 = X1 + X3
          Y2 = X2 - X4
          Y3 = W1* S1 + W2* S2
          Y4 = W1* S2 - W2* S1
          XH(1,K)  = Y1
          XH(2,K)  = Y2
          XH(1,KC) = Y3
          XH(2,KC) = Y4
          X(1,K)  =  Y1 + Y3
          X(2,K)  = -Y2 - Y4
          X(1,KC) =  Y1 - Y3
          X(2,KC) =  Y2 - Y4
        ENDDO
        IF (IPAR .EQ. 0) THEN
          NQP1 = NQP+1
          X1 = X(1,NQP1)
          X2 = X(2,NQP1)
          X(1,NQP1) = X1 + X1
          X(2,NQP1) = X2 + X2
        ENDIF
      ELSE
        NQP1 = NQP+1
        X1 = X(1,NQP1)
        X2 = X(2,NQP1)
        X(1,NQP1) = X1 + X1
        X(2,NQP1) = X2 + X2
      ENDIF
      X1 = X(1,1)
      X2 = X(2,1)
      X(1,1) = X1 + X1 + X2 + X2
      X(2,1) = X1 + X1 - X2 - X2
      RETURN
      END
      SUBROUTINE RFFTI (N,WSAVE)
C***BEGIN PROLOGUE  RFFTI
c   revised by Jim larsen 4/2/99
C***CATEGORY NO.  D6
C***KEYWORDS FFT,FAST FOURIER TRANSFORM,FOURIER TRANSFORM
C***DATE WRITTEN  FEBRUARY 1978
C***AUTHOR  SWARZTRAUBER P.N. (NCAR)
C***PURPOSE
C   INITIALIZE FOR RFFTF AND RFFTB
C***DESCRIPTION
C
C     SUBROUTINE RFFTI INITIALIZES THE ARRAY WSAVE WHICH IS USED IN
C     BOTH RFFTF AND RFFTB. THE PRIME FACTORIZATION OF N TOGETHER WITH
C     A TABULATION OF THE TRIGONOMETRIC FUNCTIONS ARE COMPUTED AND
C     STORED IN WSAVE.
C
C     INPUT PARAMETER
C
C     N       THE LENGTH OF THE SEQUENCE TO BE TRANSFORMED. N MUST BE
C             EVEN
C
C     OUTPUT PARAMETER
C
C     WSAVE   A WORK ARRAY WHICH MUST BE DIMENSIONED AT LEAST 2.5*N+15.
C             THE SAME WORK ARRAY CAN BE USED FOR BOTH RFFTF AND RFFTB
C             AS LONG AS N REMAINS UNCHANGED. DIFFERENT WSAVE ARRAYS
C             ARE REQUIRED FOR DIFFERENT VALUES OF N. THE CONTENTS OF
C             WSAVE MUST NOT BE CHANGED BETWEEN CALLS OF RFFTF OR RFFTB
C
C***REFERENCES
C***ROUTINES CALLED  CFFTI
C***END PROLOGUE  RFFTI
      DIMENSION       WSAVE(*)
C***FIRST EXECUTABLE STATEMENT  RFFTI
      IF(MOD(N,2).NE.0)PRINT *, '23HRFFTI  ---N IS NOT EVEN,23,2,1'

      N = 2* (N/2)  ! N must be even

      NS2 = N/2
      NQM = (NS2-1)/2
      DT = 2.0* 3.141592654/ REAL(N)
      DC = COS(DT)
      DS = SIN(DT)
      WSAVE(1) = DC
      WSAVE(NS2-1) = DS
      IF (NQM .GE. 2) THEN
        DO K=2,NQM
          KC = NS2-K
          W1 = WSAVE(K-1)
          W2 = WSAVE(KC+1)
          WSAVE(K)  = DC* W1 - DS* W2
          WSAVE(KC) = DS* W1 + DC* W2
        ENDDO
      ENDIF
      IW1 = NS2+1
      CALL CFFTI (NS2,WSAVE(IW1))
      RETURN
      END
