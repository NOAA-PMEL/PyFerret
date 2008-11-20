      SUBROUTINE HEAP2_STR (RA, IB, N)

C  Ansley Manke
C  From Numerical Recipes, Heapsort subroutine SORT2

      INTEGER N
      CHARACTER*512 RA(*), RRA, RRB

      INTEGER L, IR, J, I, NGD
      REAL IB(*), IIB

* From SHELLR

      L = N/2 + 1
      IR = N

   10 CONTINUE
         IF (L .GT. 1) THEN
            L = L - 1
            RRA = RA(L)
            IIB = IB(L)
         ELSE
            RRA = RA(IR)
            IIB = IB(IR) 
            RA(IR) = RA(1)
            IB(IR) = IB(1)
            IR = IR - 1
            IF (IR .LE. 1) THEN	! *acm* Num Recipes has .EQ. here,
				!       but this fails in the case where	
				!       the loop starts with L=1
               RA(1) = RRA
               IB(1) = IIB
               RETURN
            ENDIF
          ENDIF

          I = L
          J = L + L
   20     IF (J .LE. IR) THEN
             IF (J .LT. IR) THEN
                IF (RA(J) .LT. RA(J+1)) J = J + 1
             ENDIF
             IF (RRA .LT. RA(J)) THEN
                RA(I) = RA(J)
                IB(I) = IB(J)
                I = J
                J = J + J
             ELSE
                J = IR + 1
             ENDIF
             GO TO 20
          ENDIF

          RA(I) = RRA
          IB(I) = IIB
      GO TO 10

      END
