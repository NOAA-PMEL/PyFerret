      SUBROUTINE HEAP2 (RA, IB, BAD, N)

      INTEGER N
      REAL RA(*), IB(*), BAD

      INTEGER L, L, IR, J, I, NGD
      REAL RRA, IIB

* From SHELLR
* since the input array may be large but sparse, begin by sorting
* the missing value flags to the end of the list
        ngd = 0
        DO j = 1, n
           IF (RA(j) .NE. bad) THEN
              ngd = ngd + 1
              RA(ngd) = RA(j)
              IB(ngd) = IB(j)
           ENDIF 
        END DO

        DO j = ngd+1, n
           RA(j) = bad
           IB(j) = 0.0  ! just for house-keeping
        END DO


      L = NGD/2 + 1
      IR = NGD

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
            IF (IR .LE. 1) THEN
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
