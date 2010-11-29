      SUBROUTINE HEAP2_V (RA, BAD, N)

C  Ansley Manke
C  From Numerical Recipes, Heapsort subroutine SORT2
C  with change noted below for the case where N=1

C Return only the data not the indices.

      INTEGER N
      REAL RA(*), BAD

      INTEGER L, IR, J, I, NGD
      REAL RRA

* From SHELLR
* since the input array may be large but sparse, begin by sorting
* the missing value flags to the end of the list

      ngd = 0
      DO j = 1, n
         IF (RA(j) .NE. bad) THEN
            ngd = ngd + 1
            RA(ngd) = RA(j)
          ENDIF 
      END DO

      DO j = ngd+1, n
         RA(j) = bad
      END DO

      L = NGD/2 + 1
      IR = NGD

      IF (NGD .LT. 1) RETURN

   10 CONTINUE
         IF (L .GT. 1) THEN
            L = L - 1
            RRA = RA(L)
         ELSE
            RRA = RA(IR)
            RA(IR) = RA(1)
            IR = IR - 1
            IF (IR .LE. 1) THEN	! *acm* Num Recipes has .EQ. here,
				!       but this fails in the case where	
				!       the loop starts with L=1
               RA(1) = RRA
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
                I = J
                J = J + J
             ELSE
                J = IR + 1
             ENDIF
             GO TO 20
          ENDIF

          RA(I) = RRA
      GO TO 10

      END
