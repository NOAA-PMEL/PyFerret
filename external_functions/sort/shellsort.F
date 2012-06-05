      SUBROUTINE SHELL(SDAT, IDAT,N)


*from http://www.cs.mcgill.ca/~ratzer/progs15_6.html  23-APR-1998
* Changed data to REAL. Added IDAT, to sort along with SDAT.  
* Removed NCOUNTS, NCOMP, NSWAP.

      REAL     SDAT(*)
      INTEGER  IDAT(*), N, M
      INTEGER  I, J
      M=N
      DO WHILE (M .gt. 1)
         M=(M+2)/3
         DO I=M+1,N
            DO J=I,M+1,-M
               IF(SDAT(J-M) .ge. SDAT(J)) then
                 CALL SWAPR(SDAT,J,J-M)
                 CALL SWAPI(IDAT,J,J-M)
               endif
            END DO 
         END DO 
      END DO
      RETURN
      END !SUBROUTINE SHELL
!

      SUBROUTINE SWAPI(IDAT,K,L)
      INTEGER IDAT(*),K,L
      INTEGER M
      M=IDAT(K)
      IDAT(K)=IDAT(L)
      IDAT(L)=M
      RETURN
      END !SUROUTINE SWAPI

      SUBROUTINE SWAPR(SDAT,K,L)
      REAL SDAT(*), S
      INTEGER K,L
      S=SDAT(K)
      SDAT(K)=SDAT(L)
      SDAT(L)=S
      RETURN
      END !SUROUTINE SWAP
