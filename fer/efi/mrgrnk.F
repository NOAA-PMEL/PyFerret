      Subroutine MRGRNK (XVALT, IRNGT, JWRKT, NVAL)

!  From http://www.fortran-2000.com/rank/index.html#1.1  
! ACM conversion to f77
! __________________________________________________________
!   MRGRNK = Merge-sort ranking of an array
!   For performance reasons, the first 2 passes are taken
!   out of the standard loop, and use dedicated coding.
! __________________________________________________________
      Real XVALT(*)
      Integer IRNGT(*)
! __________________________________________________________
      Integer  JWRKT(*)	! same size as IRNGT
      Integer  LMTNA, LMTNC, IRNG1, IRNG2
      Integer  NVAL, IIND, IWRKD, IWRK, IWRKF, JINDA, IINDA, IINDB
      Real XVALA, XVALB

      INTEGER ii, indx
!

      IF (NVAL .EQ. 0) THEN
         Return
      ELSE IF (nval .EQ. 1) THEN
         IRNGT (1) = 1
         Return
      ENDIF
!
!  Fill-in the index array, creating ordered couples
!
      Do IIND = 2, NVAL, 2
         If (XVALT(IIND-1) .LE. XVALT(IIND)) Then
            IRNGT (IIND-1) = IIND - 1
            IRNGT (IIND) = IIND
         Else
            IRNGT (IIND-1) = IIND
            IRNGT (IIND) = IIND - 1
         End If
      End Do
      If (Mod(NVAL, 2) .NE. 0) Then
         IRNGT (NVAL) = NVAL
      End If
!
!  We will now have ordered subsets A - B - A - B - ...
!  and merge A and B couples into     C   -   C   - ...
!
      LMTNA = 2
      LMTNC = 4
!
!  First iteration. The length of the ordered subsets goes from 2 to 4
!
      Do WHILE (.TRUE.)
         If (NVAL .LE. 2) GO TO 3000
!
!   Loop on merges of A and B into C
!
         Do IWRKD = 0, NVAL - 1, 4
            If ((IWRKD+4) .GT. NVAL) Then
               If ((IWRKD+2) .GE. NVAL) GO TO 2000
!
!   1 2 3
!
               If (XVALT(IRNGT(IWRKD+2)) .LE. XVALT(IRNGT(IWRKD+3))) 
     .             GO TO 2000
!
!   1 3 2
!
               If (XVALT(IRNGT(IWRKD+1)) .LE. XVALT(IRNGT(IWRKD+3)))Then
                  IRNG2 = IRNGT (IWRKD+2)
                  IRNGT (IWRKD+2) = IRNGT (IWRKD+3)
                  IRNGT (IWRKD+3) = IRNG2
!
!   3 1 2
!
               Else
                  IRNG1 = IRNGT (IWRKD+1)
                  IRNGT (IWRKD+1) = IRNGT (IWRKD+3)
                  IRNGT (IWRKD+3) = IRNGT (IWRKD+2)
                  IRNGT (IWRKD+2) = IRNG1
               End If
               GO TO 2000
            End If
!
!   1 2 3 4
!

            If (XVALT(IRNGT(IWRKD+2)) .LE. XVALT(IRNGT(IWRKD+3))) 
     .            GO TO 1000
!
!   1 3 x x
!
            If (XVALT(IRNGT(IWRKD+1)) .LE. XVALT(IRNGT(IWRKD+3))) Then
               IRNG2 = IRNGT (IWRKD+2)
               IRNGT (IWRKD+2) = IRNGT (IWRKD+3)
               If (XVALT(IRNG2) .LE. XVALT(IRNGT(IWRKD+4))) Then
!   1 3 2 4
                  IRNGT (IWRKD+3) = IRNG2
               Else
!   1 3 4 2
                  IRNGT (IWRKD+3) = IRNGT (IWRKD+4)
                  IRNGT (IWRKD+4) = IRNG2
               End If
!
!   3 x x x
!
            Else
               IRNG1 = IRNGT (IWRKD+1)
               IRNG2 = IRNGT (IWRKD+2)
               IRNGT (IWRKD+1) = IRNGT (IWRKD+3)
               If (XVALT(IRNG1) .LE. XVALT(IRNGT(IWRKD+4))) Then
                  IRNGT (IWRKD+2) = IRNG1
                  If (XVALT(IRNG2) .LE. XVALT(IRNGT(IWRKD+4))) Then
!   3 1 2 4
                     IRNGT (IWRKD+3) = IRNG2
                  Else
!   3 1 4 2
                     IRNGT (IWRKD+3) = IRNGT (IWRKD+4)
                     IRNGT (IWRKD+4) = IRNG2
                  End If
               Else
!   3 4 1 2
                  IRNGT (IWRKD+2) = IRNGT (IWRKD+4)
                  IRNGT (IWRKD+3) = IRNG1
                  IRNGT (IWRKD+4) = IRNG2
               End If
            End If
 1000       CONTINUE
         End Do
 2000    CONTINUE
!
!  The Cs become As and Bs
!
         LMTNA = 4
         GO TO 3000
      End Do
 3000 CONTINUE

!
!  Iteration loop. Each time, the length of the ordered subsets
!  is doubled.
!
      Do WHILE (.TRUE.)
         If (LMTNA .GE. NVAL) GO TO 6000
         IWRKF = 0
         LMTNC = 2 * LMTNC
!
!   Loop on merges of A and B into C
!
         Do WHILE (.TRUE.)
 7000       CONTINUE
            IWRK = IWRKF
            IWRKD = IWRKF + 1
            JINDA = IWRKF + LMTNA
            IWRKF = IWRKF + LMTNC
            If (IWRKF .GE. NVAL) Then
               If (JINDA .GE. NVAL) GO TO 5000
               IWRKF = NVAL
            End If
            IINDA = 1
            IINDB = JINDA + 1
!
!   Shortcut for the case when the max of A is smaller
!   than the min of B. This line may be activated when the
!   initial set is already close to sorted.
!
!          IF (XVALT(IRNGT(JINDA)) .LE. XVALT(IRNGT(IINDB))) GO TO 7000
!
!  One steps in the C subset, that we build in the final rank array
!
!  Make a copy of the rank array for the merge iteration
!
            indx = iwrkd
            DO ii = 1, lmtna
              JWRKT (ii) = IRNGT (indx)
              indx = indx + 1
            ENDDO
!
            XVALA = XVALT (JWRKT(IINDA))
            XVALB = XVALT (IRNGT(IINDB))
!
            Do WHILE (.TRUE.)
               IWRK = IWRK + 1
!
!  We still have unprocessed values in both A and B
!
               If (XVALA .GT. XVALB) Then
                  IRNGT (IWRK) = IRNGT (IINDB)
                  IINDB = IINDB + 1
                  If (IINDB .GT. IWRKF) Then
!  Only A still with unprocessed values
                     indx = iinda
                     DO ii = IWRK+1, IWRKF
                        IRNGT (ii) = JWRKT (indx)
                        indx = indx + 1
                     ENDDO

                     GO TO 4000
                  End If
                  XVALB = XVALT (IRNGT(IINDB))
               Else
                  IRNGT (IWRK) = JWRKT (IINDA)
                  IINDA = IINDA + 1
                  If (IINDA .GT. LMTNA) GO TO 4000 ! Only B still with unprocessed values
                  XVALA = XVALT (JWRKT(IINDA))
               End If
!
            End Do
 4000       CONTINUE
         End Do
 5000    CONTINUE
!
!  The Cs become As and Bs
!
         LMTNA = 2 * LMTNA
      End Do
 6000 CONTINUE
!
      Return
      End	!Subroutine MRGRNK
