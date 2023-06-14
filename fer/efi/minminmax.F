
      SUBROUTINE minMINMAX  (Z, L, MM, NN, neigh_x, neigh_y, badz, 
     .                      zminlist, nzmin)
C
C  based on NCAR routine MINMAX, return minima
C  From the NCAR graphics library at http://ngwww.ucar.edu version 4.2
C  Modified by Ansley Manke to regurn only the minima for subroutine findlo.F
C
C-------------------------------------------------------------
C                                                                      
C                Copyright (C)  2000
C        University Corporation for Atmospheric Research
C                All Rights Reserved
C
C This file is free software; you can redistribute it and/or modify
C it under the terms of the GNU General Public License as published
C by the Free Software Foundation; either version 2 of the License, or
C (at your option) any later version.
C
C This software is distributed in the hope that it will be useful, but
C WITHOUT ANY WARRANTY; without even the implied warranty of
C MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
C General Public License for more details.
C
C You should have received a copy of the GNU General Public License
C along with this software; if not, write to the Free Software
C Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
C USA.
C-------------------------------------------------------------

C THIS ROUTINE FINDS RELATIVE MINIMUMS AND MAXIMUMS.  A RELATIVE MINIMUM
C (OR MAXIMUM) IS DEFINED TO BE THE LOWEST (OR HIGHEST) POINT WITHIN
C A CERTAIN NEIGHBORHOOD OF THE POINT.  THE NEIGHBORHOOD USED HERE
C IS + OR - MN IN THE X DIRECTION AND + OR - NM IN THE Y DIRECTION.
C
C ORIGINATOR       DAVID KENNISON

C  Ansley Manke changes for Ferret 7/01
C	- remove sections that put label on plot
C	- use "bad value" rather than SPECIAL VALUE, SPVAL
C	- remove references to IOFFP, JOFFDT
c	- new code and my comments in lowercase
c	- return mins.  For n = 1, nzmin:
c	    zminlist(1,n) = i index of zmin point
c	    zminlist(2,n) = j index of zmin point
c	    zminlist(3,n) = z value at zmin point
* 11/2022 *acm* Code cleanup: 
*              - Fix loops with DO termination statement which is not 
*                  END DO or CONTINUE, and loops that share termination statements
*              - Rework arithmetic IF statements 
*              - GOTO statements remain. Overall organization of the routine unchanged.
C
      IMPLICIT NONE
      integer	L, MM, NN, m, n, nm1, mm1, jp, im, mn, ip, is, it, ii,
     .		js, nm, jt, jk, ik, nzmin, neigh_x, neigh_y

      real	Z(L,NN), aa, an, badz, zminlist(3,*)
C
      M = MM
      N = NN

c      MN = min(2,MAX(2,IFIX(FLOAT(M)/8.)))
c      NM = min(2,MAX(2,IFIX(FLOAT(N)/8.)))

      MN = neigh_x
      NM = neigh_y

      nzmin = 0
C
      NM1 = N-1
      MM1 = M-1
C
C LINE LOOP FOLLOWS - THE COMPLETE TWO-DIMENSIONAL TEST FOR A MINIMUM OR
C MAXIMUM OF THE FIELD IS ONLY PERFORMED FOR POINTS WHICH ARE MINIMA OR
C MAXIMA ALONG SOME LINE - FINDING THESE CANDIDATES IS MADE EFFICIENT BY
C USING A COUNT OF CONSECUTIVE INCREASES OR DECREASES OF THE FUNCTION
C ALONG THE LINE
C
      DO 127 JP=2,NM1
C
         IM = MN-1
         IP = -1
         GO TO 126
C
C CONTROL RETURNS TO STATEMENT 10 (101?) AS LONG AS THE FUNCTION IS INCREASING
C ALONG THE LINE - WE SEEK A POSSIBLE MAXIMUM
C

  101    IP = IP+1
         AA = AN
         IF (IP .EQ. MM1) GO TO 104
         AN = Z(IP+1,JP)
         IF (AN.EQ.badz) GO TO 125
         IF (AA .LT. AN) THEN
	    IM = IM+1
	    GO TO 101
	 ELSE IF (AA .EQ. AN) THEN
	    GOTO 103
	 ELSE
	    GOTO 104
	 ENDIF
  103    IM = 0
	 GO TO 101

C
C FUNCTION DECREASED - TEST FOR MAXIMUM ON LINE
C

  104    CONTINUE

	 IF (IM .LT. MN) THEN
            IS = max(1,IP-MN)
            IT = IP-IM-1
            IF (IS .LE. IT) THEN
               DO II=IS,IT
                  IF (AA .LE. Z(II,JP)) GO TO 112
               ENDDO
	    ENDIF
	 ENDIF

	 IS = IP+2
         IT = min(M,IP+MN)
         IF (IS .LE. IT) THEN
            DO II=IS,IT
               IF (Z(II,JP).EQ.badz) THEN
                 IP = II-1
                 GO TO 125
	       ENDIF
               IF (AA .LE. Z(II,JP)) GO TO 112
            ENDDO

	 ENDIF
C
C WE HAVE MAXIMUM ON LINE - DO TWO-DIMENSIONAL TEST FOR MAXIMUM OF FIELD
C
	 JS = max(1,JP-NM)
         JT = min(N,JP+NM)
         IS = max(1,IP-MN)
         IT = min(M,IP+MN)
         DO JK=JS,JT
            IF (JK .EQ. JP) CYCLE
            DO IK=IS,IT
               IF (Z(IK,JK).GE.AA .OR. (Z(IK,JK).EQ.badz)) GO TO 112
            ENDDO
	 ENDDO

  112    CONTINUE

	 IM = 1
         IF (IP .GE. MM1) GOTO 127
C
C CONTROL RETURNS TO STATEMENT 20 (113?) AS LONG AS THE FUNCTION IS DECREASING
C ALONG THE LINE - WE SEEK A POSSIBLE MINIMUM
C
  113    CONTINUE

	 IP = IP+1
         AA = AN
         IF (IP .EQ. MM1) GO TO 116
         AN = Z(IP+1,JP)
         IF (AN.EQ.badz) GO TO 125
         IF (AA .LT. AN) GOTO 116
	 IF (AA .EQ. AN) IM = 0
	 IF (AA .GT. AN) IM = IM+1
         GO TO 113
C
C FUNCTION INCREASED - TEST FOR MINIMUM ON LINE
C
  116    CONTINUE

	 IF (IM .LT. MN) THEN
            IS = max(1,IP-MN)
            IT = IP-IM-1
            IF (IS .LE. IT) THEN
               DO II=IS,IT
                  IF (AA .GE. Z(II,JP)) GO TO 124
               ENDDO
	    ENDIF
	 ENDIF

         IS = IP+2
         IT = min(M,IP+MN)
         IF (IS .GT. IT) GO TO 121
         DO II=IS,IT
            IF (Z(II,JP).EQ.badz) THEN
               IP = II-1
               GO TO 125
	    ENDIF
            IF (AA .GE. Z(II,JP)) GO TO 124
         ENDDO
C
C WE HAVE MINIMUM ON LINE - DO TWO-DIMENSIONAL TEST FOR MINIMUM OF FIELD
C
  121    CONTINUE
	 JS = max(1,JP-NM)
         JT = min(N,JP+NM)
         IS = max(1,IP-MN)
         IT = min(M,IP+MN)
         DO JK=JS,JT
            IF (JK .EQ. JP) CYCLE
            DO IK=IS,IT
               IF (Z(IK,JK).LE.AA .OR. (Z(IK,JK).EQ.badz)) GO TO 124
            ENDDO
         ENDDO
C
	 nzmin = nzmin + 1
	 zminlist(1,nzmin) = float(ip)
	 zminlist(2,nzmin) = float(jp)
	 zminlist(3,nzmin) = z(ip,jp)

  124    CONTINUE
	 IM = 1
         IF (IP .LT. MM1) GOTO 101
	 GOTO 127
C
C SKIP SPECIAL VALUES ON LINE
C
  125    CONTINUE
         IM = 0

  126    CONTINUE
         IP = IP+1
         IF (IP .GE. MM1) GO TO 127
         IF (Z(IP+1,JP).EQ.badz) GO TO 125
         IM = IM+1
         IF (IM .LE. MN) GO TO 126
         IM = 1
         AN = Z(IP+1,JP)
         IF (Z(IP,JP) .LT. AN) GOTO 101
         IF (Z(IP,JP) .EQ. AN) GOTO 103
         IF (Z(IP,JP) .GT. AN) GOTO 113
C
  127 CONTINUE
C
      RETURN

      END
