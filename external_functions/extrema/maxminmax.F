
      SUBROUTINE maxMINMAX  (Z, L, MM, NN, neigh_x, neigh_y, badz, 
     .                      zmaxlist, nzmax)
C
C  based on NCAR routine MINMAX, return maxima
C  From the NCAR graphics library at http://ngwww.ucar.edu version 4.2
C  Modified by Ansley Manke to regurn only the maxima for subroutine findhi.F
C-------------------------------------------------------------
C	$Id$
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

CC THIS ROUTINE FINDS RELATIVE MINIMUMS AND MAXIMUMS.  A RELATIVE MINIMUM
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
c	- return maxes.  For n = 1, nzmax:
c	    zmaxlist(1,n) = i index of zmax point
c	    zmaxlist(2,n) = j index of zmax point
c	    zmaxlist(3,n) = z value at zmax point
C
      integer	L, MM, NN, m, n, nm1, mm1, jp, im, mn, ip, is, it, ii,
     .		js, nm, jt, jk, ik, nzmax, neigh_x, neigh_y

      real	Z(L,NN), aa, an, badz, zmaxlist(3,*)
      real tol, q


      tol = 0.
C
      M = MM
      N = NN

c      MN = min(2,MAX(2,IFIX(FLOAT(M)/8.)))
c      NM = min(2,MAX(2,IFIX(FLOAT(N)/8.)))

      MN = neigh_x
      NM = neigh_y

      nzmax = 0
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
C CONTROL RETURNS TO STATEMENT 10 AS LONG AS THE FUNCTION IS INCREASING
C ALONG THE LINE - WE SEEK A POSSIBLE MAXIMUM
C
  101    IP = IP+1
         AA = AN
         IF (IP .EQ. MM1) GO TO 104
         AN = Z(IP+1,JP)
         IF (AN.EQ.badz) GO TO 125
         IF (AA-AN) 102,103,104
  102    IM = IM+1
         GO TO 101
  103    IM = 0
         GO TO 101
C
C FUNCTION DECREASED - TEST FOR MAXIMUM ON LINE
C
  104    IF (IM .GE. MN) GO TO 106
         IS = max(1,IP-MN)
         IT = IP-IM-1
         IF (IS .GT. IT) GO TO 106
         DO 105 II=IS,IT
            q = Z(II,JP) - aa
            IF (AA .LE. Z(II,JP)) GO TO 112
  105    CONTINUE
  106    IS = IP+2
         IT = min(M,IP+MN)
         IF (IS .GT. IT) GO TO 109
         DO 108 II=IS,IT
            IF (Z(II,JP).NE.badz) GO TO 107
            IP = II-1
            GO TO 125
            q = Z(II,JP) - aa
  107       IF (AA .LE. Z(II,JP)) GO TO 112
  108    CONTINUE
C
C WE HAVE MAXIMUM ON LINE - DO TWO-DIMENSIONAL TEST FOR MAXIMUM OF FIELD
C
  109    JS = max(1,JP-NM)
         JT = min(N,JP+NM)
         IS = max(1,IP-MN)
         IT = min(M,IP+MN)
         DO 111 JK=JS,JT
            IF (JK .EQ. JP) GO TO 111
            DO 110 IK=IS,IT
               q = Z(IK,JK)-AA
               IF (Z(IK,JK).GE.AA .OR. (Z(IK,JK).EQ.badz)) GO TO 112
  110       CONTINUE
  111    CONTINUE

	 nzmax = nzmax + 1
	 zmaxlist(1,nzmax) = float(ip)
	 zmaxlist(2,nzmax) = float(jp)
	 zmaxlist(3,nzmax) = z(ip,jp)

  112    IM = 1
         GO TO 127

C
C
C SKIP SPECIAL VALUES ON LINE
C
  125    IM = 0
  126    IP = IP+1
         IF (IP .GE. MM1) GO TO 127
         IF (Z(IP+1,JP).EQ.badz) GO TO 125
         IM = IM+1
         IF (IM .LE. MN) GO TO 126
         IM = 1
         AN = Z(IP+1,JP)
         IF (Z(IP,JP)-AN) 101,103,127
C
  127 CONTINUE
      RETURN

      END
