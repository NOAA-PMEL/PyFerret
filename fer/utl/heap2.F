      SUBROUTINE HEAP2 (RA, IB, BAD, N)
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
C  Ansley Manke
C  From Numerical Recipes, Heapsort subroutine SORT2
C  with change noted below for the case where N=1
* V6.74 3/12 *acm* cleanup, add disclaimer

      INTEGER N
      REAL RA(*), IB(*), BAD

      INTEGER L, IR, J, I, NGD
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

      IF (NGD .LT. 1) RETURN

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
