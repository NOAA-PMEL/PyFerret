      REAL FUNCTION RANDU(JD)

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
*
* modified by steve hankin to contain explicit declarations
* and renamed RANDU from "UNI" in its original version from CMLIB
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

        IMPLICIT NONE
	INTEGER	JD,MDIG,I1MACH,M1,M2,JSEED,K0,K1,J0,J1,I,J,K


C***BEGIN PROLOGUE  UNI
C***DATE WRITTEN   810915
C***REVISION DATE  830805
C***CATEGORY NO.  L6A21
C***KEYWORDS  RANDOM NUMBERS, UNIFORM RANDOM NUMBERS
C***AUTHOR    BLUE, JAMES, SCIENTIFIC COMPUTING DIVISION, NBS
C             KAHANER, DAVID, SCIENTIFIC COMPUTING DIVISION, NBS
C             MARSAGLIA, GEORGE, COMPUTER SCIENCE DEPT., WASH STATE UNIV
C
C***PURPOSE  THIS ROUTINE GENERATES QUASI UNIFORM RANDOM NUMBERS ON [0,1
C             AND CAN BE USED ON ANY COMPUTER WITH WHICH ALLOWS INTEGERS
C             AT LEAST AS LARGE AS 32767.
C***DESCRIPTION
C
C       THIS ROUTINE GENERATES QUASI UNIFORM RANDOM NUMBERS ON THE INTER
C       [0,1).  IT CAN BE USED WITH ANY COMPUTER WHICH ALLOWS
C       INTEGERS AT LEAST AS LARGE AS 32767.
C
C
C   USE
C       FIRST TIME....
C                   Z = UNI(JD)
C                     HERE JD IS ANY  N O N - Z E R O  INTEGER.
C                     THIS CAUSES INITIALIZATION OF THE PROGRAM
C                     AND THE FIRST RANDOM NUMBER TO BE RETURNED AS Z.
C       SUBSEQUENT TIMES...
C                   Z = UNI(0)
C                     CAUSES THE NEXT RANDOM NUMBER TO BE RETURNED AS Z.
C
C
C..................................................................
C   NOTE: USERS WHO WISH TO TRANSPORT THIS PROGRAM FROM ONE COMPUTER
C         TO ANOTHER SHOULD READ THE FOLLOWING INFORMATION.....
C
C   MACHINE DEPENDENCIES...
C      MDIG = A LOWER BOUND ON THE NUMBER OF BINARY DIGITS AVAILABLE
C              FOR REPRESENTING INTEGERS, INCLUDING THE SIGN BIT.
C              THIS VALUE MUST BE AT LEAST 16, BUT MAY BE INCREASED
C              IN LINE WITH REMARK A BELOW.
C
C   REMARKS...
C     A. THIS PROGRAM CAN BE USED IN TWO WAYS:
C        (1) TO OBTAIN REPEATABLE RESULTS ON DIFFERENT COMPUTERS,
C            SET 'MDIG' TO THE SMALLEST OF ITS VALUES ON EACH, OR,
C        (2) TO ALLOW THE LONGEST SEQUENCE OF RANDOM NUMBERS TO BE
C            GENERATED WITHOUT CYCLING (REPEATING) SET 'MDIG' TO THE
C            LARGEST POSSIBLE VALUE.
C     B. THE SEQUENCE OF NUMBERS GENERATED DEPENDS ON THE INITIAL
C          INPUT 'JD' AS WELL AS THE VALUE OF 'MDIG'.
C          IF MDIG=16 ONE SHOULD FIND THAT
C            THE FIRST EVALUATION
C              Z=UNI(305) GIVES Z=.027832881...
C            THE SECOND EVALUATION
C              Z=UNI(0) GIVES   Z=.56102176...
C            THE THIRD EVALUATION
C              Z=UNI(0) GIVES   Z=.41456343...
C            THE THOUSANDTH EVALUATION
C              Z=UNI(0) GIVES   Z=.19797357...
C
C***REFERENCES  MARSAGLIA G., "COMMENTS ON THE PERFECT UNIFORM RANDOM
C                 NUMBER GENERATOR", UNPUBLISHED NOTES, WASH S. U.
C***ROUTINES CALLED  I1MACH,XERROR
C***END PROLOGUE  UNI
      INTEGER M(17)
C
      SAVE I,J,M,M1,M2
C
      DATA M(1),M(2),M(3),M(4),M(5),M(6),M(7),M(8),M(9),M(10),M(11),
     1     M(12),M(13),M(14),M(15),M(16),M(17)
     2                   / 30788,23052,2053,19346,10646,19427,23975,
     3                     19049,10949,19693,29746,26748,2796,23890,
     4                     29168,31924,16499 /
      DATA M1,M2,I,J / 32767,256,5,17 /
C***FIRST EXECUTABLE STATEMENT  UNI
      IF(JD .EQ. 0) GO TO 3
C  FILL
      MDIG=I1MACH(8)+1
C          BE SURE THAT MDIG AT LEAST 16...
      IF(MDIG.LT.16)CALL XERROR('UNI--MDIG LESS THAN 16',22,1,2)
      M1= 2**(MDIG-2) + (2**(MDIG-2)-1)
      M2 = 2**(MDIG/2)
      JSEED = MIN0(IABS(JD),M1)
      IF( MOD(JSEED,2).EQ.0 ) JSEED=JSEED-1
      K0 =MOD(9069,M2)
      K1 = 9069/M2
      J0 = MOD(JSEED,M2)
      J1 = JSEED/M2
      DO I=1,17
        JSEED = J0*K0
        J1 = MOD(JSEED/M2+J0*K1+J1*K0,M2/2)
        J0 = MOD(JSEED,M2)
        M(I) = J0+M2*J1
      ENDDO
      I=5
      J=17
C  BEGIN MAIN LOOP HERE
    3 K=M(I)-M(J)
      IF(K .LT. 0) K=K+M1
      M(J)=K
      I=I-1
      IF(I .EQ. 0) I=17
      J=J-1
      IF(J .EQ. 0) J=17
!      UNI=FLOAT(K)/FLOAT(M1)
      RANDU=FLOAT(K)/FLOAT(M1)
      RETURN
      END
