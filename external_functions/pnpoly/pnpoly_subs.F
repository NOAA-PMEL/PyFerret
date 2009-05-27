
C .................................................................. 

      SUBROUTINE POLY_BOUND_BOX (xv, yv, nv, xvmin, xvmax, yvmin, yvmax)

* Get the bounding box around the polygon. min and max variables
* have been initialized.

      REAL xv(*), yv(*), xvmin, xvmax, yvmin, yvmax
      INTEGER nv
      INTEGER i

      DO 50 i = 1, nv
         xvmin = MIN(xvmin, xv(i))
         xvmax = MAX(xvmax, xv(i))
         yvmin = MIN(yvmin, yv(i))
         yvmax = MAX(yvmax, yv(i))
  50  CONTINUE

      RETURN
      END


C .................................................................. 
	LOGICAL FUNCTION PP_FPEQ ( a, b )
*
*************
* This is a copy of the Ferret function TM_FPEQ. It is included here
* so that the function is a stand-alone function. When and if this
* External Function is statically linked into the Ferret executable,
* We can rename calls to PP_FPEQ back to TM_FPEQ and remove this subroutine.
*************
*
* test to see if REAL*4 floating point numbers are equal to within
* "reasonable" accuracy
* REAL*4 (F_floating) machine error on the VAX is "approximately one part in
* 2**23" or 1.2E-7.  Allow 4.E-7 as machine error after many typical roundoffs.
* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.00 - 11/21/85
* revision 0.10 - 10/23/87 - all variables explicitly declared
*
* calling arguments:
	REAL	a, b

* local parameters
	REAL	   VAX_epsilon
	PARAMETER (VAX_epsilon = 4.E-7)

* treat relative error case (non-zero) separate fron zero case

	IF  (a.NE.0.0   .AND.  b.NE.0.0 )  THEN

	   PP_FPEQ =  ABS(a-b) .LE. ABS(b)*VAX_epsilon		! relative error

	ELSE

	   PP_FPEQ =	ABS(a) .LT. VAX_epsilon
     .		.AND.   ABS(b) .LT. VAX_epsilon			! absolute error

	ENDIF
	RETURN
	END

C .................................................................. 
C 
C SUBROUTINE PNPOLY 
C 
C PURPOSE 
C   TO DETERMINE WHETHER A POINT IS INSIDE A POLYGON 
C 
C USAGE 
C   CALL PNPOLY (PX, PY, XX, YY, N, workx, worky, INOUT ) 
C 
C DESCRIPTION OF THE PARAMETERS 
C   PX    - X-COORDINATE OF POINT IN QUESTION. 
C   PY    - Y-COORDINATE OF POINT IN QUESTION. 
C   XX    - N LONG VECTOR CONTAINING X-COORDINATES OF 
C           VERTICES OF POLYGON. 
C   YY    - N LONG VECTOR CONTAING Y-COORDINATES OF 
C           VERTICES OF POLYGON. 
C   N     - NUMBER OF VERTICES IN THE POLYGON. 
C   workx - work array
C   worky - work array
C   INOUT - THE SIGNAL RETURNED: 
C           -1 IF THE POINT IS OUTSIDE OF THE POLYGON, 
C            0 IF THE POINT IS ON AN EDGE OR AT A VERTEX, 
C            1 IF THE POINT IS INSIDE OF THE POLYGON. 
C 
C REMARKS 
C   THE VERTICES MAY BE LISTED CLOCKWISE OR ANTICLOCKWISE. 
C   THE FIRST MAY OPTIONALLY BE REPEATED, IF SO N MAY 
C   OPTIONALLY BE INCREASED BY 1. 
C   THE INPUT POLYGON MAY BE A COMPOUND POLYGON CONSISTING 
C   OF SEVERAL SEPARATE SUBPOLYGONS. IF SO, THE FIRST VERTEX 
C   OF EACH SUBPOLYGON MUST BE REPEATED, AND WHEN CALCULATING 
C   N, THESE FIRST VERTICES MUST BE COUNTED TWICE. 
C   INOUT IS THE ONLY PARAMETER WHOSE VALUE IS CHANGED. 
C   THE SIZE OF THE ARRAYS MUST BE INCREASED IF N > MAXDIM 
C   WRITTEN BY RANDOLPH FRANKLIN, UNIVERSITY OF OTTAWA, 7/70. 
C   
C SUBROUTINES AND FUNCTION SUBPROGRAMS REQUIRED 
C   NONE 
C 
C METHOD 
C   A VERTICAL LINE IS DRAWN THRU THE POINT IN QUESTION. IF IT 
C   CROSSES THE POLYGON AN ODD NUMBER OF TIMES, THEN THE 
C   POINT IS INSIDE OF THE POLYGON. 
C 
C .................................................................. 
C 
      SUBROUTINE PNPOLY(PX,PY,XX,YY,N,X,Y,INOUT) 
      REAL PX,X(*),Y(*),XX(*),YY(*) 
      INTEGER N, INOUT

      LOGICAL PP_FPEQ, MX,MY,NX,NY, b
      INTEGER I, J, INOUT1, INOUT2
      INOUT1 = 0
      INOUT2 = 0

* Change the original to see if X(J)=Y(J)=0. This loop finds matches 
* for points on the left-hand edges 

    6 DO 1 I=1,N 
      X(I)=XX(I)-PX 
    1 Y(I)=YY(I)-PY 
      INOUT1=-1 
      DO 2 I=1,N 
      J=1+MOD(I,N) 
      MX=X(I).GE.0.0 
      NX=X(J).GE.0.0 
      MY=Y(I).GE.0.0 
      NY=Y(J).GE.0.0 

      IF (PP_FPEQ(X(J),0.) .AND. PP_FPEQ(Y(J),0.)) GOTO 4

      IF(.NOT.((MY.OR.NY).AND.(MX.OR.NX)).OR.(MX.AND.NX)) GO TO 2 
      IF(.NOT.(MY.AND.NY.AND.(MX.OR.NX).AND..NOT.(MX.AND.NX))) GO TO 3 
      INOUT1=-INOUT1 
      GO TO 2
    3 CONTINUE
      IF ( PP_FPEQ(X(J),Y(J)) ) THEN
         IF(PP_FPEQ((Y(I)*X(J)-X(I)*Y(J)),0.)) THEN  
            INOUT1 = 0
            INOUT2 = 0
            GOTO 50
         ENDIF
      ENDIF
      IF( PP_FPEQ((Y(I)*X(J)-X(I)*Y(J))/(X(J)-X(I)), 0.) ) GOTO 4
      IF((Y(I)*X(J)-X(I)*Y(J))/(X(J)-X(I))) 2,4,5 
    4 INOUT1 = 0
      INOUT2 = 0
      GOTO 50
    5 INOUT1=-INOUT1 
    2 CONTINUE 

* Now do the same with the order of subtraction reversed when 
* computing arrays X(I) and Y(I). This finds matches for points 
* on the right-hand edges.

      DO 11 I=1,N 
      X(I)=PX-XX(I)
   11 Y(I)=PY-YY(I) 
      INOUT2=-1 
      DO 12 I=1,N 
      J=1+MOD(I,N) 
      MX=X(I).GE.0.0 
      NX=X(J).GE.0.0 
      MY=Y(I).GE.0.0 
      NY=Y(J).GE.0.0 

      IF (PP_FPEQ(X(J),0.) .AND. PP_FPEQ(Y(J),0.)) GOTO 14

      IF(.NOT.((MY.OR.NY).AND.(MX.OR.NX)).OR.(MX.AND.NX)) GO TO 12 
      IF(.NOT.(MY.AND.NY.AND.(MX.OR.NX).AND..NOT.(MX.AND.NX))) GO TO 13 
      INOUT2=-INOUT2 
      GO TO 12
   13 CONTINUE
      IF ( PP_FPEQ(X(J),Y(J)) ) THEN
         IF(PP_FPEQ(Y(I)*X(J)-X(I)*Y(J), 0.)) THEN  
            INOUT1 = 0
            INOUT2 = 0
            GOTO 50
         ENDIF
      ENDIF
      IF( PP_FPEQ((Y(I)*X(J)-X(I)*Y(J))/(X(J)-X(I)), 0.) ) GOTO 14
      IF((Y(I)*X(J)-X(I)*Y(J))/(X(J)-X(I))) 12,14,15 
   14 INOUT1 = 0
      INOUT2 = 0
      GOTO 50
   15 INOUT2=-INOUT2 
   12 CONTINUE 

* Combine the results: -1 outside the polygon, 0 
* on the edges or vertices, 1 inside the polygon.

   50 CONTINUE
      INOUT = (INOUT2 + INOUT1)/2
      RETURN

      END 

*** ORIG  C >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
*** ORIG  C Here is the exact code from W. Randolph Franklin pages.
*** ORIG  C Fortran Code for the Point in Polygon Test
*** ORIG  
*** ORIG  C>>>PNP1 
*** ORIG  C 
*** ORIG  C .................................................................. 
*** ORIG  C 
*** ORIG  C SUBROUTINE PNPOLY 
*** ORIG  C 
*** ORIG  C PURPOSE 
*** ORIG  C   TO DETERMINE WHETHER A POINT IS INSIDE A POLYGON 
*** ORIG  C 
*** ORIG  C USAGE 
*** ORIG  C   CALL PNPOLY (PX, PY, XX, YY, N, INOUT ) 
*** ORIG  C 
*** ORIG  C DESCRIPTION OF THE PARAMETERS 
*** ORIG  C   PX    - X-COORDINATE OF POINT IN QUESTION. 
*** ORIG  C   PY    - Y-COORDINATE OF POINT IN QUESTION. 
*** ORIG  C   XX    - N LONG VECTOR CONTAINING X-COORDINATES OF 
*** ORIG  C           VERTICES OF POLYGON. 
*** ORIG  C   YY    - N LONG VECTOR CONTAING Y-COORDINATES OF 
*** ORIG  C           VERTICES OF POLYGON. 
*** ORIG  C   N     -  NUMBER OF VERTICES IN THE POLYGON. 
*** ORIG  C   INOUT - THE SIGNAL RETURNED: 
*** ORIG  C           -1 IF THE POINT IS OUTSIDE OF THE POLYGON, 
*** ORIG  C            0 IF THE POINT IS ON AN EDGE OR AT A VERTEX, 
*** ORIG  C            1 IF THE POINT IS INSIDE OF THE POLYGON. 
*** ORIG  C 
*** ORIG  C REMARKS 
*** ORIG  C   THE VERTICES MAY BE LISTED CLOCKWISE OR ANTICLOCKWISE. 
*** ORIG  C   THE FIRST MAY OPTIONALLY BE REPEATED, IF SO N MAY 
*** ORIG  C   OPTIONALLY BE INCREASED BY 1. 
*** ORIG  C   THE INPUT POLYGON MAY BE A COMPOUND POLYGON CONSISTING 
*** ORIG  C   OF SEVERAL SEPARATE SUBPOLYGONS. IF SO, THE FIRST VERTEX 
*** ORIG  C   OF EACH SUBPOLYGON MUST BE REPEATED, AND WHEN CALCULATING 
*** ORIG  C   N, THESE FIRST VERTICES MUST BE COUNTED TWICE. 
*** ORIG  C   INOUT IS THE ONLY PARAMETER WHOSE VALUE IS CHANGED. 
*** ORIG  C   THE SIZE OF THE ARRAYS MUST BE INCREASED IF N > MAXDIM 
*** ORIG  C   WRITTEN BY RANDOLPH FRANKLIN, UNIVERSITY OF OTTAWA, 7/70. 
*** ORIG  C   
*** ORIG  C SUBROUTINES AND FUNCTION SUBPROGRAMS REQUIRED 
*** ORIG  C   NONE 
*** ORIG  C 
*** ORIG  C METHOD 
*** ORIG  C   A VERTICAL LINE IS DRAWN THRU THE POINT IN QUESTION. IF IT 
*** ORIG  C   CROSSES THE POLYGON AN ODD NUMBER OF TIMES, THEN THE 
*** ORIG  C   POINT IS INSIDE OF THE POLYGON. 
*** ORIG  C 
*** ORIG  C .................................................................. 
*** ORIG  C 
*** ORIG        SUBROUTINE PNPOLY(PX,PY,XX,YY,N,INOUT) 
*** ORIG        REAL X(200),Y(200),XX(N),YY(N) 
*** ORIG        LOGICAL MX,MY,NX,NY 
*** ORIG        INTEGER O 
*** ORIG  
*** ORIG  C OUTPUT UNIT FOR PRINTED MESSAGES 
*** ORIG        DATA O/6/ 
*** ORIG        MAXDIM=200 
*** ORIG        IF(N.LE.MAXDIM)GO TO 6 
*** ORIG        WRITE(O,7)  
*** ORIG      7 FORMAT('0WARNING:',I5,' TOO GREAT FOR THIS VERSION OF pt_in_poly. 
*** ORIG       1 RESULTS INVALID')
*** ORIG        RETURN
*** ORIG      6 DO 1 I=1,N 
*** ORIG        X(I)=XX(I)-PX 
*** ORIG      1 Y(I)=YY(I)-PY 
*** ORIG        INOUT=-1 
*** ORIG        DO 2 I=1,N 
*** ORIG        J=1+MOD(I,N) 
*** ORIG        MX=X(I).GE.0.0 
*** ORIG        NX=X(J).GE.0.0 
*** ORIG        MY=Y(I).GE.0.0 
*** ORIG        NY=Y(J).GE.0.0 
*** ORIG        IF(.NOT.((MY.OR.NY).AND.(MX.OR.NX)).OR.(MX.AND.NX)) GO TO 2 
*** ORIG        IF(.NOT.(MY.AND.NY.AND.(MX.OR.NX).AND..NOT.(MX.AND.NX))) GO TO 3 
*** ORIG        INOUT=-INOUT 
*** ORIG        GO TO 2
*** ORIG      3 IF((Y(I)*X(J)-X(I)*Y(J))/(X(J)-X(I))) 2,4,5 
*** ORIG      4 INOUT=0   
*** ORIG        RETURN 
*** ORIG      5 INOUT=-INOUT 
*** ORIG      2 CONTINUE 
*** ORIG        RETURN 
*** ORIG        END 
