	SUBROUTINE TM_I4_TO_R4( input, vals, n)

* convert an array of INTEGER*4 values into REAL
* use backwards loop so input and output can be same array

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 2/92
* replaced "elif" syntax with
*	else
*	   if
*  for SGI port	 - kob 4/8/92
*
* argument declarations
      INTEGER n
      INTEGER*4 input(n)
      REAL  vals(n)

* local variable declarations
      INTEGER i

      DO 100 i = n, 1, -1
 100  vals(i) = input(i)

      RETURN
      END

***************************************************************

	SUBROUTINE TM_I2_TO_R4( input, vals, n)

* convert an array of INTEGER*2 values into REAL
* use backwards loop so input and output can be same array

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 2/92

* argument declarations
      INTEGER n
      INTEGER*2 input(n)
      REAL  vals(n)

* local variable declarations
      INTEGER i

      DO 100 i = n, 1, -1
 100  vals(i) = input(i)

      RETURN
      END

***************************************************************
	SUBROUTINE TM_I1_TO_R4( input, vals, n)

* convert an array of INTEGER*4 values into REAL
* use backwards loop so input and output can be same array

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 2/92

* argument declarations
      INTEGER n
      REAL  vals(n)
#ifdef sun
      BYTE input(n)
#else
#	if unix
      		INTEGER*1 input(n)
#	else
      		BYTE input(n)
#	endif
#endif

* local variable declarations
      INTEGER i

      DO 100 i = n, 1, -1
 100  vals(i) = input(i)

      RETURN
      END
