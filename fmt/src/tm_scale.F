	SUBROUTINE TM_SCALE ( factor, offset, array, n )

* rescale an array: multiply by a scale factor and add an offset

* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* revision 0.0 3/92 for TMAP library

* calling argument declarations:
      INTEGER n
      REAL factor, offset, array(n)

* internal variable declarations:
      INTEGER i

      DO 100 i = 1, n
 100  array(i) = array(i)*factor + offset

      RETURN
      END
