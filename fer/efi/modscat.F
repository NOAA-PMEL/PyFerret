
      SUBROUTINE modscat (ax, nax, nscat, scatm)

* ACM 11/00
*  Called by scat2gridlaplace* gridding functions.
*  If there is a modulo axis, apply modulo-ness to the coordinates in that 
*  direction of the scattered points.

*  ax		modulo axis
*  nax		number of points in the modulo axis
*  nscat	input: number of scattered points, output: updated number.
*  scatm	scattered points coordinate array in the direction of the 
*		modulo axis

      REAL scatm(*)
      REAL*8 ax(*)
      INTEGER nscat, i, nax
      REAL ax1, axf, scatpt, diff, range

      ax1 = ax(1)
      axf = ax(nax)
      range = axf - ax1

      DO 100 i = 1, nscat
         scatpt = scatm(i)
         diff = scatpt - axf
         DO WHILE (diff .GE. 0)
            scatpt = scatpt - range
            diff = scatpt - axf
         ENDDO

         diff = scatpt - ax1 
         DO WHILE (diff .LT. 0.)
            scatpt = scatpt + range
            diff = scatpt - ax1
         ENDDO

         scatm(i) = scatpt
  100 CONTINUE

      RETURN
      END

