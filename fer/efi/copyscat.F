
      SUBROUTINE copyscat (ax, nax, nrng, nscat, scatm, scat1, scat2)

* ACM 11/00
*  Called by scat2gridlaplace* gridding functions.
*  If there is a modulo axis, mirror the points within nrng of each 
*  end beyond the opposite end.  Changes nscat.

*  ax		modulo axis
*  nax		number of points in the modulo axis
*  nrng		gridding parameter -- range of grid points used in gridding
*  nscat	input: number of scattered points, output: updated number.
*  scatm	scattered points coordinate array in the direction of the 
*		modulo axis
*  scat1	scattered points other coordinate array
*  scat2	scattered points function values

      REAL scatm(*), scat1(*), scat2(*)
      REAL*8 ax(*)
      INTEGER nscat, nscatout, nrng, i, nax
      REAL ax1, axf, del, scatpt, cut, diff

      ax1 = ax(1)
      axf = ax(nax)
      del = (ax(nax) - ax(1))/ float(nax-1)
      cut = nrng* del

      nscatout = nscat
      DO 100 i = 1, nscat
         scatpt = scatm(i)
         diff = axf - scatpt
         IF (diff .GE. 0. .AND. diff .LT. cut) THEN
            nscatout = nscatout + 1
            scatm(nscatout) = ax1 - diff
            scat1(nscatout) = scat1(i)
            scat2(nscatout) = scat2(i)
         ENDIF

         diff = scatpt - ax1
         IF (diff .GE. 0. .AND. diff .LT. cut) THEN
            nscatout = nscatout + 1
            scatm(nscatout) = axf + diff
            scat1(nscatout) = scat1(i)
            scat2(nscatout) = scat2(i)
         ENDIF

  100 CONTINUE

      nscat = nscatout

      RETURN
      END

