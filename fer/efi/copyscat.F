      SUBROUTINE copyscat (ax, nax, nrng, nscat, scatm, scat1, scat2,
     .                     nscatout)

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
c            scatm(nscatout) = ax1 - diff
            scatm(nscatout) = ax1 + diff
            scat1(nscatout) = scat1(i)
            scat2(nscatout) = scat2(i)
         ENDIF

         diff = scatpt - ax1
         IF (diff .GE. 0. .AND. diff .LT. cut) THEN
            nscatout = nscatout + 1
c            scatm(nscatout) = axf + diff
            scatm(nscatout) = axf - diff
            scat1(nscatout) = scat1(i)
            scat2(nscatout) = scat2(i)
         ENDIF

  100 CONTINUE

      RETURN
      END

