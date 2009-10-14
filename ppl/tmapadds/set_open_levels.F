      SUBROUTINE set_open_levels
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
* V630  *acm* 9/09 Introduction of syntax for variance-based and histogram levels

      include 'parampl5_dat.decl'
      include 'PARAMPL5.DAT'
      include 'axis_inc.decl'
      include 'AXIS.INC'
      include 'plt_inc.decl'
      include 'PLT.INC'
      include 'miss_inc.decl'
      include 'MISS.INC'
      include 'hd_inc.decl'
      include 'HD.INC'
      include 'cont_inc.decl'
      include 'CONT.INC'
      include 'errmsg.parm'

      REAL    dz, fm, znp, zmp, dzf, znpf, zmpf, 
     .        znpclo,zmpclo, dzc, dzclo, dzchi, 
     .        twostd, density_factor
      INTEGER m, jj, nlevf, nlevc, nlev3, half_ncoarse

      EXTERNAL RANGE

      IF (USE_HIST) THEN  ! compute levels based on histograms.
        GOTO 3000
      ENDIF

! (approx) number of fine and coarse levels.

      nlevf = 0.8* nlev
      nlevc = MAX(0.1*FLOAT(nlev), 1.)

      IF (centered) THEN

	 twostd = 2.* zstd
	 znp = -1*twostd
	 zmp = twostd
	 CALL RANGE(znp,zmp,nlevf,znpf,zmpf,dzf)  ! fine levels

! Create centered fine levels surrounded by centered coarse levels.

         density_factor = 4.
         dzc = dzf* density_factor
	 half_ncoarse = nlevc/2

	 znpclo = lev_mean + znpf - dzc* half_ncoarse
	 zmpclo = lev_mean + znpf
	 znpchi = lev_mean + zmpf
	 zmpchi = lev_mean + zmpf + dzc* half_ncoarse

         IF (zmax .LT. znpclo  .OR.  zmin .GT. zmpchi) THEN

	    zmp = 0.6* MAX( ABS(zmin), ABS(zmax) )
	    znp = -1* zmp
	    CALL RANGE(znp,zmp,nlevf,znpf,zmpf,dzf)  ! fine levels

! Create centered fine levels surrounded by centered coarse levels.

            density_factor = 4.
            dzc = dzf* density_factor
	    half_ncoarse = nlevc/2

	    znpclo = lev_mean + znpf - dzc* half_ncoarse
	    zmpclo = lev_mean + znpf
	    znpchi = lev_mean + zmpf
	    zmpchi = lev_mean + zmpf + dzc* half_ncoarse

         ENDIF

      ELSE  ! not centered

         znp = zmean - zstd
         zmp = zmean + zstd

         CALL RANGE(znp,zmp,nlevf,znpf,zmpf,dzf)  ! fine levels

         znp = zmean - 2.* zstd
         zmp = zmean - zstd
         IF (znp .LT. zmin) znp = zmin
         IF (zmp .LT. zmin) zmp = znpf
         CALL RANGE(znp,zmp,nlevc,znpclo,zmpclo,dzclo)  ! coarse levels

         znp = zmean + zstd
         zmp = zmean + 2.* zstd
         IF (zmp .GT. zmax) zmp = zmpf
         IF (znp .GT. zmax) znp = zmax
         CALL RANGE(znp,zmp,nlevc,znpchi,zmpchi,dzchi)  ! coarse levels
         dzc = (dzclo + dzchi)/ 2  ! are they always equal??

      ENDIF  ! centered or not centered

c Set jj as in plotz based on fine levels.
      
      IF (zmpf-znpf .EQ. 0) then
         jj = nsigc
      ELSE
         jj = nsigc - IFIX(ALOG10(ABS(zmpf-znpf)) + .5)
      ENDIF 
      IF (jj .LE. 0) jj= - 1

      ! -inf
      IF (zmin .LT. znpclo) THEN
         neginf = .TRUE.
         ilev = 2
      ELSE 
         ilev = 1
      ENDIF

      nlev2 =1 

      nlev3=(zmpclo-znpclo)/ dzc + 1
      nlev2 = nlev2 + nlev3

      do 101 i = 1, nlev3 

         lwgt(ilev) = 2
         ldig(ilev) = -1
         zlev(ilev) = znpclo + dzc* FLOAT(i-1)
         IF (zlev(ilev) .GT. znpf) goto 102

         IF (zlev(ilev).GT.0.) then
            lwgt(ilev) = 1
         ELSE
            lwgt(ilev) = 3
         ENDIF
         ldig(ilev) = jj
	 IF (centered) THEN
	    ilev = ilev + 1
         ELSE
	    IF (zlev(ilev) .GT. zmax) GOTO 2000

         IF (zlev(ilev)+ dzc* FLOAT(i) .GE. zmin) 
     .	    ilev = ilev + 1
         ENDIF

 101  CONTINUE 
 102  CONTINUE 

      IF (neginf) THEN
         zlev(1) = MIN( zlev(2), zmin-1. )
         lwgt(1) = lwgt(2)
         ldig(1) = ldig(2)
      ENDIF

      nlev3=(zmpf-znpf)/ dzf + 1
      nlev2 = nlev2 + nlev3

      IF (zlev(ilev-1) .GT. znpf) ilev = ilev - 1
      IF (ilev.GT.1 .AND. zlev(ilev-1).GT.znpf) ilev = ilev - 1

      i1 = 1
      IF (ilev .GT. 1) THEN
         zlev(ilev) = znpf + dzf* FLOAT(i1-1)
         DO WHILE ((zlev(ilev) .LE. zlev(ilev-1)) ) 
	    i1 = 2
	    zlev(ilev) = znpchi + dzc* FLOAT(i1-1)
	 ENDDO
      ENDIF 

      IF (zlev(1) .GT. znpf) THEN
	 ilev = 1
	 i1 = 2
      ENDIF

      DO 103 i=i1,nlev3 

         lwgt(ilev) = 2
         ldig(ilev) = -1
         zlev(ilev) = znpf + dzf* FLOAT(i-1)
         IF (zlev(ilev) .GT. znpchi) goto 104

         IF (zlev(ilev).GT.0.) then
            lwgt(ilev) = 1
         ELSE
            lwgt(ilev) = 3
         ENDIF
         ldig(ilev) = jj

	 IF (centered) THEN
	    ilev = ilev + 1
         ELSE
	    IF (zlev(ilev) .GT. zmax) GOTO 2000

         IF (zlev(ilev)+ dzf* FLOAT(i) .GE. zmin) 
     .	    ilev = ilev + 1
         ENDIF

 103  CONTINUE 
 104  CONTINUE 

      nlev3=(zmpchi-znpchi)/ dzc + 1
      nlev2 = nlev2 + nlev3

      IF (zlev(ilev-1) .GT. znpchi) ilev = ilev - 1
      IF (zlev(ilev-1) .GT. znpchi) ilev = ilev - 1

      i1 = 1
      IF (ilev .GT. 1) THEN
         zlev(ilev) = znpchi + dzc* FLOAT(i1-1)
         DO WHILE ((zlev(ilev) .LE. zlev(ilev-1)) ) 
	    i1 = 2
	    zlev(ilev) = znpchi + dzc* FLOAT(i1-1)
	 ENDDO
      ENDIF 

      DO 105 i=i1,nlev3 

         lwgt(ilev) = 2
         ldig(ilev) = -1
         zlev(ilev) = znpchi + dzc* FLOAT(i-1)
         IF (zlev(ilev).GT.0.) then
            lwgt(ilev) = 1
         ELSE
            lwgt(ilev) = 3
         ENDIF
         ldig(ilev) = jj

	 IF (centered) THEN
	    ilev = ilev + 1
	 ELSE
	    IF (zlev(ilev) .GT. zmax) GOTO 2000

            IF (zlev(ilev)+ dzc* FLOAT(i) .GE. zmin) 
     .	       ilev = ilev + 1
         ENDIF

 105  CONTINUE 

 2000 CONTINUE
      nlev2 = ilev
      IF (zmax .GT. zmpchi) THEN
         posinf = .TRUE.
         zlev(nlev2) = MAX (zlev(nlev2-1), zmax + 1. )
      ELSE   
         zlev(nlev2) = zmpchi
      ENDIF

      lwgt(nlev2) = lwgt(nlev2-1)
      ldig(nlev2) = ldig(nlev2-1)

* If a min or max was explicitly set, then chop off the 
* levels with that setting.

 3000 CONTINUE
      IF (set_min) THEN
         neginf = .FALSE.
         IF (lev_min .LT. zlev(1)) THEN
            zlev(1) = lev_min
         ELSE
            index = 1
            DO i = nlev2, 1, -1
               IF (lev_min .LE. zlev(i)) index = i
            ENDDO
            IF (index .GT. 1) THEN
               DO i = index, nlev2
                  zlev(i-index+1) = zlev(i)
                  lwgt(i-index+1) = lwgt(i)
                  ldig(i-index+1) = ldig(i)
               ENDDO
               nlev2 = nlev2-index+1
            ENDIF
         ENDIF
      ENDIF

      IF (set_max) THEN
         posinf = .FALSE.
         IF (lev_max .LT. zlev(nlev2)) THEN
            zlev(nlev2) = lev_max
         ELSE
            index = nlev2
            DO i = 1, nlev2
                  IF (lev_max .GE. zlev(i)) index = i
            ENDDO
            IF (index .LT. nlev2) nlev2 = index
         ENDIF
      ENDIF

 5000 CONTINUE
      RETURN
      END
