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
*  updates, revisions, new versions or 'bug fixes'.
*
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP 'AS IS' AND ANY EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE. 
*
* V630  *acm* 9/09 Introduction of syntax for variance-based and histogram levels
* V65 *acm   2/10 If Open-ended levels were used, symbol LEV_OPNLEVS saves 
* the /LEV command needed to duplicate the levels
* v663 *acm* 7/10 Fix bug where min value was not inside the lowest level.
* v671 *acm* 6/11 For centered levels if there is (-inf) force (inf) and vice versa.
* v673 *acm* 10/11 Fix cut-and-paste bug on first setting of nlev3.
* V68  *acm* 1/12  changes for double-precision ferret, single-precision pplus

      IMPLICIT NONE
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

      LOGICAL         TM_FPEQ_SNGL
      REAL    dz, fm, znp, zmp, dzf, znpf, zmpf, 
     .        znpclo,zmpclo, dzc, dzclo, dzchi, 
     .        twostd, density_factor, zlast, zmpchi, znpchi
      INTEGER TM_LENSTR1, m, jj, nlevf, nlevc, nlev3, half_ncoarse, 
     .        slen, llen, dlen, hlen, if1, if2, i, i1, i2, ilev, 
     .        index, k, levcount

      CHARACTER TM_FMT_SNGL*48, lo_str*10, del_str*10, hi_str*10

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

      lev_str = ' '
      slen = 0

c Set jj as in plotz based on fine levels.
      
      IF (zmpf-znpf .EQ. 0) then
         jj = nsigc
      ELSE
         jj = nsigc - IFIX(ALOG10(ABS(zmpf-znpf)) + .5)
      ENDIF 
      IF (jj .LE. 0) jj= -1

      ! -inf
      IF (zmin .LT. znpclo) THEN
         neginf = .TRUE.
         ilev = 2
	 lev_str = '(-inf)'
	 slen = 6
      ELSE 
         ilev = 1
      ENDIF

      nlev2 =1 

      nlev3 = NINT((zmpclo-znpclo)/ dzc) + 1
      nlev2 = nlev2 + nlev3

      do 101 i = 1, nlev3 

         ldig(ilev) = -1
         zlev(ilev) = znpclo + dzc* FLOAT(i-1)
         IF (zlev(ilev) .GT. znpf) goto 102

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
         ldig(1) = ldig(2)
      ENDIF

      nlev3 = NINT((zmpf-znpf)/ dzf) + 1
      nlev2 = nlev2 + nlev3

      IF (zlev(ilev-1) .GT. znpf) ilev = ilev - 1
      IF (ilev.GT.1 .AND. zlev(ilev-1).GT.znpf) ilev = ilev - 1

      i1 = 1
      IF (ilev .GT. 1) THEN
         zlev(ilev) = znpf + dzf* FLOAT(i1-1)
         DO WHILE ((zlev(ilev) .LT. zlev(ilev-1)) ) 
	    i1 = 2
	    zlev(ilev) = znpchi + dzc* FLOAT(i1-1)
	 ENDDO
      ENDIF 

      IF (zlev(1) .GT. znpf) THEN
	 ilev = 1
	 i1 = 2
      ENDIF

      lo_str = TM_FMT_SNGL(znpclo, 3, 10, llen)
      del_str = TM_FMT_SNGL(dzc, 3, 10, dlen)
      IF (ilev .GT. 1) THEN
         hi_str = TM_FMT_SNGL(zlev(ilev-1), 3, 10, hlen)
      ELSE
         hi_str = TM_FMT_SNGL(zlev(ilev), 3, 10, hlen)
	 IF (zlev(ilev) .EQ. 0) hi_str = 
     .         TM_FMT_SNGL(zlev(ilev-1), 3, 10, hlen)
      ENDIF
      lev_str = lev_str(:slen)//'('//lo_str(:llen)//
     .          ','//hi_str(:hlen)//','//del_str(:dlen)//')'
      slen = TM_LENSTR1(lev_str)
      if1 = ilev

      DO 103 i=i1,nlev3 

         ldig(ilev) = -1
         zlev(ilev) = znpf + dzf* FLOAT(i-1)
         IF (zlev(ilev) .GT. znpchi) goto 104

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

* If min is still less than the lowest lev, add an open level
* at the start of the list.

      IF (zlev(1) .GT. zmin) THEN
         neginf = .TRUE.
	 DO i = ilev, 2, -1
	    zlev(i) = zlev(i-1)
	    ldig(i) = ldig(i-1)
	 ENDDO
	 
         zlev(1) = zmin-1.
	 ilev = ilev + 1
      ENDIF
      nlev3 = NINT((zmpchi-znpchi)/ dzc) + 1
      nlev2 = nlev2 + nlev3

      IF (zlev(ilev-1) .GT. znpchi) ilev = ilev - 1
      IF (zlev(ilev-1) .GT. znpchi) ilev = ilev - 1

      if2 = ilev
      IF (centered) if2 = ilev-1
      lo_str = TM_FMT_SNGL(zlev(if1), 3, 10, llen)
      del_str = TM_FMT_SNGL(dzf, 3, 10, dlen)
      hi_str = TM_FMT_SNGL(zlev(if2), 3, 10, hlen)
      IF (zlev(if2) .EQ. 0.) hi_str = TM_FMT_SNGL(zlev(if2-1), 3, 10, hlen)
      lev_str = lev_str(:slen)//'('//lo_str(:llen)//
     .          ','//hi_str(:hlen)//','//del_str(:dlen)//')'
      slen = TM_LENSTR1(lev_str)
      if1 = ilev

      i1 = 1
      IF (ilev .GT. 1) THEN
         zlev(ilev) = znpchi + dzc* FLOAT(i1-1)
         DO WHILE ((zlev(ilev) .LT. zlev(ilev-1)) ) 
	    i1 = 2
	    zlev(ilev) = znpchi + dzc* FLOAT(i1-1)
	 ENDDO
      ENDIF 

      DO 105 i=i1,nlev3 

         ldig(ilev) = -1
         zlev(ilev) = znpchi + dzc* FLOAT(i-1)
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

      lo_str = TM_FMT_SNGL(zlev(if1), 3, 10, llen)
      del_str = TM_FMT_SNGL(dzc, 3, 10, dlen)
      hi_str = TM_FMT_SNGL(zlev(ilev-1), 3, 10, hlen)
      IF (zlev(ilev-1) .EQ. 0) 
     .          hi_str = TM_FMT_SNGL(zlev(ilev-2), 3, 10, hlen)
      lev_str = lev_str(:slen)//'('//lo_str(:llen)//
     .          ','//hi_str(:hlen)//','//del_str(:dlen)//')'
      slen = TM_LENSTR1(lev_str)
      if1 = ilev

* If needed then add (inf)
* If centered and there is (-inf) then add (inf)

      nlev2 = ilev
      IF ((zmax .GT. zmpchi) .OR. (centered .AND. neginf)) THEN
         posinf = .TRUE.
         zlev(nlev2) = MAX (zlev(nlev2-1)+1, zmax + 1. )
	 lev_str = lev_str(:slen)//'(inf)'
         slen = TM_LENSTR1(lev_str)
      ELSE   
         zlev(nlev2) = zmpchi
      ENDIF

      IF ((centered .AND. posinf) .AND.(.NOT.neginf)) THEN
         neginf = .TRUE.
	 DO 106 i = nlev2, 1, -1
           ldig(i+1) = ldig(i)
           zlev(i+1) = zlev(i)
 106     CONTINUE 
	 nlev2 = nlev2 + 1
	 zlev(1) =  MIN (zlev(1)-1, zmin - 1. )
	 lev_str = '(-inf)'//lev_str(:slen)
         slen = TM_LENSTR1(lev_str)
      ENDIF

      ldig(nlev2) = ldig(nlev2-1)

      IF (centered .AND. (.NOT.posinf)) THEN
         IF (zlev(nlev2) .EQ. zlev(nlev2-1)) THEN
	    ilev = ilev - 1
	    nlev2 = nlev2 - 1
	 ENDIF
      ENDIF

* if centered and there is (inf) then add (-inf) if its not there.

      IF ( (centered .AND. posinf) .AND. (.NOT.neginf) ) THEN
         DO 107 i=ilev,1,-1
            ldig(i+1) = ldig(i)
            zlev(i+1) = zlev(i)
 107     CONTINUE 
         ilev = ilev + 1
	 nlev2 = ilev
	 neginf = .TRUE.
	 lev_str = '(-inf)'//lev_str(:slen)
         slen = TM_LENSTR1(lev_str)
      ENDIF

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
               DO 108 i = index, nlev2
                  zlev(i-index+1) = zlev(i)
                  ldig(i-index+1) = ldig(i)
 108           CONTINUE 
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

* Remove duplicate levels
      levcount = nlev2
      zlast = zlev(1)
      do 530 i = 2, nlev2 
         IF ( TM_FPEQ_SNGL(zlev(i), zlast) .AND. i.LT.levcount ) THEN
            do 520 k = i,levcount
               zlev(k-1) = zlev(k)
               ldig(k-1) = ldig(k)
 520        CONTINUE
            levcount = levcount-1
         ENDIF
         zlast = zlev(i)
 530  CONTINUE
      nlev2 = levcount

* Finally, because of rounding when computing nlev3, when making centered 
* levels there can still be different numbers of levels above and below 
* zero. Force centered levels.

      IF (centered) THEN
         i1 = 1
	 i2 = nlev2
	 IF (neginf) i1 = 2
	 IF (posinf) i2 = nlev2-1
	 IF (.NOT. TM_FPEQ_SNGL(ABS(zlev(i1)), zlev(i2) ) ) THEN
	    IF (ABS(zlev(i1)) .GT. zlev(i2)) THEN
	       DO 109 i = nlev2+1, i2+1, -1
	         zlev(i) = zlev(i-1)
                 ldig(i) = ldig(i-1)
 109	       CONTINUE
	       zlev(nlev2) = ABS(zlev(i1))
               ldig(nlev2) = ldig(i1)
	    ELSE
	       DO 110 i = nlev2+1, i1, -1
	         zlev(i) = zlev(i-1)
                 ldig(i) = ldig(i-1)
 110	       CONTINUE
	       zlev(nlev2) = -1.*zlev(i2)
	       ldig(nlev2) = ldig(i2)

	    ENDIF
	    nlev2 = nlev2 + 1
	 ENDIF
      ENDIF

* Set lwgt - as in lev.F
      DO 111 i = 1, nlev2
         IF (zlev(i).GT.0.) THEN
            lwgt(i) = 1
         ELSE
            lwgt(i) = 3
         ENDIF
	 IF (TM_FPEQ_SNGL(zlev(i), 0.)) lwgt(i) = 2 
 111  CONTINUE

      RETURN
      END
