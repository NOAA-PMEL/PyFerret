*  fftsubs.F
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
*   computation routines for FFT functions

      SUBROUTINE four_re (nd, x, a, b, wft, iwft)
      IMPLICIT NONE
      INTEGER nd, iwft(*)
      REAL x(*), wft(*) 
      REAL a(*), b(*)
      
      INTEGER nf, i, j
      REAL xn

c   uses NCAR FFTPACK code

C  Ansley Manke 1/2000 NOAA/PMEL   Return A, B real arrays with Fourier coefficients.
C  Uses notes by Ned Cokelet 1/2000 on Swartztrauber FFTPACK code.
C  11/2022 *acm* Code cleanup: Update FFTPACK code to F90 standard, from
C                https://github.com/fortran-lang/fftpack.git
C                with one change: instead of using the module  "use fftpack_kind",
C                just define rk as done in rk.f90

C  NF = number of frequencies, half the number of times.
C  The code returns frequencies W(i) for i=0 to ND/2, with ND/2 rounded down.
C  We do not return a(0) = R1/ND
C  We return a(i) and b(i) for i=1,... ND/2

      nf = nd/ 2

      CALL rfftf (nd, x, wft, iwft) 

C  Normalizing factor of 1./N

c      xn = 1.0
      xn = 1.0/ REAL(nd)

c   Save FFT coefficients in arrays a and b.
 
      j = 0
      DO i = 1, nf-1
        j = 2* i
        a(i) =  2.* xn* x(j)
        b(i) = -2.* xn* x(j+1) 
      ENDDO

C  Set a(nf) and b(nf) when nd is even/odd.

      IF (nf*2 .eq. nd) THEN		! even ND
         a(nf) = xn* x(nd)
         b(nf) = 0.
      ELSE				! odd ND
         a(nf) = 2.* xn* x(nd-1)
         b(nf) = -2.* xn* x(nd)
      ENDIF

      RETURN 
      END
 
      REAL FUNCTION PIMACH (DUM)
C   Returns pi
C 
      IMPLICIT NONE
      REAL DUM
      PIMACH = 4.*ATAN(1.0)
      RETURN
      END
 
C
C     RFFTF COMPUTES THE FOURIER COEFFICIENTS OF A REAL
C     PERODIC SEQUENCE (FOURIER ANALYSIS). THE TRANSFORM IS DEFINED 
C     BELOW AT OUTPUT PARAMETER R. 
C
C     INPUT PARAMETERS 
C
C     N       THE LENGTH OF THE ARRAY R TO BE TRANSFORMED.  THE METHOD
C             IS MOST EFFICIENT WHEN N IS A PRODUCT OF SMALL PRIMES.
C             N MAY CHANGE SO LONG AS DIFFERENT WORK ARRAYS ARE PROVIDED 
C
C     R       A REAL ARRAY OF LENGTH N WHICH CONTAINS THE SEQUENCE
C             TO BE TRANSFORMED
C
C     WSAVE   A WORK ARRAY WHICH MUST BE DIMENSIONED AT LEAST 2*N. 
C             IN THE PROGRAM THAT CALLS RFFTF. THE WSAVE ARRAY MUST BE
C             INITIALIZED BY CALLING SUBROUTINE RFFTI(N,WSAVE) AND A
C             DIFFERENT WSAVE ARRAY MUST BE USED FOR EACH DIFFERENT 
C             VALUE OF N. THIS INITIALIZATION DOES NOT HAVE TO BE
C             REPEATED SO LONG AS N REMAINS UNCHANGED THUS SUBSEQUENT 
C             TRANSFORMS CAN BE OBTAINED FASTER THAN THE FIRST.
C             THE SAME WSAVE ARRAY CAN BE USED BY RFFTF AND RFFTB.
C     IWFT   AN INTEGER WORK ARRAY OF LENGTH 15
C
C
C     OUTPUT PARAMETERS
C
C     R       R(1) = THE SUM FROM I=1 TO I=N OF R(I) 
C
C             IF N IS EVEN SET L =N/2   , IF N IS ODD SET L = (N+1)/2 
C
C               THEN FOR K = 2,...,L
C
C                  R(2*K-2) = THE SUM FROM I = 1 TO I = N OF
C
C                       R(I)*COS((K-1)*(I-1)*2*PI/N) 
C
C                  R(2*K-1) = THE SUM FROM I = 1 TO I = N OF
C
C                      -R(I)*SIN((K-1)*(I-1)*2*PI/N) 
C
C             IF N IS EVEN
C
C                  R(N) = THE SUM FROM I = 1 TO I = N OF
C
C                       (-1)**(I-1)*R(I)
C
C      *****  NOTE
C                  THIS TRANSFORM IS UNNORMALIZED SINCE A CALL OF RFFTF
C                  FOLLOWED BY A CALL OF RFFTB WILL MULTIPLY THE INPUT
C                  SEQUENCE BY N.
C
C     WSAVE   CONTAINS RESULTS WHICH MUST NOT BE DESTROYED BETWEEN
C             CALLS OF RFFTF OR RFFTB.
C
      SUBROUTINE RFFTF (N,R,WSAVE,IWFT)
      IMPLICIT NONE
      INTEGER N,IWFT(*)
      REAL R(*),WSAVE(*)

      IF (N .EQ. 1) RETURN
      CALL RFFTF1 (N,R,WSAVE,WSAVE(N+1),IWFT)

      RETURN 
      END
 
C
C     RFFTI INITIALIZES THE ARRAY WSAVE WHICH IS USED IN 
C     BOTH RFFTF AND RFFTB. THE PRIME FACTORIZATION OF N TOGETHER WITH
C     A TABULATION OF THE TRIGONOMETRIC FUNCTIONS ARE COMPUTED AND
C     STORED IN WSAVE. 
C
C     INPUT PARAMETER
C
C     N       THE LENGTH OF THE SEQUENCE TO BE TRANSFORMED. 
C
C     OUTPUT PARAMETER 
C
C     WSAVE   A WORK ARRAY WHICH MUST BE DIMENSIONED AT LEAST 2*N+15. 
C             THE SAME WORK ARRAY CAN BE USED FOR BOTH RFFTF AND RFFTB
C             AS LONG AS N REMAINS UNCHANGED. DIFFERENT WSAVE ARRAYS
C             ARE REQUIRED FOR DIFFERENT VALUES OF N. THE CONTENTS OF 
C             WSAVE MUST NOT BE CHANGED BETWEEN CALLS OF RFFTF OR RFFTB. 
C
      SUBROUTINE RFFTI (N,WSAVE,IWFT) 
      IMPLICIT NONE
      INTEGER N, IWFT(*)
      REAL WSAVE(*)
C
      IF (N .EQ. 1) RETURN
      CALL RFFTI1 (N,WSAVE(N+1),IWFT)
      RETURN 
      END 
 
      subroutine rffti1(n,Wa,Ifac)

c rffti1.f90 FFTPACK code updated to F90 standard, from
C https://github.com/fortran-lang/fftpack.git


      implicit none
      integer,parameter :: rk = kind(1.0d0)
      real(rk) :: arg , argh , argld , fi , Wa
      integer :: i , ib , ido , Ifac , ii , ip , ipm , is , j , k1 , l1 , 
     .             l2 , ld , n , nf , nfm1 , nl , nq , nr , ntry
      dimension Wa(*) , Ifac(*)
      integer,dimension(4),parameter :: ntryh = [4 , 2 , 3 , 5]
      real(rk),parameter :: tpi = 2.0_rk * acos(-1.0_rk) ! 2 * pi

      nl = n
      nf = 0
      j = 0
 100  j = j + 1
      if ( j<=4 ) then
         ntry = ntryh(j)
      else
         ntry = ntry + 2
      endif
 200  nq = nl/ntry
      nr = nl - ntry*nq
      if ( nr/=0 ) goto 100
      nf = nf + 1

      Ifac(nf+2) = ntry
      nl = nq
      if ( ntry==2 ) then
         if ( nf/=1 ) then
            do i = 2 , nf
               ib = nf - i + 2
               Ifac(ib+2) = Ifac(ib+1)
            enddo
            Ifac(3) = 2
         endif
      endif
      if ( nl/=1 ) goto 200
      Ifac(1) = n
      Ifac(2) = nf
      argh = tpi/real(n, rk)
      is = 0
      nfm1 = nf - 1
      l1 = 1
      if ( nfm1==0 ) return
      do k1 = 1 , nfm1
         ip = Ifac(k1+2)
         ld = 0
         l2 = l1*ip
         ido = n/l2
         ipm = ip - 1
         do j = 1 , ipm
            ld = ld + l1
            i = is
            argld = real(ld, rk)*argh
            fi = 0.0_rk
            do ii = 3 , ido , 2
               i = i + 2
               fi = fi + 1.0_rk
               arg = fi*argld
               Wa(i-1) = cos(arg)
               Wa(i) = sin(arg)
            enddo
            is = is + ido
         enddo
         l1 = l2
      enddo

      end subroutine rffti1


      subroutine rfftf1(n,c,Ch,Wa,Ifac)
      implicit none
      integer,parameter :: rk = kind(1.0d0)
      real(rk) :: c , Ch , Wa
      integer :: i , idl1 , ido , Ifac , ip , iw , ix2 , ix3 , ix4 , k1 , 
     .            kh , l1 , l2 , n , na , nf
      dimension Ch(*) , c(*) , Wa(*) , Ifac(*)

      nf = Ifac(2)
      na = 1
      l2 = n
      iw = n
      do k1 = 1 , nf
         kh = nf - k1
         ip = Ifac(kh+3)
         l1 = l2/ip
         ido = n/l2
         idl1 = ido*l1
         iw = iw - (ip-1)*ido
         na = 1 - na
         if ( ip==4 ) then
            ix2 = iw + ido
            ix3 = ix2 + ido
            if ( na/=0 ) then
               call radf4(ido,l1,Ch,c,Wa(iw),Wa(ix2),Wa(ix3))
            else
               call radf4(ido,l1,c,Ch,Wa(iw),Wa(ix2),Wa(ix3))
            endif
         elseif ( ip/=2 ) then
            if ( ip==3 ) then
               ix2 = iw + ido
               if ( na/=0 ) then
                  call radf3(ido,l1,Ch,c,Wa(iw),Wa(ix2))
               else
                  call radf3(ido,l1,c,Ch,Wa(iw),Wa(ix2))
               endif
            elseif ( ip/=5 ) then
               if ( ido==1 ) na = 1 - na
               if ( na/=0 ) then
                  call radfg(ido,ip,l1,idl1,Ch,Ch,Ch,c,c,Wa(iw))
                  na = 0
               else
                  call radfg(ido,ip,l1,idl1,c,c,c,Ch,Ch,Wa(iw))
                  na = 1
               endif
            else
               ix2 = iw + ido
               ix3 = ix2 + ido
               ix4 = ix3 + ido
               if ( na/=0 ) then
                  call radf5(ido,l1,Ch,c,Wa(iw),Wa(ix2),Wa(ix3),Wa(ix4))
               else
                  call radf5(ido,l1,c,Ch,Wa(iw),Wa(ix2),Wa(ix3),Wa(ix4))
               endif
            endif
         elseif ( na/=0 ) then
            call radf2(ido,l1,Ch,c,Wa(iw))
         else
            call radf2(ido,l1,c,Ch,Wa(iw))
         endif
         l2 = l1
      enddo
      if ( na==1 ) return
      do i = 1 , n
         c(i) = Ch(i)
      enddo
      end subroutine rfftf1

      subroutine radfg(Ido,Ip,l1,Idl1,Cc,c1,c2,Ch,Ch2,Wa)

      implicit none
      integer,parameter :: rk = kind(1.0d0)
      real(rk) :: ai1 , ai2 , ar1 , ar1h , ar2 , ar2h , arg , c1 , 
     .             c2 , Cc , Ch , Ch2 , dc2 , dcp , ds2 , dsp , 
     .             Wa
      integer :: i , ic , idij , Idl1 , Ido , idp2 , ik , Ip , ipp2 , 
     .            ipph , is , j , j2 , jc , k , l , l1 , lc , nbd
      dimension Ch(Ido,l1,Ip) , Cc(Ido,Ip,l1) , c1(Ido,l1,Ip) , 
     .           c2(Idl1,Ip) , Ch2(Idl1,Ip) , Wa(*)
      real(rk),parameter :: tpi = 2.0_rk * acos(-1.0_rk) ! 2 * pi

      arg = tpi/real(Ip, rk)
      dcp = cos(arg)
      dsp = sin(arg)
      ipph = (Ip+1)/2
      ipp2 = Ip + 2
      idp2 = Ido + 2
      nbd = (Ido-1)/2
      if ( Ido==1 ) then
         do ik = 1 , Idl1
            c2(ik,1) = Ch2(ik,1)
         enddo
      else
         do ik = 1 , Idl1
            Ch2(ik,1) = c2(ik,1)
         enddo
         do j = 2 , Ip
            do k = 1 , l1
               Ch(1,k,j) = c1(1,k,j)
            enddo
         enddo
         if ( nbd>l1 ) then
            is = -Ido
            do j = 2 , Ip
               is = is + Ido
               do k = 1 , l1
                  idij = is
                  do i = 3 , Ido , 2
                     idij = idij + 2
                     Ch(i-1,k,j) = Wa(idij-1)*c1(i-1,k,j) + Wa(idij) 
     .                              *c1(i,k,j)
                     Ch(i,k,j) = Wa(idij-1)*c1(i,k,j) - Wa(idij) 
     .                            *c1(i-1,k,j)
                  enddo
               enddo
            enddo
         else
            is = -Ido
            do j = 2 , Ip
               is = is + Ido
               idij = is
               do i = 3 , Ido , 2
                  idij = idij + 2
                  do k = 1 , l1
                     Ch(i-1,k,j) = Wa(idij-1)*c1(i-1,k,j) + Wa(idij) 
     .                              *c1(i,k,j)
                     Ch(i,k,j) = Wa(idij-1)*c1(i,k,j) - Wa(idij) 
     .                            *c1(i-1,k,j)
                  enddo
               enddo
            enddo
         endif
         if ( nbd<l1 ) then
            do j = 2 , ipph
               jc = ipp2 - j
               do i = 3 , Ido , 2
                  do k = 1 , l1
                     c1(i-1,k,j) = Ch(i-1,k,j) + Ch(i-1,k,jc)
                     c1(i-1,k,jc) = Ch(i,k,j) - Ch(i,k,jc)
                     c1(i,k,j) = Ch(i,k,j) + Ch(i,k,jc)
                     c1(i,k,jc) = Ch(i-1,k,jc) - Ch(i-1,k,j)
                  enddo
               enddo
            enddo
         else
            do j = 2 , ipph
               jc = ipp2 - j
               do k = 1 , l1
                  do i = 3 , Ido , 2
                     c1(i-1,k,j) = Ch(i-1,k,j) + Ch(i-1,k,jc)
                     c1(i-1,k,jc) = Ch(i,k,j) - Ch(i,k,jc)
                     c1(i,k,j) = Ch(i,k,j) + Ch(i,k,jc)
                     c1(i,k,jc) = Ch(i-1,k,jc) - Ch(i-1,k,j)
                  enddo
               enddo
            enddo
         endif
      endif
      do j = 2 , ipph
         jc = ipp2 - j
         do k = 1 , l1
            c1(1,k,j) = Ch(1,k,j) + Ch(1,k,jc)
            c1(1,k,jc) = Ch(1,k,jc) - Ch(1,k,j)
         enddo
      enddo
!
      ar1 = 1.0_rk
      ai1 = 0.0_rk
      do l = 2 , ipph
         lc = ipp2 - l
         ar1h = dcp*ar1 - dsp*ai1
         ai1 = dcp*ai1 + dsp*ar1
         ar1 = ar1h
         do ik = 1 , Idl1
            Ch2(ik,l) = c2(ik,1) + ar1*c2(ik,2)
            Ch2(ik,lc) = ai1*c2(ik,Ip)
         enddo
         dc2 = ar1
         ds2 = ai1
         ar2 = ar1
         ai2 = ai1
         do j = 3 , ipph
            jc = ipp2 - j
            ar2h = dc2*ar2 - ds2*ai2
            ai2 = dc2*ai2 + ds2*ar2
            ar2 = ar2h
            do ik = 1 , Idl1
               Ch2(ik,l) = Ch2(ik,l) + ar2*c2(ik,j)
               Ch2(ik,lc) = Ch2(ik,lc) + ai2*c2(ik,jc)
            enddo
         enddo
      enddo
      do j = 2 , ipph
         do ik = 1 , Idl1
            Ch2(ik,1) = Ch2(ik,1) + c2(ik,j)
         enddo
      enddo
!
      if ( Ido<l1 ) then
         do i = 1 , Ido
            do k = 1 , l1
               Cc(i,1,k) = Ch(i,k,1)
            enddo
         enddo
      else
         do k = 1 , l1
            do i = 1 , Ido
               Cc(i,1,k) = Ch(i,k,1)
            enddo
         enddo
      endif
      do j = 2 , ipph
         jc = ipp2 - j
         j2 = j + j
         do k = 1 , l1
            Cc(Ido,j2-2,k) = Ch(1,k,j)
            Cc(1,j2-1,k) = Ch(1,k,jc)
         enddo
      enddo
      if ( Ido==1 ) return
      if ( nbd<l1 ) then
         do j = 2 , ipph
            jc = ipp2 - j
            j2 = j + j
            do i = 3 , Ido , 2
               ic = idp2 - i
               do k = 1 , l1
                  Cc(i-1,j2-1,k) = Ch(i-1,k,j) + Ch(i-1,k,jc)
                  Cc(ic-1,j2-2,k) = Ch(i-1,k,j) - Ch(i-1,k,jc)
                  Cc(i,j2-1,k) = Ch(i,k,j) + Ch(i,k,jc)
                  Cc(ic,j2-2,k) = Ch(i,k,jc) - Ch(i,k,j)
               enddo
            enddo
         enddo
      else
         do j = 2 , ipph
            jc = ipp2 - j
            j2 = j + j
            do k = 1 , l1
               do i = 3 , Ido , 2
                  ic = idp2 - i
                  Cc(i-1,j2-1,k) = Ch(i-1,k,j) + Ch(i-1,k,jc)
                  Cc(ic-1,j2-2,k) = Ch(i-1,k,j) - Ch(i-1,k,jc)
                  Cc(i,j2-1,k) = Ch(i,k,j) + Ch(i,k,jc)
                  Cc(ic,j2-2,k) = Ch(i,k,jc) - Ch(i,k,j)
               enddo
            enddo
         enddo
      end if

      end subroutine radfg


            subroutine radf2(Ido,l1,Cc,Ch,Wa1)

      implicit none
      integer,parameter :: rk = kind(1.0d0)
      real(rk) :: Cc , Ch , ti2 , tr2 , Wa1
      integer :: i , ic , Ido , idp2 , k , l1
      dimension Ch(Ido,2,l1) , Cc(Ido,l1,2) , Wa1(*)

      do k = 1 , l1
         Ch(1,1,k) = Cc(1,k,1) + Cc(1,k,2)
         Ch(Ido,2,k) = Cc(1,k,1) - Cc(1,k,2)
      enddo
      if ( Ido<2 ) return
      if ( Ido/=2 ) then
         idp2 = Ido + 2
         do k = 1 , l1
            do i = 3 , Ido , 2
               ic = idp2 - i
               tr2 = Wa1(i-2)*Cc(i-1,k,2) + Wa1(i-1)*Cc(i,k,2)
               ti2 = Wa1(i-2)*Cc(i,k,2) - Wa1(i-1)*Cc(i-1,k,2)
               Ch(i,1,k) = Cc(i,k,1) + ti2
               Ch(ic,2,k) = ti2 - Cc(i,k,1)
               Ch(i-1,1,k) = Cc(i-1,k,1) + tr2
               Ch(ic-1,2,k) = Cc(i-1,k,1) - tr2
            enddo
         enddo
         if ( mod(Ido,2)==1 ) return
      endif
      do k = 1 , l1
         Ch(1,2,k) = -Cc(Ido,k,2)
         Ch(Ido,1,k) = Cc(Ido,k,1)
      enddo
      end subroutine radf2

            subroutine radf3(Ido,l1,Cc,Ch,Wa1,Wa2)

      implicit none
      integer,parameter :: rk = kind(1.0d0)
      real(rk) :: Cc , Ch , ci2 , cr2 , di2 , di3 , dr2 , dr3 , 
     .             ti2 , ti3 , tr2 , tr3 , Wa1 , Wa2
      integer :: i , ic , Ido , idp2 , k , l1
      dimension Ch(Ido,3,l1) , Cc(Ido,l1,3) , Wa1(*) , Wa2(*)
      real(rk),parameter :: taur = -0.5_rk
      ! note: original comment said this was -SQRT(3)/2 but value was 0.86602540378443864676d0
      real(rk),parameter :: taui = sqrt(3.0_rk) / 2.0_rk

      do k = 1 , l1
         cr2 = Cc(1,k,2) + Cc(1,k,3)
         Ch(1,1,k) = Cc(1,k,1) + cr2
         Ch(1,3,k) = taui*(Cc(1,k,3)-Cc(1,k,2))
         Ch(Ido,2,k) = Cc(1,k,1) + taur*cr2
      enddo
      if ( Ido==1 ) return
      idp2 = Ido + 2
      do k = 1 , l1
         do i = 3 , Ido , 2
            ic = idp2 - i
            dr2 = Wa1(i-2)*Cc(i-1,k,2) + Wa1(i-1)*Cc(i,k,2)
            di2 = Wa1(i-2)*Cc(i,k,2) - Wa1(i-1)*Cc(i-1,k,2)
            dr3 = Wa2(i-2)*Cc(i-1,k,3) + Wa2(i-1)*Cc(i,k,3)
            di3 = Wa2(i-2)*Cc(i,k,3) - Wa2(i-1)*Cc(i-1,k,3)
            cr2 = dr2 + dr3
            ci2 = di2 + di3
            Ch(i-1,1,k) = Cc(i-1,k,1) + cr2
            Ch(i,1,k) = Cc(i,k,1) + ci2
            tr2 = Cc(i-1,k,1) + taur*cr2
            ti2 = Cc(i,k,1) + taur*ci2
            tr3 = taui*(di2-di3)
            ti3 = taui*(dr3-dr2)
            Ch(i-1,3,k) = tr2 + tr3
            Ch(ic-1,2,k) = tr2 - tr3
            Ch(i,3,k) = ti2 + ti3
            Ch(ic,2,k) = ti3 - ti2
         enddo
      enddo
      end subroutine radf3



            subroutine radf4(Ido,l1,Cc,Ch,Wa1,Wa2,Wa3)
      implicit none
      integer,parameter :: rk = kind(1.0d0)
      real(rk) :: Cc , Ch , ci2 , ci3 , ci4 , cr2 , cr3 , cr4 , 
     .             ti1 , ti2 , ti3 , ti4 , tr1 , tr2 , tr3, 
     .             tr4 , Wa1 , Wa2 , Wa3
      integer :: i , ic , Ido , idp2 , k , l1
      dimension Cc(Ido,l1,4) , Ch(Ido,4,l1) , Wa1(*) , Wa2(*) , Wa3(*)
      real(rk),parameter :: hsqt2 = sqrt(2.0_rk) / 2.0_rk

      do k = 1 , l1
         tr1 = Cc(1,k,2) + Cc(1,k,4)
         tr2 = Cc(1,k,1) + Cc(1,k,3)
         Ch(1,1,k) = tr1 + tr2
         Ch(Ido,4,k) = tr2 - tr1
         Ch(Ido,2,k) = Cc(1,k,1) - Cc(1,k,3)
         Ch(1,3,k) = Cc(1,k,4) - Cc(1,k,2)
      enddo
      if ( Ido<2 ) return
      if ( Ido/=2 ) then
         idp2 = Ido + 2
         do k = 1 , l1
            do i = 3 , Ido , 2
               ic = idp2 - i
               cr2 = Wa1(i-2)*Cc(i-1,k,2) + Wa1(i-1)*Cc(i,k,2)
               ci2 = Wa1(i-2)*Cc(i,k,2) - Wa1(i-1)*Cc(i-1,k,2)
               cr3 = Wa2(i-2)*Cc(i-1,k,3) + Wa2(i-1)*Cc(i,k,3)
               ci3 = Wa2(i-2)*Cc(i,k,3) - Wa2(i-1)*Cc(i-1,k,3)
               cr4 = Wa3(i-2)*Cc(i-1,k,4) + Wa3(i-1)*Cc(i,k,4)
               ci4 = Wa3(i-2)*Cc(i,k,4) - Wa3(i-1)*Cc(i-1,k,4)
               tr1 = cr2 + cr4
               tr4 = cr4 - cr2
               ti1 = ci2 + ci4
               ti4 = ci2 - ci4
               ti2 = Cc(i,k,1) + ci3
               ti3 = Cc(i,k,1) - ci3
               tr2 = Cc(i-1,k,1) + cr3
               tr3 = Cc(i-1,k,1) - cr3
               Ch(i-1,1,k) = tr1 + tr2
               Ch(ic-1,4,k) = tr2 - tr1
               Ch(i,1,k) = ti1 + ti2
               Ch(ic,4,k) = ti1 - ti2
               Ch(i-1,3,k) = ti4 + tr3
               Ch(ic-1,2,k) = tr3 - ti4
               Ch(i,3,k) = tr4 + ti3
               Ch(ic,2,k) = tr4 - ti3
            enddo
         enddo
         if ( mod(Ido,2)==1 ) return
      endif
      do k = 1 , l1
         ti1 = -hsqt2*(Cc(Ido,k,2)+Cc(Ido,k,4))
         tr1 = hsqt2*(Cc(Ido,k,2)-Cc(Ido,k,4))
         Ch(Ido,1,k) = tr1 + Cc(Ido,k,1)
         Ch(Ido,3,k) = Cc(Ido,k,1) - tr1
         Ch(1,2,k) = ti1 - Cc(Ido,k,3)
         Ch(1,4,k) = ti1 + Cc(Ido,k,3)
      enddo
      end subroutine radf4


	subroutine radf5(Ido,l1,Cc,Ch,Wa1,Wa2,Wa3,Wa4)
      implicit none
      integer,parameter :: rk = kind(1.0d0)
      real(rk) :: Cc , Ch , ci2 , ci3 , ci4 , ci5 , cr2 , cr3 , 
     .             cr4 , cr5 , di2 , di3 , di4 , di5 , dr2 , dr3 , 
     .             dr4 , dr5
      real(rk) :: ti2 , ti3 , ti4 , ti5 , tr2 , tr3, 
     .             tr4 , tr5 , Wa1 , Wa2 , Wa3 , Wa4
      integer :: i , ic , Ido , idp2 , k , l1
      dimension Cc(Ido,l1,5) , Ch(Ido,5,l1) , Wa1(*) , Wa2(*) , Wa3(*), 
     .           Wa4(*)
      real(rk),parameter :: pi = acos(-1.0_rk)
      real(rk),parameter :: tr11 = cos(2.0_rk * pi / 5.0_rk)
      real(rk),parameter :: ti11 = sin(2.0_rk * pi / 5.0_rk)
      real(rk),parameter :: tr12 = cos(4.0_rk * pi / 5.0_rk)
      real(rk),parameter :: ti12 = sin(4.0_rk * pi / 5.0_rk)

      do k = 1 , l1
         cr2 = Cc(1,k,5) + Cc(1,k,2)
         ci5 = Cc(1,k,5) - Cc(1,k,2)
         cr3 = Cc(1,k,4) + Cc(1,k,3)
         ci4 = Cc(1,k,4) - Cc(1,k,3)
         Ch(1,1,k) = Cc(1,k,1) + cr2 + cr3
         Ch(Ido,2,k) = Cc(1,k,1) + tr11*cr2 + tr12*cr3
         Ch(1,3,k) = ti11*ci5 + ti12*ci4
         Ch(Ido,4,k) = Cc(1,k,1) + tr12*cr2 + tr11*cr3
         Ch(1,5,k) = ti12*ci5 - ti11*ci4
      enddo
      if ( Ido==1 ) return
      idp2 = Ido + 2
      do k = 1 , l1
         do i = 3 , Ido , 2
            ic = idp2 - i
            dr2 = Wa1(i-2)*Cc(i-1,k,2) + Wa1(i-1)*Cc(i,k,2)
            di2 = Wa1(i-2)*Cc(i,k,2) - Wa1(i-1)*Cc(i-1,k,2)
            dr3 = Wa2(i-2)*Cc(i-1,k,3) + Wa2(i-1)*Cc(i,k,3)
            di3 = Wa2(i-2)*Cc(i,k,3) - Wa2(i-1)*Cc(i-1,k,3)
            dr4 = Wa3(i-2)*Cc(i-1,k,4) + Wa3(i-1)*Cc(i,k,4)
            di4 = Wa3(i-2)*Cc(i,k,4) - Wa3(i-1)*Cc(i-1,k,4)
            dr5 = Wa4(i-2)*Cc(i-1,k,5) + Wa4(i-1)*Cc(i,k,5)
            di5 = Wa4(i-2)*Cc(i,k,5) - Wa4(i-1)*Cc(i-1,k,5)
            cr2 = dr2 + dr5
            ci5 = dr5 - dr2
            cr5 = di2 - di5
            ci2 = di2 + di5
            cr3 = dr3 + dr4
            ci4 = dr4 - dr3
            cr4 = di3 - di4
            ci3 = di3 + di4
            Ch(i-1,1,k) = Cc(i-1,k,1) + cr2 + cr3
            Ch(i,1,k) = Cc(i,k,1) + ci2 + ci3
            tr2 = Cc(i-1,k,1) + tr11*cr2 + tr12*cr3
            ti2 = Cc(i,k,1) + tr11*ci2 + tr12*ci3
            tr3 = Cc(i-1,k,1) + tr12*cr2 + tr11*cr3
            ti3 = Cc(i,k,1) + tr12*ci2 + tr11*ci3
            tr5 = ti11*cr5 + ti12*cr4
            ti5 = ti11*ci5 + ti12*ci4
            tr4 = ti12*cr5 - ti11*cr4
            ti4 = ti12*ci5 - ti11*ci4
            Ch(i-1,3,k) = tr2 + tr5
            Ch(ic-1,2,k) = tr2 - tr5
            Ch(i,3,k) = ti2 + ti5
            Ch(ic,2,k) = ti5 - ti2
            Ch(i-1,5,k) = tr3 + tr4
            Ch(ic-1,4,k) = tr3 - tr4
            Ch(i,5,k) = ti3 + ti4
            Ch(ic,4,k) = ti4 - ti3
         enddo
      enddo
      end subroutine radf5
