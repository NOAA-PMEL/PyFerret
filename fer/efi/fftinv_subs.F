
      SUBROUTINE FFTINV (ND, X, a, b, WFT, IWFT)
      IMPLICIT NONE
      real X(*), a(*), b(*), WFT(*)
      integer nd, IWFT(*)
      integer nf, i, j

C  From snsr, by Jim Larsen, PMEL 
C  Ansley Manke: 11/2001 Change to undo scaling in FOUR_RE
c   calls NCAR FFT code
C  11/2022 *acm* Code cleanup: Update FFTPACK code to F90 standard, from
C                https://github.com/fortran-lang/fftpack.git
C                with one change: instead of the module include  "use fftpack_kind",
C                just define rk as done in rk.f90

c   Calls:  RFFTB

      NF = ND/ 2 

      X(1) = 0.0
      X(2) = a(1)

      DO I = 1, NF-1 
        J = 2* i
        X(J)   = 0.5* a(i) 
        X(J+1) = -0.5* b(i) 
      ENDDO

      X(ND) = a(nf)

      CALL RFFTB (ND, X, WFT, IWFT) 

      RETURN 
      END

C     SUBROUTINE RFFTB(N,R,WSAVE, IWFT)                                               
C                                                                               
C     SUBROUTINE RFFTB COMPUTES THE REAL PERODIC SEQUENCE FROM ITS              
C     FOURIER COEFFICIENTS (FOURIER SYNTHESIS). THE TRANSFORM IS DEFINED        
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
C     WSAVE   A WORK ARRAY WHICH MUST BE DIMENSIONED AT LEAST 2*N+15.           
C             IN THE PROGRAM THAT CALLS RFFTB. THE WSAVE ARRAY MUST BE          
C             INITIALIZED BY CALLING SUBROUTINE RFFTI(N,WSAVE) AND A            
C             DIFFERENT WSAVE ARRAY MUST BE USED FOR EACH DIFFERENT             
C             VALUE OF N. THIS INITIALIZATION DOES NOT HAVE TO BE               
C             REPEATED SO LONG AS N REMAINS UNCHANGED THUS SUBSEQUENT           
C             TRANSFORMS CAN BE OBTAINED FASTER THAN THE FIRST.                 
C             THE SAME WSAVE ARRAY CAN BE USED BY RFFTF AND RFFTB.              
C                                                                               
C                                                                               
C     OUTPUT PARAMETERS                                                         
C                                                                               
C     R       FOR N EVEN AND FOR I = 1,...,N                                    
C                                                                               
C                  R(I) = R(1)+(-1)**(I-1)*R(N)                                 
C                                                                               
C                       PLUS THE SUM FROM K=2 TO K=N/2 OF                       
C                                                                               
C                        2.*R(2*K-2)*COS((K-1)*(I-1)*2*PI/N)                    
C                                                                               
C                       -2.*R(2*K-1)*SIN((K-1)*(I-1)*2*PI/N)                    
C                                                                               
C             FOR N ODD AND FOR I = 1,...,N                                     
C                                                                               
C                  R(I) = R(1) PLUS THE SUM FROM K=2 TO K=(N+1)/2 OF            
C                                                                               
C                       2.*R(2*K-2)*COS((K-1)*(I-1)*2*PI/N)                     
C                                                                               
C                      -2.*R(2*K-1)*SIN((K-1)*(I-1)*2*PI/N)                     
C                                                                               
C      *****  NOTE                                                              
C                  THIS TRANSFORM IS UNNORMALIZED SINCE A CALL OF RFFTF         
C                  FOLLOWED BY A CALL OF RFFTB WILL MULTIPLY THE INPUT          
C                  SEQUENCE BY N.                                               
C                                                                               
C     WSAVE   CONTAINS RESULTS WHICH MUST NOT BE DESTROYED BETWEEN              
C             CALLS OF RFFTB OR RFFTF.                                          
C                                                                               
C                                                                               
      SUBROUTINE RFFTB (N,R,WSAVE, IWFT)                                              
      IMPLICIT NONE
      INTEGER N, IWFT(*)
      REAL R(*) ,WSAVE(*)     
C                                                                               
      IF (N .EQ. 1) RETURN                                                      
      CALL RFFTB1 (N,R,WSAVE,WSAVE(N+1),IWFT)                           
      RETURN                                                                    
      END     
      
      subroutine rfftb1(n,c,Ch,Wa,Ifac)

      implicit none
      integer,parameter :: rk = kind(1.0d0)
      real(rk) :: c , Ch , Wa
      integer :: i , idl1 , ido , Ifac , ip , iw , ix2 , ix3 , ix4 , k1 , 
     .       l1 , l2 , n , na , nf
      dimension Ch(*) , c(*) , Wa(*) , Ifac(*)
      nf = Ifac(2)
      na = 0
      l1 = 1
      iw = 1
      do k1 = 1 , nf
         if (nf+2 .gt. 15) then
            print *, 'fftinv_subs: array Ifac in FFTPACK too large for array', nf+2
            stop
         endif
         ip = Ifac(k1+2)
         l2 = ip*l1
         ido = n/l2
         idl1 = ido*l1
         if ( ip==4 ) then
            ix2 = iw + ido
            ix3 = ix2 + ido
            if ( na/=0 ) then
               call radb4(ido,l1,Ch,c,Wa(iw),Wa(ix2),Wa(ix3))
            else
               call radb4(ido,l1,c,Ch,Wa(iw),Wa(ix2),Wa(ix3))
            endif
            na = 1 - na
         elseif ( ip==2 ) then
            if ( na/=0 ) then
               call radb2(ido,l1,Ch,c,Wa(iw))
            else
               call radb2(ido,l1,c,Ch,Wa(iw))
            endif
            na = 1 - na
         elseif ( ip==3 ) then
            ix2 = iw + ido
            if ( na/=0 ) then
               call radb3(ido,l1,Ch,c,Wa(iw),Wa(ix2))
            else
               call radb3(ido,l1,c,Ch,Wa(iw),Wa(ix2))
            endif
            na = 1 - na
         elseif ( ip/=5 ) then
            if ( na/=0 ) then
               call radbg(ido,ip,l1,idl1,Ch,Ch,Ch,c,c,Wa(iw))
            else
               call radbg(ido,ip,l1,idl1,c,c,c,Ch,Ch,Wa(iw))
            endif
            if ( ido==1 ) na = 1 - na
         else
            ix2 = iw + ido
            ix3 = ix2 + ido
            ix4 = ix3 + ido
            if ( na/=0 ) then
               call radb5(ido,l1,Ch,c,Wa(iw),Wa(ix2),Wa(ix3),Wa(ix4))
            else
               call radb5(ido,l1,c,Ch,Wa(iw),Wa(ix2),Wa(ix3),Wa(ix4))
            endif
            na = 1 - na
         endif
         l1 = l2
         iw = iw + (ip-1)*ido
      enddo
      if ( na==0 ) return
      do i = 1 , n
         c(i) = Ch(i)
      enddo
      end subroutine rfftb1
      
      subroutine radb2(Ido,l1,Cc,Ch,Wa1)
      implicit none
      integer,parameter :: rk = kind(1.0d0)
      real(rk) :: Cc , Ch , ti2 , tr2 , Wa1
      integer :: i , ic , Ido , idp2 , k , l1
      dimension Cc(Ido,2,l1) , Ch(Ido,l1,2) , Wa1(*)
      do k = 1 , l1
         Ch(1,k,1) = Cc(1,1,k) + Cc(Ido,2,k)
         Ch(1,k,2) = Cc(1,1,k) - Cc(Ido,2,k)
      enddo
      if ( Ido<2 ) return
      if ( Ido/=2 ) then
         idp2 = Ido + 2
         do k = 1 , l1
            do i = 3 , Ido , 2
               ic = idp2 - i
               Ch(i-1,k,1) = Cc(i-1,1,k) + Cc(ic-1,2,k)
               tr2 = Cc(i-1,1,k) - Cc(ic-1,2,k)
               Ch(i,k,1) = Cc(i,1,k) - Cc(ic,2,k)
               ti2 = Cc(i,1,k) + Cc(ic,2,k)
               Ch(i-1,k,2) = Wa1(i-2)*tr2 - Wa1(i-1)*ti2
               Ch(i,k,2) = Wa1(i-2)*ti2 + Wa1(i-1)*tr2
            enddo
         enddo
         if ( mod(Ido,2)==1 ) return
      endif
      do k = 1 , l1
         Ch(Ido,k,1) = Cc(Ido,1,k) + Cc(Ido,1,k)
         Ch(Ido,k,2) = -(Cc(1,2,k)+Cc(1,2,k))
      enddo
      end subroutine radb2

      subroutine radb3(Ido,l1,Cc,Ch,Wa1,Wa2)
      implicit none
      integer,parameter :: rk = kind(1.0d0)
      real(rk) :: Cc , Ch , ci2 , ci3 , cr2 , cr3 , di2 , di3 , 
     .        dr2 , dr3 , ti2 , tr2 , Wa1 , Wa2
      integer :: i , ic , Ido , idp2 , k , l1
      dimension Cc(Ido,3,l1) , Ch(Ido,l1,3) , Wa1(*) , Wa2(*)
      real(rk),parameter :: taur = - 0.5_rk
      real(rk),parameter :: taui = sqrt(3.0_rk) / 2.0_rk
      do k = 1 , l1
         tr2 = Cc(Ido,2,k) + Cc(Ido,2,k)
         cr2 = Cc(1,1,k) + taur*tr2
         Ch(1,k,1) = Cc(1,1,k) + tr2
         ci3 = taui*(Cc(1,3,k)+Cc(1,3,k))
         Ch(1,k,2) = cr2 - ci3
         Ch(1,k,3) = cr2 + ci3
      enddo
      if ( Ido==1 ) return
      idp2 = Ido + 2
      do k = 1 , l1
         do i = 3 , Ido , 2
            ic = idp2 - i
            tr2 = Cc(i-1,3,k) + Cc(ic-1,2,k)
            cr2 = Cc(i-1,1,k) + taur*tr2
            Ch(i-1,k,1) = Cc(i-1,1,k) + tr2
            ti2 = Cc(i,3,k) - Cc(ic,2,k)
            ci2 = Cc(i,1,k) + taur*ti2
            Ch(i,k,1) = Cc(i,1,k) + ti2
            cr3 = taui*(Cc(i-1,3,k)-Cc(ic-1,2,k))
            ci3 = taui*(Cc(i,3,k)+Cc(ic,2,k))
            dr2 = cr2 - ci3
            dr3 = cr2 + ci3
            di2 = ci2 + cr3
            di3 = ci2 - cr3
            Ch(i-1,k,2) = Wa1(i-2)*dr2 - Wa1(i-1)*di2
            Ch(i,k,2) = Wa1(i-2)*di2 + Wa1(i-1)*dr2
            Ch(i-1,k,3) = Wa2(i-2)*dr3 - Wa2(i-1)*di3
            Ch(i,k,3) = Wa2(i-2)*di3 + Wa2(i-1)*dr3
         enddo
      enddo
      end subroutine radb3
      
      subroutine radb4(Ido,l1,Cc,Ch,Wa1,Wa2,Wa3)
      implicit none
      integer,parameter :: rk = kind(1.0d0)
      real(rk) :: Cc , Ch , ci2 , ci3 , ci4 , cr2 , cr3 , cr4 , 
     .        ti1 , ti2 , ti3 , ti4 , tr1 , tr2 , tr3, 
     .        tr4 , Wa1 , Wa2 , Wa3
      integer :: i , ic , Ido , idp2 , k , l1
      dimension Cc(Ido,4,l1) , Ch(Ido,l1,4) , Wa1(*) , Wa2(*) , Wa3(*)
      real(rk),parameter :: sqrt2 = sqrt(2.0_rk)
      do k = 1 , l1
         tr1 = Cc(1,1,k) - Cc(Ido,4,k)
         tr2 = Cc(1,1,k) + Cc(Ido,4,k)
         tr3 = Cc(Ido,2,k) + Cc(Ido,2,k)
         tr4 = Cc(1,3,k) + Cc(1,3,k)
         Ch(1,k,1) = tr2 + tr3
         Ch(1,k,2) = tr1 - tr4
         Ch(1,k,3) = tr2 - tr3
         Ch(1,k,4) = tr1 + tr4
      enddo
      if ( Ido<2 ) return
      if ( Ido/=2 ) then
         idp2 = Ido + 2
         do k = 1 , l1
            do i = 3 , Ido , 2
               ic = idp2 - i
               ti1 = Cc(i,1,k) + Cc(ic,4,k)
               ti2 = Cc(i,1,k) - Cc(ic,4,k)
               ti3 = Cc(i,3,k) - Cc(ic,2,k)
               tr4 = Cc(i,3,k) + Cc(ic,2,k)
               tr1 = Cc(i-1,1,k) - Cc(ic-1,4,k)
               tr2 = Cc(i-1,1,k) + Cc(ic-1,4,k)
               ti4 = Cc(i-1,3,k) - Cc(ic-1,2,k)
               tr3 = Cc(i-1,3,k) + Cc(ic-1,2,k)
               Ch(i-1,k,1) = tr2 + tr3
               cr3 = tr2 - tr3
               Ch(i,k,1) = ti2 + ti3
               ci3 = ti2 - ti3
               cr2 = tr1 - tr4
               cr4 = tr1 + tr4
               ci2 = ti1 + ti4
               ci4 = ti1 - ti4
               Ch(i-1,k,2) = Wa1(i-2)*cr2 - Wa1(i-1)*ci2
               Ch(i,k,2) = Wa1(i-2)*ci2 + Wa1(i-1)*cr2
               Ch(i-1,k,3) = Wa2(i-2)*cr3 - Wa2(i-1)*ci3
               Ch(i,k,3) = Wa2(i-2)*ci3 + Wa2(i-1)*cr3
               Ch(i-1,k,4) = Wa3(i-2)*cr4 - Wa3(i-1)*ci4
               Ch(i,k,4) = Wa3(i-2)*ci4 + Wa3(i-1)*cr4
            enddo
         enddo
         if ( mod(Ido,2)==1 ) return
      endif
      do k = 1 , l1
         ti1 = Cc(1,2,k) + Cc(1,4,k)
         ti2 = Cc(1,4,k) - Cc(1,2,k)
         tr1 = Cc(Ido,1,k) - Cc(Ido,3,k)
         tr2 = Cc(Ido,1,k) + Cc(Ido,3,k)
         Ch(Ido,k,1) = tr2 + tr2
         Ch(Ido,k,2) = sqrt2*(tr1-ti1)
         Ch(Ido,k,3) = ti2 + ti2
         Ch(Ido,k,4) = -sqrt2*(tr1+ti1)
      enddo
      end subroutine radb4
      
      subroutine radb5(Ido,l1,Cc,Ch,Wa1,Wa2,Wa3,Wa4)
      implicit none
      integer,parameter :: rk = kind(1.0d0)
      real(rk) :: Cc , Ch , ci2 , ci3 , ci4 , ci5 , cr2 , cr3 , 
     .        cr4 , cr5 , di2 , di3 , di4 , di5 , dr2 , dr3 , 
     .        dr4 , dr5
      real(rk) :: ti2 , ti3 , ti4 , ti5 , tr2 , tr3, 
     .        tr4 , tr5 , Wa1 , Wa2 , Wa3 , Wa4
      integer :: i , ic , Ido , idp2 , k , l1
      dimension Cc(Ido,5,l1) , Ch(Ido,l1,5) , Wa1(*) , Wa2(*) , Wa3(*), 
     .      Wa4(*)
      real(rk),parameter :: pi = acos(-1.0_rk)
      real(rk),parameter :: tr11 = cos(2.0_rk * pi / 5.0_rk)
      real(rk),parameter :: ti11 = sin(2.0_rk * pi / 5.0_rk)
      real(rk),parameter :: tr12 = cos(4.0_rk * pi / 5.0_rk)
      real(rk),parameter :: ti12 = sin(4.0_rk * pi / 5.0_rk)
      do k = 1 , l1
         ti5 = Cc(1,3,k) + Cc(1,3,k)
         ti4 = Cc(1,5,k) + Cc(1,5,k)
         tr2 = Cc(Ido,2,k) + Cc(Ido,2,k)
         tr3 = Cc(Ido,4,k) + Cc(Ido,4,k)
         Ch(1,k,1) = Cc(1,1,k) + tr2 + tr3
         cr2 = Cc(1,1,k) + tr11*tr2 + tr12*tr3
         cr3 = Cc(1,1,k) + tr12*tr2 + tr11*tr3
         ci5 = ti11*ti5 + ti12*ti4
         ci4 = ti12*ti5 - ti11*ti4
         Ch(1,k,2) = cr2 - ci5
         Ch(1,k,3) = cr3 - ci4
         Ch(1,k,4) = cr3 + ci4
         Ch(1,k,5) = cr2 + ci5
      enddo
      if ( Ido==1 ) return
      idp2 = Ido + 2
      do k = 1 , l1
         do i = 3 , Ido , 2
            ic = idp2 - i
            ti5 = Cc(i,3,k) + Cc(ic,2,k)
            ti2 = Cc(i,3,k) - Cc(ic,2,k)
            ti4 = Cc(i,5,k) + Cc(ic,4,k)
            ti3 = Cc(i,5,k) - Cc(ic,4,k)
            tr5 = Cc(i-1,3,k) - Cc(ic-1,2,k)
            tr2 = Cc(i-1,3,k) + Cc(ic-1,2,k)
            tr4 = Cc(i-1,5,k) - Cc(ic-1,4,k)
            tr3 = Cc(i-1,5,k) + Cc(ic-1,4,k)
            Ch(i-1,k,1) = Cc(i-1,1,k) + tr2 + tr3
            Ch(i,k,1) = Cc(i,1,k) + ti2 + ti3
            cr2 = Cc(i-1,1,k) + tr11*tr2 + tr12*tr3
            ci2 = Cc(i,1,k) + tr11*ti2 + tr12*ti3
            cr3 = Cc(i-1,1,k) + tr12*tr2 + tr11*tr3
            ci3 = Cc(i,1,k) + tr12*ti2 + tr11*ti3
            cr5 = ti11*tr5 + ti12*tr4
            ci5 = ti11*ti5 + ti12*ti4
            cr4 = ti12*tr5 - ti11*tr4
            ci4 = ti12*ti5 - ti11*ti4
            dr3 = cr3 - ci4
            dr4 = cr3 + ci4
            di3 = ci3 + cr4
            di4 = ci3 - cr4
            dr5 = cr2 + ci5
            dr2 = cr2 - ci5
            di5 = ci2 - cr5
            di2 = ci2 + cr5
            Ch(i-1,k,2) = Wa1(i-2)*dr2 - Wa1(i-1)*di2
            Ch(i,k,2) = Wa1(i-2)*di2 + Wa1(i-1)*dr2
            Ch(i-1,k,3) = Wa2(i-2)*dr3 - Wa2(i-1)*di3
            Ch(i,k,3) = Wa2(i-2)*di3 + Wa2(i-1)*dr3
            Ch(i-1,k,4) = Wa3(i-2)*dr4 - Wa3(i-1)*di4
            Ch(i,k,4) = Wa3(i-2)*di4 + Wa3(i-1)*dr4
            Ch(i-1,k,5) = Wa4(i-2)*dr5 - Wa4(i-1)*di5
            Ch(i,k,5) = Wa4(i-2)*di5 + Wa4(i-1)*dr5
         enddo
      enddo
      end subroutine radb5
      
      subroutine radbg(Ido,Ip,l1,Idl1,Cc,c1,c2,Ch,Ch2,Wa)
      implicit none
      integer,parameter :: rk = kind(1.0d0)
      real(rk) :: ai1 , ai2 , ar1 , ar1h , ar2 , ar2h , arg , c1 , 
     .        c2 , Cc , Ch , Ch2 , dc2 , dcp , ds2 , dsp , 
     .        Wa
      integer :: i , ic , idij , Idl1 , Ido , idp2 , ik , Ip , ipp2 , 
     .       ipph , is , j , j2 , jc , k , l , l1 , lc , nbd
      dimension Ch(Ido,l1,Ip) , Cc(Ido,Ip,l1) , c1(Ido,l1,Ip) , 
     .      c2(Idl1,Ip) , Ch2(Idl1,Ip) , Wa(*)
      real(rk),parameter :: tpi = 2*acos(-1.0_rk) ! 2 * pi
      arg = tpi/real(Ip, rk)
      dcp = cos(arg)
      dsp = sin(arg)
      idp2 = Ido + 2
      nbd = (Ido-1)/2
      ipp2 = Ip + 2
      ipph = (Ip+1)/2
      if ( Ido<l1 ) then
         do i = 1 , Ido
            do k = 1 , l1
               Ch(i,k,1) = Cc(i,1,k)
            enddo
         enddo
      else
         do k = 1 , l1
            do i = 1 , Ido
               Ch(i,k,1) = Cc(i,1,k)
            enddo
         enddo
      endif
      do j = 2 , ipph
         jc = ipp2 - j
         j2 = j + j
         do k = 1 , l1
            Ch(1,k,j) = Cc(Ido,j2-2,k) + Cc(Ido,j2-2,k)
            Ch(1,k,jc) = Cc(1,j2-1,k) + Cc(1,j2-1,k)
         enddo
      enddo
      if ( Ido/=1 ) then
         if ( nbd<l1 ) then
            do j = 2 , ipph
               jc = ipp2 - j
               do i = 3 , Ido , 2
                  ic = idp2 - i
                  do k = 1 , l1
                     Ch(i-1,k,j) = Cc(i-1,2*j-1,k) + Cc(ic-1,2*j-2,k)
                     Ch(i-1,k,jc) = Cc(i-1,2*j-1,k) - Cc(ic-1,2*j-2,k)
                     Ch(i,k,j) = Cc(i,2*j-1,k) - Cc(ic,2*j-2,k)
                     Ch(i,k,jc) = Cc(i,2*j-1,k) + Cc(ic,2*j-2,k)
                  enddo
               enddo
            enddo
         else
            do j = 2 , ipph
               jc = ipp2 - j
               do k = 1 , l1
                  do i = 3 , Ido , 2
                     ic = idp2 - i
                     Ch(i-1,k,j) = Cc(i-1,2*j-1,k) + Cc(ic-1,2*j-2,k)
                     Ch(i-1,k,jc) = Cc(i-1,2*j-1,k) - Cc(ic-1,2*j-2,k)
                     Ch(i,k,j) = Cc(i,2*j-1,k) - Cc(ic,2*j-2,k)
                     Ch(i,k,jc) = Cc(i,2*j-1,k) + Cc(ic,2*j-2,k)
                  enddo
               enddo
            enddo
         endif
      endif
      ar1 = 1.0_rk
      ai1 = 0.0_rk
      do l = 2 , ipph
         lc = ipp2 - l
         ar1h = dcp*ar1 - dsp*ai1
         ai1 = dcp*ai1 + dsp*ar1
         ar1 = ar1h
         do ik = 1 , Idl1
            c2(ik,l) = Ch2(ik,1) + ar1*Ch2(ik,2)
            c2(ik,lc) = ai1*Ch2(ik,Ip)
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
               c2(ik,l) = c2(ik,l) + ar2*Ch2(ik,j)
               c2(ik,lc) = c2(ik,lc) + ai2*Ch2(ik,jc)
            enddo
         enddo
      enddo
      do j = 2 , ipph
         do ik = 1 , Idl1
            Ch2(ik,1) = Ch2(ik,1) + Ch2(ik,j)
         enddo
      enddo
      do j = 2 , ipph
         jc = ipp2 - j
         do k = 1 , l1
            Ch(1,k,j) = c1(1,k,j) - c1(1,k,jc)
            Ch(1,k,jc) = c1(1,k,j) + c1(1,k,jc)
         enddo
      enddo
      if ( Ido/=1 ) then
         if ( nbd<l1 ) then
            do j = 2 , ipph
               jc = ipp2 - j
               do i = 3 , Ido , 2
                  do k = 1 , l1
                     Ch(i-1,k,j) = c1(i-1,k,j) - c1(i,k,jc)
                     Ch(i-1,k,jc) = c1(i-1,k,j) + c1(i,k,jc)
                     Ch(i,k,j) = c1(i,k,j) + c1(i-1,k,jc)
                     Ch(i,k,jc) = c1(i,k,j) - c1(i-1,k,jc)
                  enddo
               enddo
            enddo
         else
            do j = 2 , ipph
               jc = ipp2 - j
               do k = 1 , l1
                  do i = 3 , Ido , 2
                     Ch(i-1,k,j) = c1(i-1,k,j) - c1(i,k,jc)
                     Ch(i-1,k,jc) = c1(i-1,k,j) + c1(i,k,jc)
                     Ch(i,k,j) = c1(i,k,j) + c1(i-1,k,jc)
                     Ch(i,k,jc) = c1(i,k,j) - c1(i-1,k,jc)
                  enddo
               enddo
            enddo
         endif
      endif
      if ( Ido==1 ) return
      do ik = 1 , Idl1
         c2(ik,1) = Ch2(ik,1)
      enddo
      do j = 2 , Ip
         do k = 1 , l1
            c1(1,k,j) = Ch(1,k,j)
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
                  c1(i-1,k,j) = Wa(idij-1)*Ch(i-1,k,j) - Wa(idij) 
     .                      *Ch(i,k,j)
                  c1(i,k,j) = Wa(idij-1)*Ch(i,k,j) + Wa(idij) 
     .                    *Ch(i-1,k,j)
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
                  c1(i-1,k,j) = Wa(idij-1)*Ch(i-1,k,j) - Wa(idij) 
     .                      *Ch(i,k,j)
                  c1(i,k,j) = Wa(idij-1)*Ch(i,k,j) + Wa(idij) 
     .                    *Ch(i-1,k,j)
               enddo
            enddo
         enddo
      endif
      end subroutine radbg