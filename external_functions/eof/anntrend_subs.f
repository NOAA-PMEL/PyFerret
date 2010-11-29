

c -+-+ -+-+ -+-+ -+-+ -+-+ -+-+ -+-+ -+-+ -+-+ -+-+ -+-+ -+-+ -+-+ -+-+ -+-+
      subroutine annualdai (fq, nd, x, r, gap, nf, flag)

c   computes annual variation r by robust least squares fit
c   of nf sine and cosine terms to x and remove r from series x.

c  calls: grossout, qr 

      parameter (mf = 5, mt = 2* mf)
      dimension x(*), r(*), gap(*)

      real*8  g(mt,mt), a(mt), b(mt), cw(mt), sw(mt), cwa(mt), swa(mt), 
     .        cm(mt), sm(mt), pi, freq, c, s, c2, cs, ss, cd, sd, cs2, 
     .        cd2, v1, v2, v3, u1, u2, u3, g1, g2, g3, g4, gi, wi, 
     .        vpy, ar, ai
      real fq, flag
      integer nd, nf

c   number nf of frequencies for annual & higher harmonics

      vpy = 365.25* 24.0* 7200.0* fq
      year = nd/ vpy
      if (year .lt. 3.0) return

      pi = 3.1415926535897932384

c   average over nave terms and decimate series

      nave = vpy/ (2.0* (nf + 1))
      if (nave .lt. 1) nave = 1
      nda  = nd/ nave
      
      if (nave .eq. 1) then
        do i = 1, nd
          r(i) = x(i)
        enddo
      else
        j2 = 0
        do i = 1, nda
          j1 = j2 + 1
          j2 = j2 + nave
          n  = 0
          s  = 0.d0
          do j = j1, j2
            xj = x(j)
            if (xj .ne. flag) then
              n = n + 1
              s = s + xj
            endif
          enddo
          if (n .gt. 0) then
            r(i) = s/ n
          else
            r(i) = flag
          endif
        enddo
      endif

c   outliers

      call grossout (nda, 5.0, r, gap, flag)

      ar = 0.d0
      nr = 0
      do i = 1, nda
        ri = r(i)
        if (ri .ne. flag) then
          gap(i) = 1.0
          nr = nr + 1
          ar = ar + ri
        else
          r(i)   = 0.0
          gap(i) = 0.0
        endif
      enddo

      aa = ar/ nr
      do i = 1, nda
        if (r(i) .ne. flag) r(i) = r(i) - aa
      enddo

      do n = 1, nf
        freq   = 2.d0* pi* n/ vpy
        cw(n)  = cos (freq)
        sw(n)  = sin (freq)
        cwa(n) = cos (nave* freq)
        swa(n) = sin (nave* freq)
      enddo

      do n = 1, nf
        m = n + nf
        c  = cwa(n)
        s  = swa(n)
        c2 = 2.d0* c
        u1 = 0.d0
        u2 = r(nda)
        do i = nda - 1, 2, -1
          u3 = c2* u2 - u1 + r(i)
          u1 = u2
          u2 = u3
        enddo
        a(n) = c* u2 - u1 + r(1)
        a(m) = s* u2
      enddo

c   mean of terms set equal to zero

      do n = 1, nf
        c  = cwa(n)
        s  = swa(n)
        c2 = 2.d0* c
        u1 = 0.d0
        u2 = gap(nda)
        do i = nda - 1, 2, -1
          u3 = c2* u2 - u1 + gap(i)
          u1 = u2
          u2 = u3
        enddo
        cm(n) = c* u2 - u1 + gap(1)
        sm(n) = s* u2
      enddo

      do m = 1, nf
        mm = m + nf
        do n = 1, m
          nn = n + nf
          c  = cwa(m)* cwa(n)
          s  = swa(m)* swa(n)
          cs = c - s
          cd = c + s
          s  = swa(m)* cwa(n)
          c  = cwa(m)* swa(n)
          ss = s + c
          sd = s - c
          cs2 = 2.d0* cs
          cd2 = 2.d0* cd
          gi = gap(nda)
          u1 = 0.d0
          u2 = gi
          v1 = 0.d0
          v2 = gi
          do i = nda - 1, 2, -1
            gi = gap(i)
            u3 = cs2* u2 - u1 + gi
            v3 = cd2* v2 - v1 + gi
            u1 = u2
            u2 = u3
            v1 = v2
            v2 = v3
          enddo

          g1 = gap(1) + 0.5d0* (cs* u2 - u1 + cd* v2 - v1)
          g2 =          0.5d0* (ss* u2 - sd* v2)
          g3 =          0.5d0* (ss* u2 + sd* v2)
          g4 =          0.5d0* (cd* v2 - v1 - cs* u2 + u1)

          g1 = g1 - cm(m)* cm(n)/ nr
          g2 = g2 - cm(m)* sm(n)/ nr
          g3 = g3 - sm(m)* cm(n)/ nr
          g4 = g4 - sm(m)* sm(n)/ nr

          g( m, n) = g1
          g( n, m) = g1
          g( m,nn) = g2
          g(nn, m) = g2
          g(mm, n) = g3
          g( n,mm) = g3
          g(mm,nn) = g4
          g(nn,mm) = g4

        enddo
      enddo

      nt = 2* nf

      call qr (mt, nt, nt, g, a, b, resq)

c   correction for averaging

      if (nave .gt. 1) then
        do n = 1, nf
          m = n + nf
            wi = pi* n/ vpy
            ww = nave* sin (wi)/ sin (nave* wi)
            c  = cos ((nave - 1)* wi)
            s  = sin ((nave - 1)* wi)
            ar = b(n)
            ai = b(m)
            b(n) = ww* (c* ar - s* ai)
            b(m) = ww* (s* ar + c* ai)
        enddo
      endif

c   annual variation r

      do i = 1, nd
        r(i) = 0.0
      enddo

      do n = 1, nf
        c  = cw(n)
        s  = sw(n)
        c2 = 2.d0* c
        u1 = b(n)
        u2 = c* u1 + s* b(n + nf)
        r(1) = r(1) + u1
        r(2) = r(2) + u2
        do i = 3, nd
          u3 = c2* u2 - u1
          r(i) = r(i) + u3
          u1 = u2
          u2 = u3
        enddo
      enddo

c   remove annual variation r from series x

      do i = 1, nd
        if (x(i) .ne. flag) x(i) = x(i) - r(i)
      enddo

      return
      end

      subroutine trendflag2 (nd, x, flag)
      real*8 xi, x0, x1, t0, t1, t2, wi, w1, w2, r1, r2, d 
      dimension x(*)
      real flag
      integer nd

      r1 = 1.d0/ nd
      r2 = 2.d0/ nd
      w1 = r2/ (1.d0 - r1)
      w2 = (1.d0 + r1)/ (1.d0 - r1)

      x0 = 0.d0
      x1 = 0.d0
      t0 = 0.d0
      t1 = 0.d0
      t2 = 0.d0
      do i = 1, nd 
        if (x(i) .ne. flag) then
          xi = x(i)
          wi = w1* i - w2
          x0 = x0 + xi 
          x1 = x1 + xi* wi
          t0 = t0 + 1.d0
          t1 = t1 + wi
          t2 = t2 + wi**2
        endif
      enddo
      x0 = x0/ t0
      x1 = x1/ t0
      t1 = t1/ t0
      t2 = t2/ t0

      d  = t2 - t1**2

      c0 = (t2* x0 - t1* x1)/ d
      c1 = (x1 - t1* x0)/ d 

      do i = 1, nd 
        xi = x(i)
        if (xi .ne. flag) then
          fi = w1* i - w2
          x(i) = xi - c1* fi - c0
        endif
      enddo

      return 
      end
      subroutine piw(sst, sstyear, w, ndim, flag)

      integer ndim, j, n

      integer nj, k, j1, j2

      real sst(*), sstyear(*), sd, sw, sj, wk
      real w(50)

      real flag

          sw = 0.0
          sd = 0.0
          nj = 0

          do n = 1, ndim
c            j1 = n - 6
c            j2 = n + 5
            j1 = n - 12
            j2 = n + 11
            nj = 0
            if (j1 .lt. 1) then
              sw = 0.0
              sd = 0.0
            else
              if (j2 .lt. ndim) then
                if(sst(j1)   .ne. flag .and. 
     .             sst(j2+1) .ne. flag) then
                  sd = 0.5* w(1)* (sst(j1) + sst(j2+1))
                  sw = w(1)
                else
                  if(sst(j1) .ne. flag) then
                    sd = w(1)* sst(j1)
                    sw = w(1)
                  else if(sst(j2+1) .ne. flag) then
                    sd = w(1)* sst(j2+1)
                    sw = w(1)
                  endif
                endif
              else
                if(sst(j1) .ne. flag) then
                  sd = w(1)* sst(j1)
                  sw = w(1)
                endif
              endif
              nj = 1
            endif
            k = 1
            do j = j1 + 1, j2
              k = k + 1
              if(j .ge. 1 .and. j .le. ndim) then
                sj = sst(j)
                if(sj .ne. flag) then
                  wk = w(k)
                  nj = nj + 1
                  sw = sw + wk
                  sd = sd + wk*sj
                endif
              endif
            enddo
c            if(sw .ne. 0.0) then
            if(nj .ge. 12 .and. sw .ne. 0.0) then
              sstyear(n) = sd / sw
            else
              sstyear(n) = flag
            endif
          enddo

          do n = 1, ndim
            sst(n) = sstyear(n)
          enddo

          return
          end

      subroutine prolate (w, nd, isw)

c  calculates prolate spheroidal wavefunction data window for
c  high resolution fourier analysis
c  ref: d.j.thomson, bell syst. tech. j. 56,1769-1815 (1977)
c
c  w is a single precison real array of data window values
c  nd is the number of points in w
c  isw is a switch--isw=4 means use a 4-pi window, isw=1 means use
c      the higher resolution pi window
c
c  scale factors=integral(boxcar)/integral(prolate window) are:
c    4 pi prolate window--1.425658520238489
c    pi prolate window--1.057568010371401
c  these are the numbers to multiply the spectrum by for comparison
c  with other windows
c
      ingeger nd, isw
      real w(*)
      real*8 xi, xn, x, d, u, dd, x1, x2
      x1 = dble(1)
      x2 = dble(2)
      xn = dble(nd)/ x2
      if (isw .eq. 4) then
      	d = sqrt (x2/ 0.508125548147497d0)
       	do i = 1, nd
      	  xi = dble(i) - x1
      	  x = xi/ xn - x1
      	  u = (x1 - x)* (x1 + x)
      	  dd = d*(((((((((((((((((((((
     $        2.6197747176990866d-11*u+2.9812025862125737d-10)*u+
     $        3.0793023552299688d-9)*u+2.8727486379692354d-8)*u+
     $        2.4073904863499725d-7)*u+1.8011359410323110d-6)*u+
     $        1.1948784162527709d-5)*u+6.9746276641509466d-5)*u+
     $        3.5507361197109845d-4)*u+1.5607376779150113d-3)*u+
     $        5.8542015072142441d-3)*u+1.8482388295519675d-2)*u+
     $        4.8315671140720506d-2)*u+1.0252816895203814d-1)*u+
     $        1.7233583271499150d-1)*u+2.2242525852102708d-1)*u+
     $        2.1163435697968192d-1)*u+1.4041394473085307d-1)*u+
     $        5.9923940532892353d-2)*u+1.4476509897632850d-2)*u+
     $        1.5672417352380246d-3)*u+4.2904633140034110d-5)
          w(i) = dd
      	enddo
      	return
      else
      	d = sqrt (x2)
      	do i = 1, nd
      	  xi = dble(i) - x1
      	  x = xi/ xn - x1
      	  u = (x1 - x)* (x1 + x)
      	  dd = d*((((((((((
     $        5.3476939016920851d-11*u+2.2654256220146656d-9)*u+
     $        7.8075102004229667d-8)*u+2.1373409644281953d-6)*u+
     $        4.5094847544714943d-5)*u+7.0498957221483167d-4)*u+
     $        7.7412693304064753d-3)*u+5.5280627452077586d-2)*u+
     $        2.2753754228751827d-1)*u+4.3433904277546202d-1)*u+
     $        2.2902051859068017d-1)
          w(i) = dd
        enddo
        return
      endif
      end
      subroutine grossout (nd, sdmax, x, r)

c   finds outliers in x at the sdmax standard deviation level and sets
c   x = flag for outliers

c   calls: robustsd

      dimension  x(*), r(*)
      integer nd
      real sdmax

      jd = 0
      do i = 1, nd
        xi = x(i)
        if (xi .ne. flag) then
          jd = jd + 1
          r(jd) = xi
        endif
      enddo

      call robustsd (jd, r, sd)
        
      jd = 0
      do i = 1, nd
        if (x(i) .ne. flag) then
          jd = jd + 1
          if (abs (r(jd)) .gt. sdmax) x(i) = flag
        endif
      enddo
 
      return
      end

      subroutine qr(ndim, m, n, a, b, x, resq)

      implicit double precision (a-h, o-z)
      integer ndim, m, n

c$$$$  calls no other routines
c  solves over-determined least-squares problem  ax = b
c  where  a  is an  m by n  matrix,  b  is an m-vector .
c  resq  is the sum of squared residuals of optimal solution.  also used
c  to signal error conditions - if -2 , system is underdetermined,  if
c  -1,  system is singular.
c  method - successive householder rotations.  see lawson+hanson - solv
c  -ing least squares problems.
c  routine will also work when m=n.
c*****   caution -  a and b  are overwritten by this routine.

      dimension a(ndim,*), b(*), x(*)
      double precision sum,dot
      real*4 resq
c
      resq=-2.0
      if (m.lt.n) return
c   loop ending on 1800 rotates  a  into upper triangular form
      do 1800 j=1,n
c  find constants for rotation and diagonal entry
      sq=0.0
      do 1100 i=j,m
 1100 sq=a(i,j)**2 + sq

cc      qv1=-sign(dsqrt(sq),a(j,j))
      signa = 1.
      if (abs(a(j,j)) .ne. a(j,j)) signa = -1.
      qv1=-1.* abs(dsqrt(sq)) * signa

      u1=a(j,j) - qv1
      a(j,j)=qv1
      j1=j + 1
      if (j1.gt.n) go to 1500
c  rotate remaining columns of sub-matrix
      do 1400 jj=j1,n
      dot=u1*a(j,jj)
      do 1200 i=j1,m
 1200 dot=a(i,jj)*a(i,j) + dot
      const=dot/dabs(qv1*u1)
      do 1300 i=j1,m
 1300 a(i,jj)=a(i,jj) - const*a(i,j)
      a(j,jj)=a(j,jj) - const*u1
 1400 continue
c  rotate  b  vector
 1500 dot=u1*b(j)
      if (j1.gt.m) go to 1610
      do 1600 i=j1,m
 1600 dot=b(i)*a(i,j) + dot
 1610 const=dot/dabs(qv1*u1)
      b(j)=b(j) - const*u1
      if (j1.gt.m) go to 1800
      do 1700 i=j1,m
 1700 b(i)=b(i) - const*a(i,j)
 1800 continue
c  solve triangular system by back-substitution.
      resq=-1.0
      do 2200 ii=1,n
      i=n-ii+1
      sum=b(i)
      if (ii.eq.1) go to 2110
      i1=i+1
      do 2100 j=i1,n
 2100 sum=sum - a(i,j)*x(j)
 2110 if (a(i,i).eq. 0.0) return
 2200 x(i)=sum/a(i,i)
c  find residual in overdetermined case.
      resq=0.0
      if (m.eq.n) return
      i1=n+1
      m=m
      n=n
      do 2300 i=i1,m
 2300 resq=b(i)**2 + resq
      return
      end

      subroutine robustsd (nd, r, sd)

c   robustly estimated standard deviation sd for nd values of r based on
c   median of the absolute value of the deviations.

c   calls: sort

      dimension r(*), dum(1005)
      integer nd, nh, nmax

      nh = nd/ 2

      smad = 0.67449

      nmax = 1000
      if (nd .le. nmax) then

        do i = 1, nd
          dum(i) = abs (r(i))
        enddo

        call sort (nd, dum)
 
        if (2* nh .eq. nd) then
          sd = 0.5* (dum(nh) + dum(nh + 1))/ smad
        else
          sd = dum(nh + 1)/ smad
        endif

        if (sd .eq. 0.0) then
          do i = 1, nd
            sd = sd + r(i)**2
          enddo
          sd = sqrt (sd/ nd)
        endif

      else

c   preliminary standard deviation
      
        s = 0.0
        do i = 1, nd
          s  = s + r(i)**2
        enddo
        s = sqrt (s/ nd)

        ds = sqrt (2.0* 3.141592654)* s* exp(0.5*smad**2)

        smed = smad* s

        mid = 0.6 + 0.5* nd

c   median absolute deviation

        nmed = -mid
        do i = 1, nd
          if (abs (r(i)) .le. smed) nmed = nmed + 1
        enddo

        if (nmed. eq. 0) go to 20

        ds = - ds* nmed/ nd

    2   sneg = smed
        nneg = nmed
        smed = smed + ds
        nmed = -mid
        do i = 1, nd
          if (abs (r(i)) .le. smed) nmed = nmed + 1
        enddo
cc        if (nneg* sign(1, nmed) .gt. 0) go to 2

        nsigna = 1
        if (abs(nmed) .ne. nmed) nsigna = -1
        if (nneg* nsigna .gt. 0) go to 2

        if (nmed .eq. 0) go to 20

        it = 0
    4   it = it + 1

        if (sneg .gt. smed) then
          sd   = smed
          smed = sneg
          sneg = sd
          nn   = nmed
          nmed = nneg
          nneg = nn
        endif

cc        if (nneg* sign(1, nmed) .lt. 0 .and. nmed - nneg .le. nmax) then

        nsigna = 1
        if (abs(nmed) .ne. nmed) nsigna = -1
        if (nneg* nsigna .gt. 0) go to 2

        if (nneg* nsigna .lt. 0 .and. nmed - nneg .le. nmax) then

          jd = 0
          do i = 1, nd
            ri = abs (r(i))
            if (ri .gt. sneg .and. ri .lt. smed) then
              jd = jd + 1
              dum(jd) = ri
            endif
          enddo

          call sort (jd, dum)

          if (2* nh .eq. nd) then
            smed = 0.5* (dum( -nneg) + dum( -nneg + 1))
          else
            smed = dum( -nneg + 1)
          endif

          go to 20

        endif

        sd = 0.5* (sneg + smed)
        nn = -mid
        do i = 1, nd
          if (abs (r(i)) .le. sd) nn = nn + 1
        enddo

        if (nn .eq. 0) then
          smed = sd
          go to 20
        endif

cc        if (nn* sign(1, nneg) .gt. 0) then

        nsigna = 1
        if (abs(nneg) .ne. nneg) nsigna = -1

        if (nn* nsigna .gt. 0) then
          sneg = sd
          nneg = nn
        else
          smed = sd
          nmed = nn
        endif

        if (it .lt. 10) go to 4

   20   sd = smed/ smad

      endif

      rd = 1.0/ sd
      do i = 1, nd
        r(i) = rd* r(i)
      enddo

      return
      end
      subroutine sort (nd, x)

c   ranks x base on code from numerical recipes.

      dimension x(*)

      if (nd .le. 1) return
      l = nd/2 + 1
      ir = nd

   10 continue
      
      if (l .gt. 1) then
        l = l - 1
        xi = x(l)
      else
        xi = x(ir)
        x(ir) = x(1)
        ir = ir - 1
        if (ir .eq. 1) then
          x(1) = xi
          return
        endif
      endif

      i = l
      j = l + l
   20 if (j .le. ir) then
        if (j .lt. ir) then
          jj = j + 1
          if (x(j) .lt. x(jj)) j = jj
        endif
        if (xi .lt. x(j)) then
          x(i) = x(j)
          i = j
          j = j + j
        else
          j = ir + 1
        endif
        go to 20
      endif

      x(i) = xi

      go to 10
      
      end
