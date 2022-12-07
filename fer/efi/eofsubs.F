C eofsubs.F
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration''s (NOAA) Pacific Marine Environmental Lab(PMEL),
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
C  EOF code from Billy Kessler.  For cases where time series do not
C  have any gaps.  Implemented 7/2001
C
C  At end are routines count_eof and pack_eof written with Dai McClurg
C  to count the good timeseries and pack the X,Y,T  array into an neof by T
C  array.
C
C
C  Ansley Manke       Change dimension statements for arguments to be (*) or
C                     (NX,*), etc.  The work arrays in the calling functions
C                     have extra space to accomodate I = 1,N... BB(I+1) in 
C                     QRSYM, for example.
C
C                     Declare all variables.
C
C Aug 2001 move to fer/efi directory, EPFs to be statically linked to Ferret.  
C	   Change INCLUDE statements to remove directory spec.
C

C v5.41 *kob* 6/02 - remove unneeded ferret_cmn from INCLUDE lines - 
C                    cygwin does not like them for some reason
C
c Date: Thu, 18 Jan 2001 16:44:10 -0800 (PST)
c From: kessler@pmel.noaa.gov (Billy Kessler)
c Subject: more EOF code (reverse time and space)
c 
c The attached code (one file) contains the following subroutines:
c 
c First set are three different front ends that call the
c routines at bottom that do the work:
c 
c EOFIN         ! for usual EOFs (more times than locations)
c EOFIN2	! EOFs when there are more locations than times
c EOFIN3C	! routine that allows weighting the data (probably 
c	           get rid of this one for ferret)
c
c These are the black-box routines that do the actual work
c 
c FRACVAR	! find percent variance at each point
c QRSYM		! call all the other routines in turn
c HOUSEH	! Householder tridiagonalization of matrix
c QRSTD		! find eigenvalues of tridiagonal matrix
c TRIDIN	! find eigenvectors of tridiagonal matrix
c BACKS		! does something
c 
c * Actually, either EOFIN or EOFIN2 produces exactly the same
c result (but see note 1). These two routines do not care about 
c zero eigenvalues, so there is really no restriction at all on 
c their use. However, EOFIN2 will be faster if there are more 
c locations than times. So a possible choice for this whole mess 
c would be just to use EOFIN for all cases (except for gappy time 
c series). That might be the most straightforward thing to do,
c even if it is not the most elegant code.
c 
c Notes: 
c 
c Note 1: The eigenvalues out of EOFIN2 differ from those of EOFIN
c by the factor (NT/NX). THis is due to the fact that there are NT 
c non-zero eigenvalues produced by EOFIN2 (since it assumes that NT<NX), 
c while there are NX produced by EOFIN. However, I have scaled the 
c EOFs and PCs to be identical. This should not matter to anyone.
c 
c Note 3: Differences between the outputs of these three occur for
c the higher eigenvalues/vectors, as the values of the eigenvalues
c come down towards the computational noise and slight differences
c in cutoff values hardwired into the routines. These are iterative
c search routines and I have not tried to make the cutoffs (when the
c routine decides it has found a value) consistent. This should not 
c matter for geophysical work.
c 
c Note 4: Odd results can occur in SVDEOF for input fields that do
c not have NX degrees of freedom. I spent an embarrassingly long
c amount of time this afternoon "debugging" a test program that I
c was feeding a simple idealized (product of sinusoids) field. The
c field could be represented in only a handful of eigenvectors, but
c SVDEOF cranked off NX of them, and the higher eigenvalues were
c pure noise. Some of them were large though, and overwhelmed the
c others. So if you are making up test fields, use the RANDU 
c function to add sufficient noise to "use up" all the eigenvectors.
c This will not be a problem for geophysical fields, which always
c have sufficient noise. For some reason EOFIN and EOFIN2 do not
c appear to have this problem.
c 
c Note 5: I wrote this code in 1982-83, with Nancy Soreide (then a 
c programmer for Stan Hayes) holding my hand. It was the first thing
c I did as a graduate student.
c 
c Billy

C-------------------------------------------------------------------------------
C	EOFSUB_2.FOR MODIFIES TRIDIN TO PASS OVERFLOW. SEE NEAR END OF TRIDIN.
C	ALSO MODIFIES QRSYM TO NOTE PROGRESS TO 6.
C-------------------------------------------------------------------------------

C.................EOFIN.FOR is a front end to the EOF subroutines which follow.
C.................Arrangement of output:
C					  VEC(position,EOF#)
C					  TAF(EOF#,time)

c	Variables:
c	data(nx,nt)	I/O	original data		Returned demeaned
c	nx		I	number of spatial locations
c	nt		I	number of time realizations
c	val(nx)		O	eigenvalues (Lambda)
c	vec(nx,nx)	O	eigenvectors (Lambda*U). Same units as data.
c	taf(nx,nt)	O	time amplitude functions (V). Dimensionless.
c	pct(nx)		O	% variance represented by each EOF.
		
	SUBROUTINE EOFIN(DATA,NX,NT,VAL,VEC,TAF,PCT,C,eofwork)

        IMPLICIT NONE

C  Calling argument declarations.

        INTEGER   NX, NT
	REAL      DATA(NX,*),PCT(*),VAL(*),VEC(NX,*),TAF(NX,*),C(NX,*)
	REAL eofwork(*)

C  Internal declarations

	INTEGER I, J, I1, I2, IE, IS
	REAL TVAR, RNT
C---------------------------------------------------------------------
C.................Find mean, demean DATA.
C.................Note PCT is used first as a dummy to save the mean.
	RNT = NT
	DO 100 I=1,NX
	PCT(I)=0.0
	DO J=1,NT
	   PCT(I)=PCT(I)+DATA(I,J)/RNT
	ENDDO
	DO J=1,NT
	   DATA(I,J)=DATA(I,J)-PCT(I)
	ENDDO
100	CONTINUE
C--------------------------------------------------------------------
C.................Form the mean product matrix.
	DO 210 I1=1,NX
	DO 211 I2=1,NX

	DO J=1,NT
	   C(I1,I2) = C(I1,I2) + DATA(I1,J) * DATA(I2,J)
	ENDDO
	C(I1,I2) = C(I1,I2)/ RNT

211	CONTINUE
210	CONTINUE
C------------------------------------------------------------------------
C.................Call the subroutines which do the work.

	CALL QRSYM(C,NX,VAL,VEC, eofwork)

C.................VEC(IS,IE) is arranged so that IS indexes the space
C.................dimension of each eigenvector and IE the EOF number.
C------------------------------------------------------------------------
C.................Find the percentage of total variance accounted for by
C.................each eigenvector.
	TVAR=0.0
	DO I=1,NX
	   TVAR=TVAR+VAL(I)
	ENDDO
	DO I=1,NX
	   PCT(I)=100.*ABS(VAL(I)/TVAR)
	ENDDO
C------------------------------------------------------------------------
C.................Renormalize EOFs.
C.................Thus EOFs have units of data, while time amplitude functions
C.................are dimensionless.
C.................Thus the sum of the squares of a given EOF = eigenvalue.
C.................And the mean of a given TAF = 1.
C-------------------------------------------------------------------------
C.................First compute the time amplitude function from the data.
C.................Renormalize each TAF by dividing by SQRT of its eigenvalue.

	DO 350 J=1,NT			! loop over time
	DO 360 IE=1,NX			! loop over EOF numbers

	TAF(IE,J)=0.0

	DO IS=1,NX
	   TAF(IE,J) = TAF(IE,J) + VEC(IS,IE)*DATA(IS,J)/SQRT(ABS(VAL(IE)))
	ENDDO

360	CONTINUE
350	CONTINUE

C.................Renormalize each EOF by the SQRT of its eigenvalue.

	DO IE=1,NX			! loop over EOF numbers
	   DO IS=1,NX
	      VEC(IS,IE) = VEC(IS,IE)*SQRT(ABS(VAL(IE)))
	   ENDDO
	ENDDO
C----------------------------------------------------------------------
	RETURN
	END
		

C************************************************************************


c -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

C  eofin_chel_gap from Billy Kessler.  
C  Ansley Manke 9/98  Change to have the bad-data flag "realbad" an argument
C                     and to test for .GT. realbad rather than .LT.
C                     Add argument pct_cutoff for where to stop the computations
C                     that normalize eigenvectors and compute time functions,
C                     and returning nout = number of functions to output.
C  Ansley Manke 8/99  Change dimension statements for arguments to be (*) or
C                     (NX,*), etc.  The work arrays in the calling functions
C                     now have extra space to accomodate I = 1,N... BB(I+1) in 
C                     QRSYM, for example.
C  Ansley Manke 10/00 Change bad-data flag "realbad"  to bad_flag_dat
C                     and a separate flag for bad_flag_result, the bad-data
C                     flag for the result.
C  Ansley Manke 1/01  declare all variables.
C
C  Ansley Manke 5/01  remove pct_cutoff as a parameter
C 
C  Ansley Manke 9/05  Make all workspace be allocated as external function work 
C                     arrays. Arrays in TRIDIN passed in rather than declared.
C  Ansley Manke 7/12  Changes for solving EOF's in double precision.
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c	eof_gap_sub4.f 
c	implements Chelton's '82 method for finding EOFs of gappy time series.
c	4 fixes a few imprecisions in earlier versions:
c		1) sets negative eigenvalues to eps = 1/10th smallest EV
c		2) blanks TAF if ALL data missing at a given time realization.
c	If no gaps, reduces to ordinary EOFs (but computationally wasteful).
c	Requires eigenvalue finder SUBROUTINEs in eofsub_2.for.
c	Gaps assumed to be marked by values = 1.e34.
c***************************************************************************

C..........SUBROUTINE EOFIN_CHEL_GAP 
c..........is a front end to the EOF SUBROUTINEs in eofsub_2.for.


c	Variables:
c	data(nx,nt)	I/O	original data.	Returned demeaned.
c	nx		I	number of spatial locations
c	nt		I	number of time realizations
c	val(nx)		O	eigenvalues (Lambda)
c	vec(nx,nx)	O	eigenvectors (Lambda*U). Same units as data.
c	taf(nx,nt)	O	time amplitude functions (V). Dimensionless.
c	pct(nx)		O	% variance represented by each EOF.
c	c(nx,nx)	O	work space for cov matrix (garbage output)

C..........Arrangement of output:
C					  VEC(position,EOF#)
C					  TAF(EOF#,time)
		
C---------------------------------------------------------------------
      SUBROUTINE EOFIN_CHEL_GAP (DATA,NX,NT,VAL,VEC,TAF,PCT,C, 
     +      eofwork, bad_flag_dat, bad_flag_result, err_msg, ier)

        IMPLICIT NONE

        INTEGER nx
	REAL DATA(NX,*), PCT(*), VAL(*), VEC(NX,*),
     +            TAF(NX,*), C(NX,*)

      CHARACTER*(*) err_msg

      INTEGER nt, ier, ix, i2, it, i, j, i1, ie, npr,
     .        ic, is
      REAL ct, eps, sumljgji, beta, tvar

      REAL eofwork(*)
      REAL bad_flag_dat, bad_flag_result

C---------------------------------------------------------------------
c.............initializations.

        ier = 0
	do 50 ix=1,nx
	val(ix)=0.
	pct(ix)=0.
	do i2=1,nx
	   vec(ix,i2)=0.
	   c(ix,i2)=0.
	ENDDO
	do it=1,nt
	   taf(ix,it)=0.
	ENDDO
50	continue
C---------------------------------------------------------------------
C.................Find mean, demean DATA.
C.................Note PCT is used first as a dummy to save the mean.
	DO 100 I=1,NX
	ct=0.
	DO 110 J=1,NT
	if (data(i,j).GT.bad_flag_dat) then
		PCT(I)=PCT(I)+DATA(I,J)
		ct=ct+1.
	endif
110	continue
	if (ct.gt.0.) then
		pct(i)=pct(i)/ct
	else
	   write(err_msg,*) ' EOFIN_CHEL_GAP found the time series',
     .                     ' at ix=',i,' with no values.'
           ier = 1
	   return
	endif
	DO 120 J=1,NT
	   if (data(i,j).GT.bad_flag_dat) then
		DATA(I,J)=DATA(I,J)-PCT(I)
	   endif
120	continue
100	CONTINUE
C--------------------------------------------------------------------
C.................Form the mean product matrix.
	DO I1=1,NX
	   DO I2=1,NX

	      ct=0.
	      DO J=1,NT
	        if (data(i1,j).GT.bad_flag_dat .and. 
     .              data(i2,j).GT.bad_flag_dat) then
		   ct=ct+1.
		   C(I1,I2) = C(I1,I2) + DATA(I1,J) * DATA(I2,J)
	         endif
	      ENDDO

	      if (ct.gt.0.) then
		c(i1,i2)=c(i1,i2)/ct
	      else
                 write(err_msg,*) ' EOFIN_CHEL_GAP found no overlapping',
     .          ' values in two time series at EOF-1D-coordinates ', 
     .          i1,i2, ' Will be a zero in covariance matrix'
                 ier = 1
                 return
	      endif
	   ENDDO
	ENDDO
C------------------------------------------------------------------------
C.................Call the subroutines which do the work.

	CALL QRSYM(C,NX,VAL,VEC, eofwork) 

C.................VEC(IS,IE) is arranged so that IS indexes the space
C.................dimension of each eigenvector and IE the EOF number.
C------------------------------------------------------------------------
c...................check for negative eigenvalues, set to eps.
	do ie=1,nx
	   if (val(ie).lt.0.) GOTO 251
	ENDDO
	GOTO 252

251	continue

	eps=1.e20
	do ie=1,nx
	   if (abs(val(ie)).lt.eps) eps=abs(val(ie))
	enddo
	eps=eps/10.
	do ie=1,nx
	   if (val(ie).lt.0.) val(ie)=eps
	enddo

252	continue

C----------------------------------------------------------------------
C !ACM  Array pctis used as a dummy array below, and we will compute it 
C       again before returning.
C.................Find the percentage of total variance accounted for by
C.................each eigenvector.
c.................Place this at the end to be able to use pct as dummy
c.................for counting missing data in TAF calculation.
	TVAR=0.0
	DO I=1,NX
	   TVAR=TVAR+VAL(I)
	ENDDO
	DO I=1,NX
	   PCT(I)=100.*ABS(VAL(I)/TVAR)
	ENDDO

C------------------------------------------------------------------------
C.................Renormalize EOFs.
C.................Thus EOFs have units of data, while time amplitude functions
C.................are dimensionless.
C.................Thus the sum of the values of a given EOF = SQRT(eigenvalue).
C.................And the mean of a given TAF = 1.
C-------------------------------------------------------------------------
C.................First compute the time amplitude function from the data.
C.................Renormalize each TAF by dividing by SQRT of its eigenvalue.

c...........................loop over time

	DO 350 J=1,NT
	npr = nt/ 10 
        if (nt .gt. 100) npr = (npr/ 10) * 10     

c............find out if any gaps exist at this time.
c............use ic to count them and pct(nx) to keep track of where they are.
	ic=0
	do is=1,nx
	   if (data(is,j).GT.bad_flag_dat) then
		ic=ic+1
		pct(is)=0.
	   else
		pct(is)=1.
	   endif
	enddo

c..............if no blanks at this time, compute TAF as usual.
	if (ic.eq.nx) then
c................................loop over EOF numbers, then over space.
		DO IE=1,NX

		   DO IS=1,NX
		      TAF(IE,J) = TAF(IE,J) + 
     .			    VEC(IS,IE)*DATA(IS,J)
		   ENDDO
		ENDDO
	elseif (ic.eq.0) then
c..............If there are NO data values at time j, then blank the TAF.
		do ie=1,nx
		taf(ie,j)=bad_flag_result
		enddo
	else
c.............If there are some blanks, then use Chelton's estimate Beta_i(t).
c.............Now use c(nx,nx) as dummy to keep Chelton's Gamma_ji.
c.............Fill c(i,j), where i and j are both EOF #s. 
c.............Sum over MISSING data points only.

		do i1=1,NX
		do i2=1,nx
		   c(i1,i2)=0.
		   do ix=1,nx
		      if (pct(ix).eq.1)
     .			  c(i1,i2)=c(i1,i2)+vec(ix,i1)*vec(ix,i2)
		   enddo
		enddo
		enddo

c..................Loop over EOF numbers.
		do ie=1,NX

c......................Find Beta_i(t) from Gamma_ij.
c......................Also need the sum of Lambda_j*Gamma_ji**2 over all j.ne.i

c.......................loop 430 sums over space.
		   sumljgji=0.
                   do i2=1,nx
                      if (i2.ne.ie) sumljgji=sumljgji+val(i2)*c(i2,ie)**2
                   enddo

c.....................find beta.
		   beta = ( (1.-c(ie,ie)) * val(ie) ) / 
     .			 ( val(ie) * (1.-c(ie,ie))**2 + sumljgji )

c...................Now find TAF. Summation in space over existing points only.
		   do is=1,nx
			if (pct(is).eq.0.) taf(ie,j) = taf(ie,j) + 
     .			   	beta * vec(is,ie) * data(is,j)
		   enddo

		enddo


	endif
		
350	CONTINUE

C-------------------------------------------------------------------------
C.................Renormalize each EOF by the SQRT of its eigenvalue.
c.................loop over EOF numbers, then over space.

	DO IE=1,NX

	   DO j=1,nt
	      if (taf(ie,j) .GT. bad_flag_dat) taf(ie,j)=taf(ie,j)/sqrt(val(ie))
	   ENDDO
	   DO IS=1,NX
	      VEC(IS,IE) = VEC(IS,IE)*SQRT(VAL(IE))
	   ENDDO

	ENDDO
C----------------------------------------------------------------------
C.................Find the percentage of total variance accounted for by
C.................each eigenvector.
c.................Place this at the end to be able to use pct as dummy
c.................for counting missing data in TAF calculation.
	TVAR=0.0
	DO I=1,NX
	   TVAR=TVAR+VAL(I)
	ENDDO
	DO I=1,NX
	   PCT(I)=100.*ABS(VAL(I)/TVAR)
	ENDDO
C------------------------------------------------------------------------
	RETURN
	END

C************************************************************************
      SUBROUTINE QRSYM (A,N,E,X, eofwork)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

      INTEGER N
      REAL A(N,*),E(*),X(N,*)

* No need to declare Z,T,E,F dimensions since they are not used
* and Fortran arrays are column major
      REAL eofwork(wrk9lox:wrk9hix, wrk9loy:wrk9hiy)

*  
* 5/99 ACM  Use a work array eofwork rather than explicitly dimensioned
*           arrays ALPHA, BETA, BB, and P (originally declared in HOUSEH)
c      DIMENSION ALPHA(4000),BETA(4000),BB(4000)

      INTEGER nb
      REAL rnorm, eps
      REAL macheps

      nb=n

* Set macheps once and pass into QRSTD where its used as ETA
* and into TRIDIN.
* Assign 2**(-48) to a variable so it is the correct size REAL
* (should this be different under double-precision?)
      macheps = 2.**(-48)

      CALL HOUSEH(A,N,eofwork(1,1),eofwork(1,2),NB,
     .            eofwork(1,4))

      CALL QRSTD (eofwork(1,1),eofwork(1,2),N,E,
     .            eofwork(1,3),RNORM,MACHEPS, EPS)

      CALL TRIDIN(eofwork(1,1),eofwork(1,2),N,E,RNORM,
     .            N,macheps,X,NB,eofwork(1,5),eofwork(1,6),
     .            eofwork(1,7),eofwork(1,8),eofwork(1,9),
     .            eofwork(1,10) )

      CALL BACKS(eofwork(1,2),A,N,X,N,EPS,NB)

      RETURN
      END


C**********************************************************************

      SUBROUTINE HOUSEH (G,N,A,B,NB, p)

c  7/2002 ACM  Make the sums etc for calculating the matrix inverse
c              double precision.

      INTEGER NB
      REAL G(NB,*),A(*),B(*)

      REAL P(*)

      INTEGER n, n2, k, k1, i, j, i1
      REAL*8 sum, sigma, absb, alpha, beta, gamma, t

      N2=N-2
      IF(N2 .EQ. 0) THEN 
         A(N-1)=G(N-1,N-1)
         B(N-1)=G(N,N-1)
	 GOTO 45
      ELSE IF (N2 .LT. 0) THEN
         GOTO 45
      ENDIF

      DO K=1,N2
         A(K)=G(K,K)
         SUM=0
         K1=K+1
         DO I=K1,N
            SUM=SUM+G(I,K)**2
         ENDDO

         SIGMA=SUM
         ABSB=SQRT(SIGMA)
         ALPHA=G(K+1,K)
         BETA=-ABSB
         IF(ALPHA .LT. 0.) BETA=ABSB
         BETA=ABSB
         B(K)=BETA
         IF(SIGMA .EQ. 0.) GOTO 43 
         GAMMA=1/(SIGMA-ALPHA*BETA)
         G(K+1,K)=ALPHA-BETA		! change C so now not Mean Prod Matrix

         DO I=K1,N
            SUM=0
            DO J=K1,I
              SUM=SUM+G(I,J)*G(J,K)
            ENDDO
	    IF(I .NE. N) THEN
               I1=I+1
               DO J=I1,N
                  SUM=SUM+G(J,I)*G(J,K)
	       ENDDO
	    ENDIF
            P(I)=GAMMA*SUM
         ENDDO
         SUM=0
         DO I=K1,N
            SUM=SUM+G(I,K)*P(I)
	 ENDDO
         T=0.5*GAMMA*SUM

         DO I=K1,N
            P(I)=P(I)-T*G(I,K)
	 ENDDO

         DO I=K1,N
            DO J=K1,I
               G(I,J)=G(I,J)-G(I,K)*P(J)-P(I)*G(J,K)
            ENDDO
	 ENDDO
43       CONTINUE
      ENDDO
     
      A(N-1)=G(N-1,N-1)
      B(N-1)=G(N,N-1)
      
   45 CONTINUE
      A(N)=G(N,N)
      B(N)=0.0

      RETURN
      END


C************************************************************************

      SUBROUTINE QRSTD (ALPHA,BETA,N,E,BB,RNORM,ETA,EPS)
      REAL ALPHA(*),BETA(*),E(*),BB(*)

      INTEGER n, i, k, m, n1, i1, j
      REAL eta, delta, t, r, w, c, s, shift, g, p, ek1, rnorm, eps


C W.KAHAN AND J.VARAH, TWO WORKING ALGORITHMS FOR THE EIGENVALUES OF A
C SYMMETRIC TRIDIAGONAL MATRIX. TECHNICAL REPORT NO. CS43, AUGUST 11,
C 1966. COMP.SC.DEPT. STANFORD UNIVERSITY.

      DO I=1,N
         E(I)=ALPHA(I)
         BB(I+1)=BETA(I)**2
      ENDDO
      BB(1)=0.0
      BB(N+1)=0.0
C INFINITY NORM OF TRIDIAGONAL MATRIX
      RNORM=0.0
      DO I=1,N
         RNORM=AMAX1(RNORM,SQRT(BB(I))+ABS(E(I))+SQRT(BB(I+1)))
      ENDDO
C ETA = RELATIVE MACHINE PRECISION
      DELTA=ETA*RNORM
      EPS=DELTA**2
      IF (EPS.EQ.0) RETURN
      K=N
    6 M=K
      IF(M.LE.0)GO TO 56
    8 K=K-1
      IF(BB(K+1).GE.EPS)GO TO 8
C NEXT
      IF(K.NE.M-1)GO TO 13
      BB(K+1)=0.0
      GO TO 6
C TWOBY2
   13 T=E(M)-E(M-1)
      R=BB(M)
      IF(K.GE.M-2)GO TO 22
      W=BB(M-1)
      C=T**2
      S=R/(C+W)
      IF(S*(W+S*C).GE.EPS)GO TO 22
      M=M-1
      BB(M+1)=0.0
      GO TO 13
C END NEGLIGIBLE BB
   22 IF(ABS(T).GE.DELTA)GO TO 25
      S=SQRT(R)
      GO TO 28
   25 W=2.0/T
      S=W*R/(SQRT(W**2*R+1.0)+1.0)
   28 IF(K.NE.M-2)GO TO 33
      E(M)=E(M)+S
      E(M-1)=E(M-1)-S
      BB(K+1)=0.0
      GO TO 6
C DO A QR STEP ON ROWS AND COLUMNS K+1 THROUGH M
   33 SHIFT=E(M)+S
      IF(ABS(T).GE.DELTA)GO TO 37
      W=E(M-1)-S
      IF(ABS(W).LT.ABS(SHIFT))SHIFT=W
   37 S=0.0
      G=E(K+1)-SHIFT
      C=1.0
      GO TO 45
C LOOP
   40 C=P/T
      S=W/T
      W=G
      EK1=E(K+1)
      G=C*(EK1-SHIFT)-S*W
      E(K)=(W-G)+EK1
C ENTRY
   45 IF(ABS(G).GE.DELTA)GO TO 48
      IF(G.GE.0.0)GO TO 47
      G=G-C*DELTA
      GO TO 48
   47 G=G+C*DELTA
   48 P=G**2/C
      K=K+1
      W=BB(K+1)
      T=W+P
      BB(K)=S*T
   50 IF(K.LT.M)GO TO 40
      E(K)=G+SHIFT
      GO TO 6
C SORT
   56 IF(N.EQ.1)RETURN
      N1=N-1
      DO 70 I=1,N1
      K=I
      T=E(I)
      I1=I+1
      DO 62 J=I1,N
      IF(E(J).LE.T)GO TO 62
      T=E(J)
      K=J
   62 CONTINUE
      IF(I.EQ.K)GO TO 70
      E(K)=E(I)
      E(I)=T
   70 CONTINUE
      RETURN
      END

C*************************************************************************

      SUBROUTINE TRIDIN (C,B,N,W,NORM,M1,MACHEPS,Z,NB,M,P,Q,R,INT,X)

      INTEGER nb, nminus1, ii
      REAL C(*),B(*),W(*),Z(NB,*),
     .     M(*),P(*),Q(*),R(*),INT(*),X(*)

C J.H.WILKINSON, CALCULATION OF THE EIGENVECTORS OF A SYMMETRIC
C TRIDIAGONAL MATRIX BY INVERSE ITERATION. NUMERISCHE MATHEMATIK 4,
C 368-376 (1962)

* 9/05 ACM  Use a work array eofwork rather than explicitly dimensioned
*           arrays M, P, Q, R, INT, and X

      INTEGER N,M1
      REAL NORM,MACHEPS

      INTEGER I,J
      REAL BI,BI1,LAMBDA,U,V,H,EPS,ETA,BIG,BIGGER


* In single-precision these were 1.e10 and 1.e20
* Determined by trial and error.
	
      big = 1.E20
      bigger = 1.E30

      IF (N .LT. 1) RETURN
      IF (N .EQ. 0)  Z(1,1)=1.

      LAMBDA=NORM
      EPS=MACHEPS*NORM
      DO 90 J=1,M1
      LAMBDA=LAMBDA-EPS
      IF(W(J).LT.LAMBDA)LAMBDA=W(J)
      U=C(1)-LAMBDA
      V=B(1)
      IF(V.EQ.0)V=EPS
      NMINUS1=N-1
      DO 60 I=1,NMINUS1
      BI=B(I)
      IF(BI.EQ.0)BI=EPS
      BI1=B(I+1)
      IF(BI1.EQ.0)BI1=EPS
      IF(ABS(BI).LT.ABS(U))GO TO 50
      M(I+1)=U/BI
      IF((M(I+1).EQ.0).AND.(BI.LE.EPS))M(I+1)=1
      P(I)=BI
      Q(I)=C(I+1)-LAMBDA
      R(I)=BI1
      U=V-M(I+1)*Q(I)
      V=-M(I+1)*R(I)
      INT(I+1)=+1
      GO TO 60
   50 M(I+1)=BI/U
      P(I)=U
      Q(I)=V
       R(I)=0
      U=C(I+1)-LAMBDA-M(I+1)*V
      V=BI1
      INT(I+1)=-1
   60 CONTINUE
      P(N)=U
      Q(N)=0
      R(N)=0
      X(N+1)=0
      X(N+2)=0
      H=0
      ETA=1.0/N
      DO 67 II=1,N
      I=N-II+1
      U=ETA-Q(I)*X(I+1)-R(I)*X(I+2)
      IF(P(I).NE.0)GO TO 65
      X(I)=U/EPS
      GO TO 66
   65 X(I)=U/P(I)
   66 H=H+ABS(X(I))
   67 CONTINUE

      DO I=1,N
         X(I)=X(I)/H
      ENDDO

      DO 75 I=2,N
      IF(INT(I).LE.0)GO TO 70
      U=X(I-1)
      X(I-1)=X(I)
      X(I)=U-M(I)*X(I-1)
      GO TO 75
   70 X(I)=X(I)-M(I)*X(I-1)
   75 CONTINUE

      H=0
      DO 82 II=1,N
      I=N-II+1
      U=X(I)-Q(I)*X(I+1)-R(I)*X(I+2)
      IF(P(I).NE.0)GO TO 80
      X(I)=U/EPS
      GO TO 81
   80 X(I)=U/P(I)

 81   IF (X(I).GT.BIG) THEN
        H=BIGGER
      ELSE
        H=H+X(I)**2
      ENDIF
   82 CONTINUE
      H=SQRT(H)
C STORE VECTORS AS COLUMNS
      DO I=1,N
         Z(I,J)=X(I)/H
      ENDDO

   90 CONTINUE
      RETURN
      END

C**********************************************************************

      SUBROUTINE BACKS (BETA,A,N,X,M1,EPS,NB)
      INTEGER n, nb, m1 
      REAL BETA(*), A(NB,*), X(NB,*), EPS

      INTEGER j, kk, k, k1, i
      REAL s


C BACKTRANSFORMATION

      IF (N.LE.2)RETURN

      DO 90 J=1,M1
      DO 85 KK=3,N
      K=N-KK+1
      IF (ABS(BETA(K)).LE.EPS) GO TO 85
      S=0.0
      K1=K+1

      DO I=K1,N
         S=S+A(I,K)*X(I,J)
      ENDDO

      S=S/(BETA(K)*A(K1,K))
      DO I=K1,N
         X(I,J)=X(I,J)+S*A(I,K)
      ENDDO

   85 CONTINUE
   90 CONTINUE

      RETURN
      END

************************************************************************

      SUBROUTINE count_neof (arg_1, neof, ok, nx, ny, nt, k1, m1, n1,
     .                       arg_lo_ss, arg_hi_ss, arg_incr, mx, 
     .                       flag, frac_timeser, err_msg, ier)

*  Find number of time series with valid data.  This will be neof.
*  Set array OK to mark which (x,y) have some good data.

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

* arguments
      INTEGER neof, nx, ny, nt, k1, m1, n1, mx, ier
      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, mem1loz:mem1hiz, 
     .           mem1lot:mem1hit, mem1loe:mem1hie, mem1lof:mem1hif)
      REAL ok(nx,ny)
      INTEGER arg_lo_ss(6,EF_MAX_ARGS),
     .        arg_hi_ss(6,EF_MAX_ARGS),
     .        arg_incr (6,EF_MAX_ARGS)
      REAL flag, frac_timeser
      CHARACTER*(*) err_msg

* local variables
      INTEGER i, j, i1, j1, l1
      REAL a_nt

*  Set array ok = 1 if frac_timeser fraction of valid data exists at the point.

      a_nt = float(nt)

      neof = 0
      j1 = arg_lo_ss(Y_AXIS,ARG1)
      DO 30 j = 1, ny
         i1 = arg_lo_ss(X_AXIS,ARG1)
         DO 20 i = 1, nx

            ok(i,j) = 0.
            DO 10 l1 = arg_lo_ss(T_AXIS,ARG1), arg_hi_ss(T_AXIS,ARG1)
               IF (arg_1(i1,j1,k1,l1,m1,n1) .ne. flag) then
                  ok(i,j) = ok(i,j) + 1.
               endif
  10        CONTINUE
            ok(i,j) = ok(i,j) / a_nt
            IF (ok(i,j) .GE. frac_timeser) neof = neof + 1 

            i1 = i1 + arg_incr(X_AXIS,ARG1) 
  20     CONTINUE
         j1 = j1 + arg_incr(Y_AXIS,ARG1) 
  30  CONTINUE

      ier = 0
      IF (neof .gt. mx) then
        WRITE(err_msg,*) 'Increase parameter mx in eof.F ',
     .                   'Set mx at least', neof
        ier = 1
        RETURN
      ENDIF

      RETURN
      END

************************************************************************

      SUBROUTINE pack_ef (arg_1, ddat_1d, isave_jsave, neof,  
     .                    ok, frac_timeser, nx, ny, nt, k1, m1, n1,
     .                    arg_lo_ss, arg_hi_ss, arg_incr)

*  put arg_1(x,y,t) into one list ddat_1d(neof,nt)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

      INTEGER neof, nx, ny, nt, k1, m1, n1
      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, mem1loz:mem1hiz, 
     .           mem1lot:mem1hit, mem1loe:mem1hie, mem1lof:mem1hif)
      REAL ddat_1d(neof,nt)
      REAL isave_jsave(wrk7lox:wrk7hix, wrk7loy:wrk7hiy,
     .                 wrk7loz:wrk7hiz, wrk7lot:wrk7hit,
     .                 wrk7loe:wrk7hie, wrk7lof:wrk7hif)
      REAL ok(nx,ny)
      REAL frac_timeser
      INTEGER arg_lo_ss(6,EF_MAX_ARGS),
     .        arg_hi_ss(6,EF_MAX_ARGS),
     .        arg_incr (6,EF_MAX_ARGS)

      INTEGER i, j, l, i1, j1, l1
      INTEGER ipack

      ipack = 0

      j1 = arg_lo_ss(Y_AXIS,ARG1)
      DO 130 j = 1, ny
         i1 = arg_lo_ss(X_AXIS,ARG1)
         DO 120 i = 1, nx
            IF (ok(i,j) .GE. frac_timeser) THEN
               ipack = ipack + 1
 
               isave_jsave(ipack,1,1,1,1,1) = i
               isave_jsave(ipack,2,1,1,1,1) = j
 
               l1 = arg_lo_ss(T_AXIS,ARG1)
               DO 110 l = 1, nt
                  ddat_1d(ipack, l) = arg_1(i1,j1,k1,l1,m1,n1)
                  l1 = l1 + arg_incr(T_AXIS,ARG1) 
 110           CONTINUE

             ENDIF
             i1 = i1 + arg_incr(X_AXIS,ARG1) 
 120     CONTINUE
         j1 = j1 + arg_incr(Y_AXIS,ARG1) 
 130  CONTINUE

      RETURN
      END
