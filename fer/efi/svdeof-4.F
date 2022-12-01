c..........svdeof-4.f: Do EOFs with either NX > NT or NT > NX

c   User description:
c..........Subroutine finds "traditionally-scaled" EOFs using SVD routine.
c.........."Traditionally-scaled" means that the units of the EOFs are the
c..........  units of the variable, while the TAFs are dimensionless.
c..........Routine is based on Numerical Recipes routine svdcmp.f.
c..........Result of EOF decomposition is that a function f(x,t) of space and  
c..........  time is decomposed into a sum of products: f(x,t) = SUM[v(x)*a(t)].
c..........The v(x) are referred to as the EOFs, or eigenvectors, while the
c..........  a(t) are the "Time Amplitude Functions" (TAFs) (Also called
c..........  "Principal Components") 
c..........The number of terms in the sum (number of non-zero EOFs)  
c..........  is the minimum of (# of locations,# of times). 

c..........Much CPU is saved by choosing to do the EOFs according to which of
c..........NX,NT is larger, since the number of nonzero eigenvalues is min(NX,NT).
c..........The routine can be called with any (NX,NT) (makes the choice automatically)

c---------------------------------------------------------------------------
c.....Use: call the front-end routine DO_SVD_EOF with the following arguments:

c    A(NX,NT)	I/O	data set to be decomposed => returned as TAFs(neof,time)
c    NX			I	# of spatial locations in data set A
c    NT			I	# of time realizations in data set A
c    W(NX)	O	eigenvalues of decomposition
c    V(NX,NX)	O	EOFs in columns (V(space,neof))
c    PCT(NX)	O	% variance represented by each EOF

c   Notes:
c..........In the "standard" form, NT.ge.NX (more times than locations),
c..........there are NX non-zero EOFs and the output arrays are full.
c..........In the reversed form, NT.lt.NX (more locations than times),
c..........there are only NT non-zero EOFs (remaining values are zero).

c...>>>>....Data input matrix A is destroyed on output (overwritten with TAFs).
c...>>>>....Writes error messages to unit 6.
c...>>>>....There are 2 hardwired values in these routines:
c			IWRK=200000, declared in DO_SVD_EOF for use in sub MATRIXTRANS
c				IWRK must be at least (NX+NT)/2 (array MOVE(IWRK)).
c			NMAX=100000, declared in SVDEOF (workspace vectors RV1 and INDX)
c				NMAX is the maximum size of the smaller of (NX,NT)

c...........Included subroutines, called by DO_SVD_EOF (no external routines required)
c			FIX_V (rearranges array V in the case of reversed EOFs) (comments at fix_v)
c			SCALING (does "traditional scaling")
c			DEMEAN (demeans input data)
c			SVDEOF (modified Numerical Recipes routine SVDCMP)
c			INDEXX (NR routine to sort eigenvalues to size order)
c			MATRIXTRANS (matrix transpose routine from U.Kent, UK)
c			SWITCH_A_V (switches (back) the roles of A and V if reversed EOFS)

c   History:
c...........Based on Numerical Recipes subroutine svdcmp.
c...........Aug 1992: svdeof-1.f: svdcmp2 modified to transpose input array A 
c...........  to usual form: A(N,M), where N is space and M is time. M.ge.N.
c...........May 1998: Sorting EOFs by size of eigenvalue (svdeof-2.f).
c...........Jun 1006: svdeof-4.f handles either NX>NT or NT>NX
c...........Everything beyond svdcmp programmed by Billy Kessler (NOAA/PMEL)

c ......... 7/2012 ACM Implement as Ferret external function. Workspace is allocated
c ......... and sent in to the function. A status flag is returned.

C  11/2022 *acm* Code cleanup: In the calling routines allocate the integer array 
C                work3 instead of using a (real*8) external-functions work array.

c*********************************************************************

c.......Front-end subroutine to choose which way to do the EOFs, based on size of M and N.
c.......DO_SVD_EOF is called from the main program, and calls SVDEOF to do the work.

c..........It is assumed that the input data is A(space,time) (NX,NT).

	SUBROUTINE DO_SVD_EOF (A, NX, NT, w, v, pct, move, work2, work3, ier)
        IMPLICIT REAL(A-H, O-Z), INTEGER(I-N)
c...................................maximum size of (m+n)/2.
cc	PARAMETER (iwrk=200000)		! HARDWIRING
	INTEGER ier

c...........it is ok to use NX for dim(W,V,PCT) because that is really NEOF
c...........which is the minimum of (NX,NT)
	real A(NX,NT),W(NX),V(NX,NX),pct(NX), work2(*)
	integer work3(*)

c......................................internal arrays.
	integer move(*)
	INTEGER iwrk
	iwrk = (nx+nt)/2

c-----------------------------------------------------------------------
c........check that enough space is provided in hardwired array (iwrk)
cc	if ((nx+nt)/2.gt.iwrk) then
cc	   	write(6,*) '(nx+nt)/2.gt.',iwrk,' in routine DO_SVD_EOF. Error. Returning.'
cc	   	return
cc	endif
	
c.................demean data array A. (Do first to handle either transpose)
c.................now do by subroutine to allow A to be A(NX*NT).
	call demean (a, nx, nt)
	
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
c........IF-block to work with either usual or reversed EOFs

c........find out whether NX or NT is smaller
	if (nt.ge.nx) then		! more times than locations = usual EOFs

C...............do the work
		CALL SVDEOF(a, nt, nx, w, v, pct, work2, work3, ier)
		IF (ier .NE. 0) RETURN

c...........NEOF=NX nonzero EOFs 
c...........EOFs in V(NX,NEOF)   NX = NEOF
c...........TAFs in A(NEOF,NT)   NT > NEOF

c...........do the "traditional scaling"
		call scaling(a, nx, nt, w, v, pct, nx)
		
	else					! more locations than times. Must reverse order

c..........transpose the input data array A(NX,NT)
c..........after transpose, A is used as A(NT,NX)=(time,space)
c..........(but this can only be done in a sub call where A is just an address)
		call MATRIXTRANS(a,nx,nt,nt*nx,move,iwrk,iok)	

c...............now call the SVD calculation with transposed data array A
		CALL SVDEOF(a, nx, nt, w, v, pct, work2, work3, ier)
		IF (ier .NE. 0) RETURN
		
c......There are NEOF=NT nonzero EOFs (note NT < NX). 

c......The EOFs are in A(NEOF=NT,NX) which is filled. 
c......It must be transposed to be in the same order as the usual case
c......(but note it is not square as are the usual EOFs stored in V(NX,NX))
		call MATRIXTRANS(a,nt,nx,nt*nx,move,iwrk,iok)

c......The output of SVDEOF makes use of W(NT), PCT(NT) and V(NT,NT).
c......W and PCT are OK as is, since they are 1-d vectors, but the
c......needed elements of V are stored as a sequence of length NT*NT, 
c......(NT TAFs in sequence, followed by zeroes), stored in an actual 
c......array V(NX,NX) (recall NX > NT).
c......Fix V in a subroutine: 
c......Place each TAF running down sequential columns of V(NX,NX).
c......When this is done, the TAFs take up an NT by NT square in the 
c......upper left corner of the larger array V(NX,NX).
		call fix_v(v,nt,nx)
				
c......Transpose V to get the usual order, but note the zeroes. 
c......Must give the complete size of V to the transpose routine.
		call MATRIXTRANS(v,nx,nx,nx*nx,move,iwrk,iok)

c......Note reversal of position of A and V in this call.
		call scaling(v, nx, nt, w, a, pct, nt)
		
c......Switch the EOFs and TAFs to V and A, respectively.
c......Also clean up by zeroing out the unused parts (also W and PCT).
		call switch_a_v(a,v,w,pct,nx,nt)
				
	endif
	
	return
	end
	
c*********************************************************************
c.....subroutine to switch (back) the roles of A and V in the case of reversed EOFS.
c.....The result is that the EOFs are in the columns of V, and the TAFs in the rows of A.
c.....This works because NT < NX. The original array V(NX,NX) only contains the
c.....TAFS in a square (NT,NT) in its upper left corner, so there is at least one
c.....free column at the right side of V, for work space.

c.....Also clean up by zeroing out the unused parts (including W and PCT)

c.....Method:
c	1) move the last (n=NEOF=NT) column of A to the last column (NX) of V
c	2) move the final element of each TAF from V(n,NT) to A(n,NT)
c	3) repeat (1) and (2) with the second-last column and second-last elements
c	   continue the above operations NT times
c	4) shift the rows of V (EOFs) over to the left side of V

	subroutine switch_a_v(a,v,w,pct,nx,nt)
	IMPLICIT REAL(A-H, O-Z), INTEGER(I-N)

	real a(nx,nt), v(nx,nx)
	real w(nx), pct(nx)
	
c......move the elements to the other array
	do n=1,nt	! column number (nt-n+1 or nx-n+1)
	  do i=1,nx	! row number
	    v(i,nx-n+1)=a(i,nt-n+1)
	  enddo
	  do l=1,nt   ! row number in both (shift to same position in other array)
 	    a(nt-l+1,nt-n+1)=v(nt-l+1,nt-n+1)
	  enddo
	enddo

c.....shift the EOFs to the first columns of V
c.....the first one is in column (NX-NT+1)
	do i=1,nx
	  do n=1,nt
	    v(i,n)=v(i,n+nx-nt)
	  enddo
	enddo

c.....clean up by zeroing out the bottom of A and right side of V
	do i=1,nx
	  do n=nt+1,nx
	    v(i,n)=0.
	  enddo
	enddo

	do n=nt+1,nx
	  do l=1,nt
	    a(n,l)=0.
	  enddo
	enddo

c.......while we're here, clean up W and PCT
	do l=nt+1,nx
	  w(l)=0.
	  pct(l)=0.
	enddo

	return
	end
c*********************************************************************
c......output of SVDEOF has W(NT), PCT(NT) and V(NT,NT). 
c......W and PCT are OK as is, since they are 1-d vectors, 
c......but V is really a vector of length NT*NT, (TAFs in sequence, then zeroes),
c......stored in an actual array V(NX,NX) (note NX > NT)
c......Fix V here: Make first NT indices of each of the first NT columns a TAF
c......This might leave some stray leftovers, but it should be ok (?)

	subroutine fix_v(v,nt,nx)
	IMPLICIT REAL(A-H, O-Z), INTEGER(I-N)
	
	real v(nx*nx)		! true size of V
	
	do i=2,nt	! leave the first NT of em alone
	  do j=1,nt
	    ij=(i-1)*nt+j	! position of TAF(neof i,time j)
	    ij2=(i-1)*nx+j	! position where each column is NX long
	    v(ij2)=v(ij)
	  enddo
	enddo

	return
	end
c*********************************************************************
c..........."Traditional scaling"
c...........Section add by BK. 24 Aug 1992.
c...........Scale as my usual EOFs.
c...........Changes made here: 
c            1) Eigenvalues (vector W) output as W**2/M.
c			 2) EOFs (array V) output as V*W/sqrt(J).
c			 3) TAFs (array A) output as A*sqrt(J)
c.................Also find percent variance.

c.........23 Jun 2006: add an argument for the number of EOFs found.
c.........Need this to deal with reversed EOFs (which have only NT < NX EOFs),
c.........But still want to call with the same declared array sizes.
c.........Otherwise have array overflow problems (in V).

	subroutine scaling (a, n, m, w, v, pct, neof)
	IMPLICIT REAL(A-H, O-Z), INTEGER(I-N)
	
c	real A(n,m),W(n),V(n,n),pct(n)
	real A(n,m),W(neof),V(n,neof),pct(neof)

	rim=real(m)
	sqm=sqrt(rim)

	do ix=1,n
	  do ieof=1,neof
	    v(ix,ieof)=v(ix,ieof)*w(ieof)/sqm
	  enddo
	enddo

	totvar=0.
	do ieof=1,neof
	   w(ieof)=w(ieof)**2/rim
	   totvar=totvar+w(ieof)
	enddo

	do ieof=1,neof
	  pct(ieof)=w(ieof)/totvar
	enddo

	do ieof=1,neof
	  do it=1,m
	    a(ieof,it)=a(ieof,it)*sqm
	  enddo
	enddo

	return
	end

c*********************************************************************
c.......demean A in time. Returned demeaned.

	subroutine demean (a, nx, nt)
	IMPLICIT REAL(A-H, O-Z), INTEGER(I-N)
	
	real a(nx,nt)
	
	do i=1,nx
	  av=0.
	  do j=1,nt
	    av=av+a(i,j)/real(nt)
	  enddo
	  do j=1,nt
	    a(i,j)=a(i,j)-av
	  enddo
	enddo

	return
	end

c*********************************************************************
c........modified Numerical Recipes subroutine SVDCMP
c........subroutine now does no scaling (handles either transposition)
c........=> demeaning must have been done previously

      SUBROUTINE SVDEOF (A, M, N, W, V, pct, RV1, indx, ier)
	IMPLICIT REAL(A-H, O-Z), INTEGER(I-N)

c......................................maximum number of spatial locations.
cc      PARAMETER (NMAX=100000)		! HARDWIRED
c......................................arrays passed from main program.
      real A(N,M),W(N),V(N,N),pct(n),RV1(*)
      integer indx(*)
      integer ier

	ier = 0  ! success

      G=0.0
      SCALE=0.0
      ANORM=0.0
      DO I=1,N
        L=I+1
        RV1(I)=SCALE*G
        G=0.0
        S=0.0
        SCALE=0.0
        IF (I.LE.M) THEN
          DO K=I,M
            SCALE=SCALE+ABS(A(i,k))
          ENDDO
          IF (SCALE.NE.0.0) THEN
            DO K=I,M
              A(i,k)=A(i,k)/SCALE
              S=S+A(i,k)*A(i,k)
          ENDDO
            F=A(I,I)
            G=-SIGN(SQRT(S),F)
            H=F*G-S
            A(I,I)=F-G
            IF (I.NE.N) THEN
              DO J=L,N
                S=0.0
                DO K=I,M
                  S=S+A(i,k)*A(j,k)
                ENDDO
                F=S/H
                DO K=I,M
                  A(j,k)=A(j,k)+F*A(i,k)
                ENDDO
              ENDDO
            ENDIF
            DO K= I,M
              A(i,k)=SCALE*A(i,k)
            ENDDO
          ENDIF
        ENDIF
        W(I)=SCALE *G
        G=0.0
        S=0.0
        SCALE=0.0
        IF ((I.LE.M).AND.(I.NE.N)) THEN
          DO K=L,N
            SCALE=SCALE+ABS(A(k,i))
	  ENDDO
          IF (SCALE.NE.0.0) THEN
            DO K=L,N
              A(k,i)=A(k,i)/SCALE
              S=S+A(k,i)*A(k,i)
	    ENDDO
            F=A(l,i)
            G=-SIGN(SQRT(S),F)
            H=F*G-S
            A(l,i)=F-G
            DO K=L,N
              RV1(K)=A(k,i)/H
	    ENDDO
            IF (I.NE.M) THEN
              DO J=L,M
                S=0.0
                DO K=L,N
                  S=S+A(k,j)*A(k,i)
		ENDDO
                DO K=L,N
                  A(k,j)=A(k,j)+S*RV1(K)
		ENDDO
              ENDDO
            ENDIF
            DO K=L,N
              A(k,i)=SCALE*A(k,i)
	    ENDDO
          ENDIF
        ENDIF
        ANORM=MAX(ANORM,(ABS(W(I))+ABS(RV1(I))))
      ENDDO
      DO I=N,1,-1
        IF (I.LT.N) THEN
          IF (G.NE.0.0) THEN
            DO J=L,N
              V(J,I)=(A(j,i)/A(l,i))/G
	    ENDDO
            DO J=L,N
              S=0.0
              DO K=L,N
                S=S+A(k,i)*V(K,J)
	      ENDDO
              DO K=L,N
                V(K,J)=V(K,J)+S*V(K,I)
	      ENDDO
	    ENDDO
          ENDIF
          DO J=L,N
            V(I,J)=0.0
            V(J,I)=0.0
	  ENDDO
        ENDIF
        V(I,I)=1.0
        G=RV1(I)
        L=I
      ENDDO
      DO I=N,1,-1
        L=I+1
        G=W(I)
        IF (I.LT.N) THEN
          DO J=L,N
            A(j,i)=0.0
	  ENDDO
        ENDIF
        IF (G.NE.0.0) THEN
          G=1.0/G
          IF (I.NE.N) THEN
            DO J=L,N
              S=0.0
              DO K=L,M
                S=S+A(i,k)*A(j,k)
              ENDDO
              F=(S/A(I,I))*G
              DO K=I,M
                A(j,k)=A(j,k)+F*A(i,k)
              ENDDO
            ENDDO
          ENDIF
          DO J=I,M
            A(i,j)=A(i,j)*G
          ENDDO
        ELSE
          DO J= I,M
            A(i,j)=0.0
          ENDDO
        ENDIF
        A(I,I)=A(I,I)+1.0
      ENDDO
      DO K=N,1,-1
        DO ITS=1,30
          DO L=K,1,-1
            NM=L-1
            IF ((ABS(RV1(L))+ANORM).EQ.ANORM)  GO TO 2
            IF ((ABS(W(NM))+ANORM).EQ.ANORM)  EXIT  ! from loop
	  ENDDO
1         C=0.0
          S=1.0
          DO I=L,K
            F=S*RV1(I)
            IF ((ABS(F)+ANORM).NE.ANORM) THEN
              G=W(I)
              H=SQRT(F*F+G*G)
              W(I)=H
              H=1.0/H
              C= (G*H)
              S=-(F*H)
              DO J=1,M
                Y=A(nm,j)
                Z=A(i,j)
                A(nm,j)=(Y*C)+(Z*S)
                A(i,j)=-(Y*S)+(Z*C)
	      ENDDO
            ENDIF
          ENDDO
2         CONTINUE
	  Z=W(K)
          IF (L.EQ.K) THEN
            IF (Z.LT.0.0) THEN
              W(K)=-Z
              DO J=1,N
                V(J,K)=-V(J,K)
              ENDDO
            ENDIF
            GO TO 3
          ENDIF
          IF (ITS.EQ.60) then
	        ier = -1
cc		write(6,*) 'SVDEOF. No convergence in 60 iterations.'
cc		write(6,*) 'Returning with no EOFs found......'
		return
	  endif
          X=W(L)
          NM=K-1
          Y=W(NM)
          G=RV1(NM)
          H=RV1(K)
          F=((Y-Z)*(Y+Z)+(G-H)*(G+H))/(2.0*H*Y)
          G=SQRT(F*F+1.0)
          F=((X-Z)*(X+Z)+H*((Y/(F+SIGN(G,F)))-H))/X
          C=1.0
          S=1.0
          DO J=L,NM
            I=J+1
            G=RV1(I)
            Y=W(I)
            H=S*G
            G=C*G
            Z=SQRT(F*F+H*H)
            RV1(J)=Z
            C=F/Z
            S=H/Z
            F= (X*C)+(G*S)
            G=-(X*S)+(G*C)
            H=Y*S
            Y=Y*C
            DO NM=1,N
              X=V(NM,J)
              Z=V(NM,I)
              V(NM,J)= (X*C)+(Z*S)
              V(NM,I)=-(X*S)+(Z*C)
	    ENDDO
            Z=SQRT(F*F+H*H)
            W(J)=Z
            IF (Z.NE.0.0) THEN
              Z=1.0/Z
              C=F*Z
              S=H*Z
            ENDIF
            F= (C*G)+(S*Y)
            X=-(S*G)+(C*Y)
            DO NM=1,M
              Y=A(j,nm)
              Z=A(i,nm)
              A(j,nm)= (Y*C)+(Z*S)
              A(i,nm)=-(Y*S)+(Z*C)
            ENDDO
          ENDDO
          RV1(L)=0.0
          RV1(K)=F
          W(K)=X
	ENDDO
3       CONTINUE
      ENDDO
c-----------------------------------------------------------------------
c....now do scaling in sub SCALING (allows SVDEOF to handle either transposition)
c-----------------------------------------------------------------------
c...............make sure that eigenvalues are arranged in size order.
c...............if not, rearrange. Check first, may not need it.
	do ieof=2,n
	  if (w(ieof).gt.w(ieof-1)) go to 201
	enddo
	go to 299

c...............Since both the v's and a's must be rearranged, use
c...............the NR subroutine indexx to make a rank table, then
c...............rearrange in this program by hand.
201     continue
	call indexx (n,w,indx)

c...............Now rearrange using index table INDX. Use RV1 as dummy.
c...............Note that table is in ASCENDING order 
c...............(W(INDX(1)) is smallest, W(INDX(N)) is largest).
	do ieof=1,n
	  rv1(ieof)=w(ieof)
	enddo
	do ieof=1,n
	  w(ieof)=rv1(indx(n-ieof+1))
	enddo

	do ieof=1,n
	  rv1(ieof)=pct(ieof)
	enddo
	do ieof=1,n
	  pct(ieof)=rv1(indx(n-ieof+1))
	enddo

c.............for V and A (2-D arrays), must rearrange each row/col separately.
	do ix=1,n
	  do ieof=1,n
	    rv1(ieof)=v(ix,ieof)
	  enddo
	  do ieof=1,n
	    v(ix,ieof)=rv1(indx(n-ieof+1))
	  enddo
	enddo

	do it=1,m
	  do ieof=1,n
	    rv1(ieof)=a(ieof,it)
	  enddo
	  do ieof=1,n
	    a(ieof,it)=rv1(indx(n-ieof+1))
	  enddo
	enddo
c-----------------------------------------------------------------------
299   continue

      RETURN
      END

c***********************************************************************

c...............another NR subroutine.
c...............Makes an index table INDX(N), such that ARRIN(INDX(J))
c					is in ascending order for J=1,2,...,N. 
c...............The inputs N and ARRIN are not changed.
c...............Method is heapsort.

      SUBROUTINE INDEXX(N,ARRIN,INDX)
	IMPLICIT REAL(A-H, O-Z), INTEGER(I-N)

      DIMENSION ARRIN(N),INDX(N)

      DO J=1,N
        INDX(J)=J
      ENDDO
      L=N/2+1
      IR=N

10    CONTINUE

        IF(L.GT.1)THEN
          L=L-1
          INDXT=INDX(L)
          Q=ARRIN(INDXT)
        ELSE
          INDXT=INDX(IR)
          Q=ARRIN(INDXT)
          INDX(IR)=INDX(1)
          IR=IR-1
          IF(IR.EQ.1)THEN
            INDX(1)=INDXT
            RETURN
          ENDIF
        ENDIF
        I=L
        J=L+L

20      CONTINUE

	IF(J.LE.IR)THEN
          IF(J.LT.IR)THEN
            IF(ARRIN(INDX(J)).LT.ARRIN(INDX(J+1)))J=J+1
          ENDIF
          IF(Q.LT.ARRIN(INDX(J)))THEN
            INDX(I)=INDX(J)
            I=J
            J=J+J
          ELSE
            J=IR+1
          ENDIF
        GO TO 20
        ENDIF
        INDX(I)=INDXT

      GO TO 10
      END

c***********************************************************************

c...............subroutine to transpose matrices
c...............from http://www.netlib.org/toms/513

      SUBROUTINE MATRIXTRANS(A, M, N, MN, MOVE, IWRK, IOK)
	IMPLICIT REAL(A-H, O-Z), INTEGER(I-N)
C *****
C  ALGORITHM 380 - REVISED
C *****
C  A IS A ONE-DIMENSIONAL ARRAY OF LENGTH MN=M*N, WHICH
C  CONTAINS THE MXN MATRIX TO BE TRANSPOSED (STORED
C  COLUMWISE). MOVE IS A ONE-DIMENSIONAL ARRAY OF LENGTH IWRK
C  USED TO STORE INFORMATION TO SPEED UP THE PROCESS.  THE
C  VALUE IWRK=(M+N)/2 IS RECOMMENDED. IOK INDICATES THE
C  SUCCESS OR FAILURE OF THE ROUTINE.
C  NORMAL RETURN  IOK=0
C  ERRORS         IOK=-1 ,MN NOT EQUAL TO M*N
C                 IOK=-2 ,IWRK NEGATIVE OR ZERO
C                 IOK.GT.0, (SHOULD NEVER OCCUR),IN THIS CASE
C  WE SET IOK EQUAL TO THE FINAL VALUE OF I WHEN THE SEARCH
C  IS COMPLETED BUT SOME LOOPS HAVE NOT BEEN MOVED
C  NOTE * MOVE(I) WILL STAY ZERO FOR FIXED POINTS
      DIMENSION A(MN), MOVE(*)
C CHECK ARGUMENTS AND INITIALIZE.
      IF (M.LT.2 .OR. N.LT.2) GO TO 120
      IF (MN.NE.M*N) GO TO 180
      IF (IWRK.LT.1) GO TO 190
      IF (M.EQ.N) GO TO 130
      NCOUNT = 2
      K = MN - 1
      DO I=1,IWRK
        MOVE(I) = 0
      ENDDO
      IF (M.LT.3 .OR. N.LT.3) GO TO 30
C CALCULATE THE NUMBER OF FIXED POINTS, EUCLIDS ALGORITHM
C FOR GCD(M-1,N-1).
      IR2 = M - 1
      IR1 = N - 1

   20 CONTINUE
      IR0 = MOD(IR2,IR1)
      IR2 = IR1
      IR1 = IR0
      IF (IR0.NE.0) GO TO 20
      NCOUNT = NCOUNT + IR2 - 1

C SET INITIAL VALUES FOR SEARCH
   30 CONTINUE
      I = 1
      IM = M

C AT LEAST ONE LOOP MUST BE RE-ARRANGED
      GO TO 80

C SEARCH FOR LOOPS TO REARRANGE
   40 CONTINUE
      MAX = K - I
      I = I + 1
      IF (I.GT.MAX) GO TO 160
      IM = IM + M
      IF (IM.GT.K) IM = IM - K
      I2 = IM
      IF (I.EQ.I2) GO TO 40
      IF (I.GT.IWRK) GO TO 60
      IF (MOVE(I).EQ.0) GO TO 80
      GO TO 40

   50 CONTINUE
      I2 = M*I1 - K*(I1/N)

   60 CONTINUE
      IF (I2.LE.I .OR. I2.GE.MAX) GO TO 70
      I1 = I2
      GO TO 50

   70 CONTINUE
      IF (I2.NE.I) GO TO 40

C REARRANGE THE ELEMENTS OF A LOOP AND ITS COMPANION LOOP
   80 CONTINUE
      I1 = I
      KMI = K - I
      B = A(I1+1)
      I1C = KMI
      C = A(I1C+1)

   90 CONTINUE
      I2 = M*I1 - K*(I1/N)
      I2C = K - I2
      IF (I1.LE.IWRK) MOVE(I1) = 2
      IF (I1C.LE.IWRK) MOVE(I1C) = 2
      NCOUNT = NCOUNT + 2
      IF (I2.EQ.I) GO TO 110
      IF (I2.EQ.KMI) GO TO 100
      A(I1+1) = A(I2+1)
      A(I1C+1) = A(I2C+1)
      I1 = I2
      I1C = I2C
      GO TO 90

C FINAL STORE AND TEST FOR FINISHED
  100 CONTINUE
      D = B
      B = C

      CONTINUE
      C = D
  110 A(I1+1) = B
      A(I1C+1) = C
      IF (NCOUNT.LT.MN) GO TO 40

C NORMAL RETURN
  120 CONTINUE
      IOK = 0
      RETURN

C IF MATRIX IS SQUARE,EXCHANGE ELEMENTS A(I,J) AND A(J,I).
  130 CONTINUE
      N1 = N - 1
      DO I=1,N1
        J1 = I + 1
        DO J=J1,N
          I1 = I + (J-1)*N
          I2 = J + (I-1)*M
          B = A(I1)
          A(I1) = A(I2)
          A(I2) = B
        ENDDO
      ENDDO
      GO TO 120

C ERROR RETURNS.
  160 CONTINUE
      IOK = I

  170 CONTINUE
      RETURN

  180 CONTINUE
      IOK = -1
      GO TO 170

  190 CONTINUE
      IOK = -2
      GO TO 170
      END
