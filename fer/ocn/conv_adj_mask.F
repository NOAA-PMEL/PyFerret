	SUBROUTINE  CONV_ADJ_mask( t    , mt,
     .				   s    , ms,
     .				   cmask, mcmask )

* compute a mask with bad data flags indicating regions of convective adjustment
* and 1's indicating regions without it
* note: at K=1 this algorithm fails to detect a pass #1 convective adjustment
*	that gets wiped out by pass #2 - ie k=1 is mixed with  k=2 on pass one,
*	then k=2 is mixed with k=3 on pass #2 --> K=1 and K=2 are unequal at the
*	end
*	At K=km it will succeed or fail depending on whether km is odd or even

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 -7/22/87
*			with major modifications to convert i-k oriented
*			calculations to i-j oriented calculations
* V200:  7/27/89 - 4D symmetrical version
*	10/11/89 - modified array declarations using XMEM_SUBSC.CMN

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/gfdl.parm'		! parameter definitions
	include	'ferret_cmn/xvariables.cmn'
	include	'ferret_cmn/xmem_subsc.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:GFDL.PARM'		! parameter definitions
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XMEM_SUBSC.CMN'
#endif

* calling argument declarations:
	INTEGER	mt, ms, mcmask

* subscript ranges from memory variable table ...
	REAL        t( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit ),
     .		    s( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,m2lot:m2hit ),
     .		cmask( m3lox:m3hix,m3loy:m3hiy,m3loz:m3hiz,m3lot:m3hit )

* internal variable declarations:
	INTEGER	i_lo, i_hi, j_lo, j_hi, i, j, k, l
	REAL	bad_t, bad_cmask

****************   205 model code   ******************************************
******************************************************************************
* 3199 C   FIND DENSITY FOR STABILITY DETERMINATION
* 3200 C
* 3201       DO 965 KS=1,2
* 3202       CALL STATEC(TA,TA(1,1,2),RHON,TDIF,TDIF(1,1,2),KS)
* 3203 C
* 3204 C   CONVECTIVELY ADJUST ADJACENT PAIRS OF BOXES IF GRAVITATIONALLY UNSTA
* 3205 C
* 3206       DO 960 M=1,NT1
* 3207       TEMP(1,1;IMT*KM)=0.5*C2DZ2D(1,1;IMT*KM)*TA(1,1,M;IMT*KM)
* 3208       LN=(2*((KMM1-KS)/2)+1)*IMT
* 3209       WHERE (KALTBV(1,KS,KS;LN).AND.RHON(1,KS;LN).GT.RHON(1,KS+1;LN))
* 3210         TA(1,KS,M;LN)=(TEMP(1,KS;LN)+TEMP(1,KS+1;LN))
* 3211      *                 *DZZ2R2D(1,KS+1;LN)
* 3212         TA(1,KS+1,M;LN)=TA(1,KS,M;LN)
* 3213       ENDWHERE
* 3214  960  CONTINUE
* 3215  965  CONTINUE
*****************************************************************************
* --- end of introductory code ---

* limits for calculation
	i_lo = mr_lo_s1(mcmask)
	i_hi = mr_hi_s1(mcmask)
	j_lo = mr_lo_s2(mcmask)
	j_hi = mr_hi_s2(mcmask)

* flag for bad/missing data
	bad_t    = mr_bad_data( mt )
	bad_cmask = bad_val4

	DO 2000 l = mr_lo_s4(mcmask), mr_hi_s4(mcmask)
	DO 2000 k = mr_lo_s3(mcmask), mr_hi_s3(mcmask)

* preset entire mask to a value of 1.0
	DO 10 j = j_lo, j_hi
	DO 10 i = i_lo, i_hi
 10	cmask(i,j,k,l) = 1.0

* since convective adjustment is computed in 2 passes we presume a point has
* been convectively adjusted in this time step if its density is identical to
* the density immediately above or below

	IF ( k .GT. 1 ) THEN
* ... check against plane above
	   DO 200 j = j_lo, j_hi
	      DO 100 i = i_lo, i_hi

* ... only check temp for bad data - assume salt is bad at the same points
	         IF ( t(i,j,k-1,l) .EQ. bad_t
     .	         .OR. t(i,j,k  ,l) .EQ. bad_t ) THEN
	            cmask(i,j,k,l) = bad_cmask

	         ELSEIF (t(i,j,k,l) .EQ. t(i,j,k-1,l)
     .	          .AND.  s(i,j,k,l) .EQ. s(i,j,k-1,l) ) THEN
	            cmask(i,j,k,l) = bad_cmask

	         ENDIF
 100	      CONTINUE
 200	   CONTINUE	
	ENDIF

	IF ( k .LT. km ) THEN
* ... check against plane below
	   DO 400 j = j_lo, j_hi
	      DO 300 i = i_lo, i_hi

* ... only check temp for bad data - assume salt is bad at the same points
	         IF ( t(i,j,k+1,l) .EQ. bad_t
     .	         .OR. t(i,j,k  ,l) .EQ. bad_t ) THEN
	            cmask(i,j,k,l) = bad_cmask

	         ELSEIF (t(i,j,k,l) .EQ. t(i,j,k+1,l)
     .	          .AND.  s(i,j,k,l) .EQ. s(i,j,k+1,l) ) THEN
	            cmask(i,j,k,l) = bad_cmask

	         ENDIF
 300	      CONTINUE
 400	   CONTINUE	
	ENDIF

 2000	CONTINUE
	RETURN
	END
