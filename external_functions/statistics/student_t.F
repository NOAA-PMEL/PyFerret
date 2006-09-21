	!******************************************
	! 
	! Get the upper cutoff point of the Student T distribution 
	! (P.341 Koopmans .The Spectral Analysis of Time Series)
	!
	!******************************************
	   
	REAL FUNCTION GetStudentT(P,Nf)
        IMPLICIT NONE
	INTEGER Nf,P,df  
	REAL ST_90(15)
	REAL ST_95(15)
	REAL ST_99(15)
	REAL T
	DATA ST_90/2.92,2.132,1.943,1.86,1.812,1.782,1.761,1.746,
     .           1.734,1.725,1.717,1.711,1.706,1.701,1.697/
	DATA ST_95/4.303,2.776,2.447,2.306,2.228,2.179,2.145,2.12,
     .           2.101,2.086,2.074,2.064,2.056,2.048,2.042/
  	DATA ST_99/9.925,4.604,3.707,3.355,3.169,3.055,2.977,2.921,
     .           2.878,2.845,2.819,2.797,2.779,2.763,2.75/

	IF(P.NE.90.AND.P.NE.95.AND.P.NE.99)THEN
           P=90 
	   WRITE(6,*)'WARNING::P must be 90,95 or 99' 
           WRITE(6,*)'         in SUBROUTINE PhaseConfidenceLimits'
           WRITE(6,*)'  '
           WRITE(6,*)'P set to 90%'
	ENDIF
 
        df=Nf*2

	IF(df.GT.120)THEN
     	  IF(P.EQ.90)THEN
		T=1.645
          ELSEIF(P.EQ.95)THEN
		T=1.96
	  ELSE  !  P.EQ.99
		T=2.576
	  ENDIF
	ELSEIF(df.GT.80)THEN
	  IF(P.EQ.90)THEN
		T=1.658
          ELSEIF(P.EQ.95)THEN
		T=1.98
	  ELSE  !  P.EQ.99
		T=2.617
	  ENDIF
	ELSEIF(df.GT.50)THEN
	  IF(P.EQ.90)THEN
		T=1.671
          ELSEIF(P.EQ.95)THEN
		T=2.
	  ELSE  !  P.EQ.99
		T=2.66
	  ENDIF
	ELSEIF(df.GT.30)THEN
	  IF(P.EQ.90)THEN
		T=1.684
          ELSEIF(P.EQ.95)THEN
		T=2.021
	  ELSE  !  P.EQ.99
		T=2.704
	  ENDIF
	ELSE ! df <= 30
	  IF(P.EQ.90)THEN
	    T=ST_90(Nf)
          ELSEIF(P.EQ.95)THEN
	    T=ST_95(Nf)
	  ELSE  !  P.EQ.99
	    T=ST_99(Nf)
	  ENDIF
	ENDIF
	GetStudentT=T
	END



