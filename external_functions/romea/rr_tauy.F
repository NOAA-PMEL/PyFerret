*  rr_tauy.F
*
*     Rick Romea
*     Feb. 4, 2000
*
*   Sea Surface Wind Stress  (DYNE/CM**2)  
*  
************************************************************************

      SUBROUTINE RR_tauy_init(id)
      IMPLICIT NONE
      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INTEGER id

      CALL ef_set_desc             (id, 
     .       'Meridional Wind Stress(dyn/cm^3)')
      CALL ef_set_num_args         (id, 4)
      CALL ef_set_axis_inheritance (id, IMPLIED_BY_ARGS,IMPLIED_BY_ARGS, 
     .                                  IMPLIED_BY_ARGS,IMPLIED_BY_ARGS)
      CALL ef_set_axis_reduction   (id, RETAINED,RETAINED,         
     .                                  RETAINED,RETAINED)          
      CALL ef_set_arg_name         (id, ARG1, 'U')
      CALL ef_set_arg_desc         (id, ARG1, 'Zonal wind speed')
      CALL ef_set_arg_unit         (id, ARG1, 'm/s')
      CALL ef_set_arg_type         (id, ARG1, FLOAT_ARG)
      CALL ef_set_axis_influence   (id, ARG1, YES,YES,YES,YES)

      CALL ef_set_arg_name         (id, ARG2, 'V')
      CALL ef_set_arg_desc         (id, ARG2, 'Meridional wind speed')
      CALL ef_set_arg_unit         (id, ARG2, 'm/s')
      CALL ef_set_arg_type         (id, ARG2, FLOAT_ARG)
      CALL ef_set_axis_influence   (id, ARG2, YES,YES,YES,YES)

      CALL ef_set_arg_name         (id, ARG3, 'AIRT')
      CALL ef_set_arg_desc         (id, ARG3, 'AIR TEMPERATURE')
      CALL ef_set_arg_unit         (id, ARG3, 'oC')
      CALL ef_set_arg_type         (id, ARG3, FLOAT_ARG)
      CALL ef_set_axis_influence   (id, ARG3, YES,YES,YES,YES)

      CALL ef_set_arg_name         (id, ARG4, 'SST')
      CALL ef_set_arg_desc         (id, ARG4, 'Sea Surface Temperature')
      CALL ef_set_arg_unit         (id, ARG4, 'oC')
      CALL ef_set_arg_type         (id, ARG4, FLOAT_ARG)
      CALL ef_set_axis_influence   (id, ARG2, YES,YES,YES,YES)

      END

c*******************************************************************

      SUBROUTINE RR_tauy_compute(id,arg_1,arg_2,arg_3,arg_4,result)
      IMPLICIT NONE
      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id

      REAL bad_flag(EF_MAX_ARGS)
      REAL bad_flag_result
      REAL arg_1  (mem1lox:mem1hix, mem1loy:mem1hiy, 
     .             mem1loz:mem1hiz, mem1lot:mem1hit)
      REAL arg_2  (mem2lox:mem2hix, mem2loy:mem2hiy, 
     .             mem2loz:mem2hiz, mem2lot:mem2hit)
      REAL arg_3  (mem3lox:mem3hix, mem3loy:mem3hiy, 
     .             mem3loz:mem3hiz, mem3lot:mem3hit)
      REAL arg_4  (mem4lox:mem4hix, mem4loy:mem4hiy, 
     .             mem4loz:mem4hiz, mem4lot:mem4hit)
      REAL result (memreslox:memreshix, memresloy:memreshiy, 
     .             memresloz:memreshiz, memreslot:memreshit)

      INTEGER  res_lo_ss (4)
      INTEGER  res_hi_ss (4)
      INTEGER  res_incr  (4)

      INTEGER  arg_lo_ss (4,EF_MAX_ARGS)
      INTEGER  arg_hi_ss (4,EF_MAX_ARGS)
      INTEGER  arg_incr  (4,EF_MAX_ARGS)

      INTEGER  i,  j,  k,  l
      INTEGER  i1, j1, l1
      INTEGER  i2, j2, l2
      INTEGER  i3, j3, l3
      INTEGER  i4, j4, l4

      REAL AirDensity
      PARAMETER (AirDensity = 0.00125)
      REAL zu
      PARAMETER (zu = 10.)
      REAL zt
      PARAMETER (zt = 10.)
      REAL zq
      PARAMETER (zq = 0.)
      REAL Cd, uz
      REAL QZ,TOL,UW,WT,WQ

      CALL ef_get_res_subscripts (id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts (id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags      (id, bad_flag,  bad_flag_result)

      ! This is a dummy axis, since z is reduced
      DO k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

        l1 = arg_lo_ss(T_AXIS,ARG1)
        l2 = arg_lo_ss(T_AXIS,ARG2)
        l3 = arg_lo_ss(T_AXIS,ARG3)
        l4 = arg_lo_ss(T_AXIS,ARG4)

        DO l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

          j1 = arg_lo_ss(Y_AXIS,ARG1) 
          j2 = arg_lo_ss(Y_AXIS,ARG2) 
          j3 = arg_lo_ss(Y_AXIS,ARG2) 
          j4 = arg_lo_ss(Y_AXIS,ARG3) 
        
          DO j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

            i1 = arg_lo_ss(X_AXIS,ARG1) 
            i2 = arg_lo_ss(X_AXIS,ARG2)
            i3 = arg_lo_ss(X_AXIS,ARG3)
            i4 = arg_lo_ss(X_AXIS,ARG4)
 
            DO i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)
              
              IF ( arg_1(i1,j1,k,l1) .EQ. bad_flag(ARG1) .OR.
     .            arg_2(i2,j2,k,l2) .EQ. bad_flag(ARG2) .OR.
     .            arg_3(i3,j3,k,l3) .EQ. bad_flag(ARG3) .OR.
     .            arg_4(i4,j4,k,l4) .EQ. bad_flag(ARG4) )THEN
                 result(i,j,k,l) = bad_flag_result
              ELSE

                uz = SQRT(arg_1(i1,j1,k,l1)*arg_1(i1,j1,k,l1)
     .                +  arg_2(i2,j2,k,l2)*arg_2(i2,j2,k,l2))
                IF (uz .LT. 0.1) THEN
                  result(i,j,k,l) = 0.
                ELSE 
                  CALL Bulk(uz,zu,arg_3(i3,j3,k,l3),zt,QZ,zq,
     .                       arg_4(i4,j4,k,l4),TOL,UW,WT,WQ,Cd)
                  result(i,j,k,l) = AirDensity * Cd * uz 
     .                                * arg_2(i2,j2,k,l2)*1.e4
                ENDIF 
              ENDIF

              i1 = i1 + arg_incr(X_AXIS,ARG1)
              i2 = i2 + arg_incr(X_AXIS,ARG2)
              i3 = i3 + arg_incr(X_AXIS,ARG3)
              i4 = i4 + arg_incr(X_AXIS,ARG4)
            ENDDO ! i index

            j1 = j1 + arg_incr(Y_AXIS,ARG1)
            j2 = j2 + arg_incr(Y_AXIS,ARG2)
            j3 = j3 + arg_incr(Y_AXIS,ARG3)
            j4 = j4 + arg_incr(Y_AXIS,ARG4)
          ENDDO ! j index

          l1 = l1 + arg_incr(T_AXIS,ARG1)
          l2 = l2 + arg_incr(T_AXIS,ARG2)
          l3 = l3 + arg_incr(T_AXIS,ARG3)
          l4 = l4 + arg_incr(T_AXIS,ARG4)
        ENDDO ! l index

      ENDDO  ! k index

      END





	SUBROUTINE BULK(UZ,ZU,TZ,ZT,QZ,ZQ,TSFC,TOL,UW,WT,WQ,CD)

*	Minor (I hope) mods made for SSMI stress calc's
*	J Davison 1.24.91
*	No mods made for ECMWF stress calc's
*	J Davison 3.22.93

C	FOLLOWING PROCEDURES AND COEFFICIENTS IN LARGE AND POND (1981 AND
C	1982 BULK; ESTIMATE
C		1) TOL, THE STABILITY PARAMETER Z/L AT Z=10 METERS
C		2)  UW, THE KINEMATIC MOMENTUM FLUX (M/S)**2
C		3)  WT, THE KINEMATIC SENSIBLE HEAT FLUX (DEGREES M/S)
C		4)  WQ, THE KINEMATIC LATENT HEAT FLUX (M/S, G/M**3)
C
C	FROM	1)  UZ, THE MEAN WIND SPEED  [M/S -- JD 1.24.91] AT HEIGHT ZU
C		2)  TZ, THE MEAN AIR TEMPERATURE (CELSIUS)       AT HEIGHT ZT
C		3)  QZ, THE MEAN AIR HUMIDITY (G/M**3)           AT HEIGHT ZQ
C		4) TSFC, THE SEA SURFACE TEMPERATURE (CELSIUS)
C
c      OPTIONS  1) set ZQ=0 to assume a relative humidity of RH=75%
c                DEH changed so RH is set in calling pgm and passed thru common
c                if zq=0.    change made 12/30/85

*		 Not 'passed thru common' but initialized in a data stmt JD 1.91

C		2) SET ZT=0 TO FIND ONLY UW AT NEUTRAL STABILITY
C		3) SET ZU=0 TO RETURN ONLY WITH TOL
C
C     ASSUME	1) A NEUTRAL 10M DRAG COEFFICIENT CDN=0.00115  U10<UCH  11M/S
C				          =0.00049 + 0.000065  U10>UCH  11M/S
C		2) A NEUTRAL 10M STANTON NUMBER   CTN=0.00115  Z/L<0    0
C						     =0.00075  Z/L>0    0
C		3) A NEUTRAL 10M DALTON NUMBER    CEN=0.00115  ALWAYS
C		4) VONK = 0.40 VON KARMAN'S CONSTANT
C		5) THE SMALL WT REGRESSION, WTR BELOW IF BIG=FALSE
C		       WITH A=(.0008,  Z/L>0)   =(.0010,  Z/L<0)
	LOGICAL BIG

*  ARGUMENT DECLARATIONS

      REAL UZ, ZU, TZ, ZT, QZ, ZQ, TSFC, TOL, UW, WT, WQ, CD

*  LOCAL VARIABLE DECLARATIONS

      INTEGER iter
      REAL    a, u, dth, t, cda, cts, ctu, cea, uch, rh, vonk, delth, 
     .        qsat, delq, tkelv, f1, t0, udt, f2, wtr, u3, zuol, ztol, 
     .        zqol, p10, pg10, s10, sg10, phim, phi, psim, phit, psimt, 
     .        psit, phiq, psimq, rk, up, cp, u10, diff, cdn, sqcd, rcd,
     .        ctn, rcdt, ct, rkt, dth10, cen, rcdq, ce, psi, psiq

	WTR(A,U,DTH) = 0.002 + A*U*DTH
C
C		6) THE SATURATION HUMIDITY OF AIR AT T CELSIUS QSAT(T) SAT(T) I
C
	QSAT(T) = 640380000./EXP(5107.4/(273.16 + T))
C
	DATA CDA,CTS,CTU,CEA,UCH/0.00115,0.00075,0.00115,0.00115,10./
	DATA RH,VONK/0.75,0.40/

*	Mod made to incorporate INITAL2 within the BULK subroutine JD 1.24.91
	tol = 0.0
	uw  = 0.0
	wt  = 0.0
	wq  = 0.0
	cd  = 0.0
*	End mod JD 1.24.91

C				ARE THE TEMPERATURES RELIABLE??

	IF(ZT .GT. 0.2) GO TO 1
	TOL = 0.0
	GO TO 3
C
C YES		CALCULATE A BULK STABILITY PARAMETER AT 10 METERS FROM
C		   DELTH, SURFACE-AIR POTENTIAL TEMPERATURE DIFFERENCE
C		   QSFC,  SURFACE  ABSOLUTE  HUMIDITY
C		   T0,    LOCAL MEAN VIRTUAL TEMPERATURE
   1	DELTH = TSFC-TZ-0.01*ZT
	IF(ZQ .LE. 0.0) QZ = RH*QSAT(TZ)
	DELQ = 0.98*QSAT(TSFC)-QZ
	TKELV = 273.16 + TZ
	F1 = 0.00000172
	T0 = TKELV + QZ*F1*TKELV**2
C
C		IS /WT/ LARGE ENOUGH TO USE TRANSFER COEFFICIENTS?
C
	UDT = UZ*DELTH
	IF((UDT .LT. 10.) .AND. (UDT .GT. -15.)) GO TO 2
C YES
	BIG = .TRUE.
	F2 = 1.00
	IF(DELTH .LT. 0.0) F2 = 0.70
	TOL = 0.0 - 1000.*F2/T0/UZ**2*(DELTH + F1/F2*DELQ*T0**2)
	GO TO 3
C
C NO		       DEAL WITH SMALL SENSIBLE HEAT FLUXES
C
   2	BIG = .FALSE.
	A = 0.0010
	IF(DELTH .LT. 0.0) A = 0.0008
	WT = WTR(A,UZ,DELTH)
	WQ = CEA*UZ*DELQ
	U3 = (SQRT(CDA)*UZ)**3
	TOL = 0.0 - 39./U3/T0*(WT + F1*WQ*T0**2)
C
C		RETURN WITH ONLY TOL IF ZU NOT GREATER THAN 0.2
C 
  3	IF(ZU .LE. 0.2) RETURN
	ZUOL = TOL*ZU/10.0
	ZTOL = TOL*ZT/10.0
	ZQOL = TOL*ZQ/10.0
C		EVALUATE THE STABILITY FUNCTIONS IN FZOL
	CALL FZOL(TOL,P10,PG10,S10,SG10)
	CALL FZOL(ZUOL,PHIM,PHI,PSIM,PSI)
	CALL FZOL(ZTOL,PHI,PHIT,PSIMT,PSIT)
	CALL FZOL(ZQOL,PHI,PHIQ,PSIMQ,PSIQ)
C
C	SOLVE FOR U10 AND CDN ITERATIVELY
C
	RK = (ALOG(ZU/10.0) - PSIM + S10)/VONK
	ITER = 1
	UP = UZ/(1.0 + SQRT(CDA)*RK)
  22	CP = CDA
	IF(UP .GT. UCH) CP = (0.49 + 0.065*UP)/1000.
	U10 = UZ/(1.0 + SQRT(CP)*RK)
	DIFF = (UP - U10)/UP
	UP = U10
	IF(DIFF .LE. 0.01) GO TO 4
	ITER = ITER + 1
	IF(ITER .LE. 5) GO TO 22
	WRITE(8,66) ITER,U10,DIFF,CP
  66	FORMAT('-',5X,'TOO BAD, FAILED TO CONVERGE IN 5 ITERATIONS',
     .  I6,3F16.4)
	STOP
C
C		SOLVED FOR U10, NOW FIND CDN AND CD
C		THEN GET UW FROM BULK FORMULA
C
   4	CDN = CDA
	IF(U10 .GT. UCH) CDN = (0.49 + 0.065*U10)/1000.
	SQCD = SQRT(CDN)
	RCD = 1.0 + SQCD/VONK*(ALOG(ZU/10.0) - PSIM)
	CD = CDN/RCD**2
	UW = 0.0 - CD*UZ**2
C
C		SENSIBLE HEAT FLUX IF ZT SET GREATER THAN 0.02
C
	IF(ZT .LE. 0.02) GO TO 5
	CTN = CTU
	IF(DELTH .LT. 0.0) CTN = CTS
	RCDT = 1.0 + SQCD/VONK*(ALOG(ZT/10.0) - PSIMT)
	CT = CTN/RCDT/(1.0 + CTN/VONK/SQCD*(ALOG(ZT/10.) - PSIT))
	WT = CT*UZ*DELTH
	IF(BIG) GO TO 5
C			CALCULATE SMALL SENSIBLE HEAT FLUX
	RKT = (ALOG(ZT/10.0) - PSIT + SG10)/VONK
	DTH10 = DELTH - CT/SQRT(CD)*RKT*(DELTH)
	WT = WTR(A,U10,DTH10)
C
C		LATENT (MOISTURE) FLUX IF ZQ SET GREATER THAN 0.02
C
   5	IF(ZQ .LE. 0.02) RETURN
	CEN = CEA
	RCDQ = 1.0 + SQCD/VONK*(ALOG(ZQ/10.0) - PSIMQ)
	CE = CEN/RCDQ/(1.0 + CEN/VONK/SQCD*(ALOG(ZQ/10.) - PSIQ))
	WQ = CE*UZ*DELQ
	RETURN
	END
C
	SUBROUTINE FZOL(ZOL,PHIM,PHIG,PSIM,PSIG)
C FZOL:		SUBROUTINE TO ESTIMATE THE STABILITY FUNCTIONS
C		USING THE MOST UP TO DATE FLUX-PROFILE RELATIONSHIPS
C

*  ARGUMENT DECLARATIONS

        REAL ZOL,PHIM,PHIG,PSIM,PSIG

*  LOCAL VARIABLE DECLARATIONS

	IF(ZOL .GE. 0.0) GO TO 2
C		UNSTABLE
	X = (1.0 - 16.0*ZOL)**0.25
	PHIM = 1.0/X
	PHIG = PHIM**2
	PSIG = 2.0*ALOG(0.5 + 0.5*X**2)
	PSIM = 0.5*PSIG + 2.0*ALOG(0.5*(1.0 + X)) - 2.0*ATAN(X) + 1.571
	RETURN
C
C		STABLE
C
  2	PHIG = 1.0 + 7.0*ZOL
	PHIM = 1.0 + 7.0*ZOL
	PSIG = -7.0*ZOL
	PSIM = -7.0*ZOL
	RETURN
	END

