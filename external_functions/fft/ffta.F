*
* ffta.F
*
* Ansley Manke
* Dec 1998
*
*
* This function computes fft amplitude spectrum for each time series.  Result
* time axis is a custom axis in period 1./time  The input variable must have
* the time axis specified explicitly in the function call e.g. 
*    LET ssffta = ffta(ssttim[l=1:492])
* and the time axis must be regular.

*  Note current limitations as of 28-Apr-1999
*   - result T-axis units is set to FREQ: want to get the time step units and
*     set the frequency axis to 1./days, etc.
*   - Will have a utility to get the length of the input time axis at the
*     point when the custom freq axis is set up, to use in computing the
*     length of the frequency axis.  Currently need to specify explicitly, 
*     in the function call e.g. ffta(sst[l=1:400]).

*
* In this subroutine we provide information about
* the function.  The user configurable information 
* consists of the following:
*
* descr              Text description of the function
*
* num_args           Required number of arguments
*
* axis_inheritance   Type of axis for the result
*                       ( CUSTOM, IMPLIED_BY_ARGS, NORMAL, ABSTRACT )
*                       CUSTOM          - user defined axis
*                       IMPLIED_BY_ARGS - same axis as the incoming argument
*                       NORMAL          - the result is normal to this axis
*                       ABSTRACT        - an axis which only has index values
*
* piecemeal_ok       For memory optimization:
*                       axes where calculation may be performed piecemeal
*                       ( YES, NO )
* 
*
* For each argument we provide the following information:
*
* name               Text name for an argument
*
* unit               Text units for an argument
*
* desc               Text description of an argument
*
* axis_influence     Are this argument's axes the same as the result grid?
*                       ( YES, NO )
*
* axis_extend       How much does Ferret need to extend arg limits relative to result 
*


      SUBROUTINE ffta_init(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

      CHARACTER*110 arg_desc
      CALL ef_version_test(ef_version)

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V
      CALL ef_set_desc(id,'Computes fft amplitude spectra' )

      CALL ef_set_num_args(id, 1)
      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS, 
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, CUSTOM)
      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)

      CALL ef_set_num_work_arrays(id, 5)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'A')
      CALL ef_set_arg_unit(id, arg, ' ')
      WRITE (arg_desc, 10) 
   10 FORMAT ('Variable with regular time axis. Specify time ', 
     .         'explicitly e.g. ffta(var[l=1,120])')
      CALL ef_set_arg_desc(id, arg, arg_desc)

      CALL ef_set_axis_influence(id, arg, YES, YES, YES, NO)


*                                                                     ^
*                                                                     |
*                                           USER CONFIGURABLE PORTION |
***********************************************************************

      RETURN 
      END


*
* In this subroutine we provide information about the lo and hi
* limits associated with each abstract or custom axis.   The user 
* configurable information consists of the following:
*
* lo_ss               lo subscript for an axis
*
* hi_ss               hi subscript for an axis
*

      SUBROUTINE ffta_result_limits(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id

* **********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V

      COMMON /fftalim/ nfreq_lo_l, nfreq_hi_l
      INTEGER nfreq_lo_l, nfreq_hi_l

      INTEGER arg
      INTEGER arg_lo_ss(4,EF_MAX_ARGS), arg_hi_ss(4,EF_MAX_ARGS),
     .     arg_incr(4,EF_MAX_ARGS)


*     Use utility functions to get context information about the argument.

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)

      nfreq_lo_l = 1
      nfreq_hi_l = 1


      arg = 1
      nfreq_hi_l = 
     .        (arg_hi_ss(T_AXIS,arg) - arg_lo_ss(T_AXIS,arg) + 1)/ 2

      CALL ef_set_axis_limits(id, T_AXIS, nfreq_lo_l, nfreq_hi_l)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END

      SUBROUTINE ffta_custom_axes(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INTEGER id

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                      
*
*     Use utility functions to get grid information about the 
*     argument; use this info to determine the custom axis lo/hi/del 

      COMMON /fftalim/ nfreq_lo_l, nfreq_hi_l
      INTEGER nfreq_lo_l, nfreq_hi_l

      INTEGER arg_lo_ss(4,EF_MAX_ARGS), arg_hi_ss(4,EF_MAX_ARGS),
     .     arg_incr(4,EF_MAX_ARGS)

      INTEGER arg, unspecified_int4
      INTEGER nfreq

      REAL yquist, freq1, freqn
      CHARACTER outunits*20

      unspecified_int4 = -111
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)

      arg = 1
      IF ( arg_lo_ss(T_AXIS,arg) .EQ. unspecified_int4 ) THEN
         nfreq_lo_l = 1

      ELSE
         nfreq_lo_l = arg_lo_ss(T_AXIS,arg)
      ENDIF

      IF ( arg_hi_ss(T_AXIS,arg) .EQ. unspecified_int4 ) THEN
         nfreq_hi_l = 2		! use: size of t axis from get_arg_axis_limits

      ELSE
         nfreq_hi_l = arg_hi_ss(T_AXIS,arg)
      ENDIF

      arg = 1
      nfreq = (nfreq_hi_l - nfreq_lo_l + 1)/ 2

      yquist = 0.5              ! Nyquist frequency
      freq1 = 1.* yquist/ float(nfreq)
      freqn = yquist

      outunits = 'FREQ'
      CALL ef_set_custom_axis(id, T_AXIS, freq1, freqn, freq1, 
     .   outunits, NO)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END

*
* In this subroutine we request an amount of storage to be supplied
* by Ferret and passed as an additional argument.
*
      SUBROUTINE ffta_work_size(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
* 
* Set the work arrays,  X/Y/Z/T dimensions
*
* ef_set_work_array_lens(id, array #, X len, Y len, Z len, T len)
*
      COMMON /STOR/ mtdat, mtwork
      INTEGER mtdat, mtwork

      INTEGER arg_lo_ss(4,1:EF_MAX_ARGS), arg_hi_ss(4,1:EF_MAX_ARGS),
     .     arg_incr(4,1:EF_MAX_ARGS)

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)

      mtdat = 1 + arg_hi_ss(T_AXIS,ARG1) - arg_lo_ss(T_AXIS,ARG1)

      mtwork = 2.5* mtdat + 15
  
* creal
      CALL ef_set_work_array_lens (id, 1, mtdat, 1, 1, 1)

* cimag
      CALL ef_set_work_array_lens (id, 2, mtdat, 1, 1, 1)

* wft
      CALL ef_set_work_array_lens (id, 3, mtwork, 1, 1, 1)

* ts
      CALL ef_set_work_array_lens (id, 4, mtdat, 1, 1, 1)

* ws
      CALL ef_set_work_array_lens (id, 5, mtdat, 1, 1, 1)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN
      END


*
* In this subroutine we compute the result
*
      SUBROUTINE ffta_compute(id, arg_1, result, creal, cimag, wft, 
     .                ts, ws)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

          INTEGER id

      REAL bad_flag(1:EF_MAX_ARGS), bad_flag_result
      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, 
     .           mem1loz:mem1hiz, mem1lot:mem1hit)
      REAL result(memreslox:memreshix, memresloy:memreshiy,
     .            memresloz:memreshiz, memreslot:memreshit)

* After initialization, the 'res_' arrays contain indexing information 
* for the result axes.  The 'arg_' arrays will contain the indexing 
* information for each variable's axes. 

      INTEGER res_lo_ss(4), res_hi_ss(4), res_incr(4)
      INTEGER arg_lo_ss(4,1:EF_MAX_ARGS), arg_hi_ss(4,1:EF_MAX_ARGS),
     .     arg_incr(4,1:EF_MAX_ARGS)


* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V
      COMMON /STOR/ mtdat, mtwork
      INTEGER mtdat, mtwork

      CHARACTER ax_name(4)*16, ax_units(4)*16
      LOGICAL backward(4), modulo(4), regular(4)

*  Set up work arrays

      REAL creal(1:mtdat, 1:1, 1:1, 1:1)
      REAL cimag(1:mtdat, 1:1, 1:1, 1:1)
      REAL wft(1:mtwork, 1:1, 1:1, 1:1)
      REAL ts(1:mtdat, 1:1, 1:1, 1:1)
      REAL ws(1:mtdat, 1:1, 1:1, 1:1)

      COMMON /fftalim/ nfreq_lo_l, nfreq_hi_l
      INTEGER i,j,k,l
      INTEGER i1, j1, k1, l1

      CHARACTER*80 err_msg

   40 FORMAT ('FFT encountered missing data at (i,j,k,l)', 4I5)

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

*  FFT routine works on an even number of time points.

      nd = mtdat
      nf = nd/ 2
      IF (nd .NE. nf*2) nd = nd-1

      nr = res_hi_ss(T_AXIS) - res_lo_ss(T_AXIS) + 1

      IF (nr .ne. nf) then
         WRITE (err_msg, *) 'Specify # of time steps explicitly',
     .     ' e.g. LET sstfft = ffta(ssttim[l=1:120])'
         GO TO 999
      ENDIF

*  Check that the time axis is regular.

      arg = 1
      CALL ef_get_axis_info (id, 1, ax_name, ax_units, backward,
     .     modulo, regular)

      IF (.NOT. regular(T_AXIS)) THEN
            WRITE (err_msg, *) 'Time axis must be a regular axis'
            GO TO 999
      ENDIF

*  Set WFT for fft; prime factorization of ND and trig functions.
*  Set weights for 10%, 5%  or 1% cosine window.  
*  Or CALL PROLATE (WS, ND, IP) w/ IP=1 OR 4  for pi or 4pi prolate window.)

      CALL rffti (nd, wft)
      CALL coswindow (ws, nd, 5)

      i1 = arg_lo_ss(X_AXIS,ARG1)
      DO 400 i=res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

         j1 = arg_lo_ss(Y_AXIS,ARG1)
         DO 300 j=res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

            k1 = arg_lo_ss(Z_AXIS,ARG1)
            DO 200 k=res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

*
* Calculate the FFT for each time series which have no missing data.
*
               l1 =arg_lo_ss(T_AXIS,ARG1)
               DO 100 l = 1, nd
                  IF (arg_1(i1,j1,k1,l1) .EQ. bad_flag(ARG1)) THEN
                     WRITE (err_msg, 40) i1,j1,k1,l
                     GO TO 999 
                  ENDIF

                  ts(l,1,1,1) = ws(l,1,1,1)* arg_1(i1,j1,k1,l1)
                  l1 = l1 + arg_incr(T_AXIS,ARG1)
 100           CONTINUE


               l1 = arg_lo_ss(T_AXIS,ARG1)
               l = res_lo_ss(T_AXIS)

               CALL FOUR_RE (nd, ts, creal, cimag, wft)

               DO 110 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS) 
                   rr = creal(l,1,1,1)
                   ri = cimag(l,1,1,1)
                   result(i,j,k,l) = SQRT(rr*rr + ri*ri)
 110           CONTINUE

               k1 = k1 + arg_incr(Z_AXIS,ARG1)
 200        CONTINUE

            j1 = j1 + arg_incr(Y_AXIS,ARG1)
 300     CONTINUE

         i1 = i1 + arg_incr(X_AXIS,ARG1)
 400  CONTINUE

      RETURN

 999  CALL ef_bail_out (id, err_msg)
*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END

* --+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+

      SUBROUTINE FOUR_RE (ND, X, CREAL, CIMAG, WFT)
      REAL X(*), WFT(*) 
      REAL CREAL(*), CIMAG(*), CR, CI

c   uses NCAR FFT code

c   written by Jim Larsen
c   revised 11/19/96

C  Version by Ansley Manke that replaces complex array C with 
C  real arrays CREAL and CIMAG.

c   Calls: RFFTF

      NF = ND/ 2

c   save X

      DO I = 1, NF
        J = I + I
        CREAL(I) =X(J-1)
        CIMAG(I) = X(J)
      ENDDO

      CALL RFFTF (ND, X, WFT)   !  -> 2 Cn exp(iwt)

      XN = 1.0/ ND
      DO I = 1, ND
        X(I) = XN* X(I)
      ENDDO

c   restore X and compute complex Cn

      CR = 0.5* X(2) 
      DO I = 1, NF-1
        J = I + I
        CR = CREAL(I)
        CI = CIMAG(I)
        X(J-1) = CR
        X(J)   = CI
        CREAL(I) = X(J+1)
        CIMAG(I) = X(J+2)
      ENDDO
      CR = CREAL(NF)
      CI = CIMAG(NF)
      X(ND-1) = CR
      X(ND)   = CI
      CREAL(NF) = CN
      CIMAG(NF) = 0.0

      RETURN 
      END


* --+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+

*DECK CFFTB
      SUBROUTINE CFFTB (N,C,WSAVE)
C***BEGIN PROLOGUE  CFFTB
C***REVISION DATE  811015   (YYMMDD)
C***CATEGORY NO.  D6
C***KEYWORDS FFT,FAST FOURIER TRANSFORM,INVERSE,INVERSE FFT,COMPLEX
C***DATE WRITTEN  FEBRUARY 1978
C***AUTHOR  SWARZTRAUBER P.N. (NCAR)
C***PURPOSE
C INVERSE FFT OF A COMPLEX PERIODIC SEQUENCE
C***DESCRIPTION
C     *****************************************************************
C
C     SUBROUTINE CFFTB(N,C,WSAVE)
C
C     *****************************************************************
C
C     SUBROUTINE CFFTB COMPUTES THE BACKWARD COMPLEX DISCRETE FOURIER
C     TRANSFORM (THE FOURIER SYNTHESIS). EQUIVALENTLY , CFFTB COMPUTES
C     A COMPLEX PERIODIC SEQUENCE FROM ITS FOURIER COEFFICIENTS.
C     THE TRANSFORM IS DEFINED BELOW AT OUTPUT PARAMETER C.
C
C     A CALL OF CFFTF FOLLOWED BY A CALL OF CFFTB WILL MULTIPLY THE
C     SEQUENCE BY N.
C
C     THE ARRAY WSAVE WHICH IS USED BY SUBROUTINE CFFTB MUST BE
C     INITIALIZED BY CALLING SUBROUTINE CFFTI(N,WSAVE).
C
C     INPUT PARAMETERS
C
C
C     N      THE LENGTH OF THE COMPLEX SEQUENCE C. THE METHOD IS
C            MORE EFFICIENT WHEN N IS THE PRODUCT OF SMALL PRIMES.
C
C     C      A COMPLEX ARRAY OF LENGTH N WHICH CONTAINS THE SEQUENCE
C
C     WSAVE   A REAL WORK ARRAY WHICH MUST BE DIMENSIONED AT LEAST 4N+1
C             IN THE PROGRAM THAT CALLS CFFTB. THE WSAVE ARRAY MUST BE
C             INITIALIZED BY CALLING SUBROUTINE CFFTI(N,WSAVE) AND A
C             DIFFERENT WSAVE ARRAY MUST BE USED FOR EACH DIFFERENT
C             VALUE OF N. THIS INITIALIZATION DOES NOT HAVE TO BE
C             REPEATED SO LONG AS N REMAINS UNCHANGED THUS SUBSEQUENT
C             TRANSFORMS CAN BE OBTAINED FASTER THAN THE FIRST.
C             THE SAME WSAVE ARRAY CAN BE USED BY CFFTF AND CFFTB.
C
C     OUTPUT PARAMETERS
C
C     C      FOR J=1,...,N
C
C                C(J)=THE SUM FROM K=1,...,N OF
C
C                      C(K)*EXP(I*J*K*2*PI/N)
C
C                            WHERE I=SQRT(-1)
C
C     WSAVE   CONTAINS INITIALIZATION CALCULATIONS WHICH MUST NOT BE
C             DESTROYED BETWEEN CALLS OF SUBROUTINE CFFTF OR CFFTB
C***REFERENCES
C***ROUTINES CALLED  CFFTB1
C***END PROLOGUE  CFFTB
      DIMENSION       C(*)       ,WSAVE(*)
C***FIRST EXECUTABLE STATEMENT  CFFTB
      IF (N .EQ. 1) RETURN
      IW1 = N+N+1
      IW2 = IW1+N+N
      CALL CFFTB1 (N,C,WSAVE,WSAVE(IW1),WSAVE(IW2))
      RETURN
      END

* -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+
*DECK CFFTB1
      SUBROUTINE CFFTB1 (N,C,CH,WA,IFAC)
C***BEGIN PROLOGUE  CFFTB1
C***REFER TO CFFTB
C***ROUTINES CALLED  PASSB,PASSB5,PASSB3,PASSB2,PASSB4
C***END PROLOGUE  CFFTB1
      DIMENSION       CH(*)      ,C(*)       ,WA(*)      ,IFAC(*)
C***FIRST EXECUTABLE STATEMENT  CFFTB1
      NF = IFAC(2)
      NA = 0
      L1 = 1
      IW = 1
      DO 116 K1=1,NF
         IP = IFAC(K1+2)
         L2 = IP*L1
         IDO = N/L2
         IDOT = IDO+IDO
         IDL1 = IDOT*L1
         IF (IP .NE. 4) GO TO 103
         IX2 = IW+IDOT
         IX3 = IX2+IDOT
         IF (NA .NE. 0) GO TO 101
         CALL PASSB4 (IDOT,L1,C,CH,WA(IW),WA(IX2),WA(IX3))
         GO TO 102
  101    CALL PASSB4 (IDOT,L1,CH,C,WA(IW),WA(IX2),WA(IX3))
  102    NA = 1-NA
         GO TO 115
  103    IF (IP .NE. 2) GO TO 106
         IF (NA .NE. 0) GO TO 104
         CALL PASSB2 (IDOT,L1,C,CH,WA(IW))
         GO TO 105
  104    CALL PASSB2 (IDOT,L1,CH,C,WA(IW))
  105    NA = 1-NA
         GO TO 115
  106    IF (IP .NE. 3) GO TO 109
         IX2 = IW+IDOT
         IF (NA .NE. 0) GO TO 107
         CALL PASSB3 (IDOT,L1,C,CH,WA(IW),WA(IX2))
         GO TO 108
  107    CALL PASSB3 (IDOT,L1,CH,C,WA(IW),WA(IX2))
  108    NA = 1-NA
         GO TO 115
  109    IF (IP .NE. 5) GO TO 112
         IX2 = IW+IDOT
         IX3 = IX2+IDOT
         IX4 = IX3+IDOT
         IF (NA .NE. 0) GO TO 110
         CALL PASSB5 (IDOT,L1,C,CH,WA(IW),WA(IX2),WA(IX3),WA(IX4))
         GO TO 111
  110    CALL PASSB5 (IDOT,L1,CH,C,WA(IW),WA(IX2),WA(IX3),WA(IX4))
  111    NA = 1-NA
         GO TO 115
  112    IF (NA .NE. 0) GO TO 113
         CALL PASSB (NAC,IDOT,IP,L1,IDL1,C,C,C,CH,CH,WA(IW))
         GO TO 114
  113    CALL PASSB (NAC,IDOT,IP,L1,IDL1,CH,CH,CH,C,C,WA(IW))
  114    IF (NAC .NE. 0) NA = 1-NA
  115    L1 = L2
         IW = IW+(IP-1)*IDOT
  116 CONTINUE
      IF (NA .EQ. 0) RETURN
      N2 = N+N
      DO 117 I=1,N2
         C(I) = CH(I)
  117 CONTINUE
      RETURN
      END

* -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+
*DECK CFFTF
      SUBROUTINE CFFTF (N,C,WSAVE)
C***BEGIN PROLOGUE  CFFTF
C***REVISION DATE  811015   (YYMMDD)
C***CATEGORY NO.  D6
C***KEYWORDS FFT,FAST FOURIER TRANSFORM,COMPLEX
C***DATE WRITTEN  FEBRUARY 1978
C***AUTHOR  SWARZTRAUBER P.N. (NCAR)
C***PURPOSE
C  FORWARD FFT OF A COMPLEX PERIODIC SEQUENCE
C***DESCRIPTION
C     *****************************************************************
C
C     SUBROUTINE CFFTF(N,C,WSAVE)
C
C     *****************************************************************
C
C     SUBROUTINE CFFTF COMPUTES THE FORWARD COMPLEX DISCRETE FOURIER
C     TRANSFORM (THE FOURIER ANALYSIS). EQUIVALENTLY , CFFTF COMPUTES
C     THE FOURIER COEFFICIENTS OF A COMPLEX PERIODIC SEQUENCE.
C     THE TRANSFORM IS DEFINED BELOW AT OUTPUT PARAMETER C.
C
C     THE TRANSFORM IS NOT NORMALIZED. TO OBTAIN A NORMALIZED TRANSFORM
C     THE OUTPUT MUST BE DIVIDED BY N. OTHERWISE A CALL OF CFFTF
C     FOLLOWED BY A CALL OF CFFTB WILL MULTIPLY THE SEQUENCE BY N.
C
C     THE ARRAY WSAVE WHICH IS USED BY SUBROUTINE CFFTF MUST BE
C     INITIALIZED BY CALLING SUBROUTINE CFFTI(N,WSAVE).
C
C     INPUT PARAMETERS
C
C
C     N      THE LENGTH OF THE COMPLEX SEQUENCE C. THE METHOD IS
C            MORE EFFICIENT WHEN N IS THE PRODUCT OF SMALL PRIMES. N
C
C     C      A COMPLEX ARRAY OF LENGTH N WHICH CONTAINS THE SEQUENCE
C
C     WSAVE   A REAL WORK ARRAY WHICH MUST BE DIMENSIONED AT LEAST 4N+1
C             IN THE PROGRAM THAT CALLS CFFTF. THE WSAVE ARRAY MUST BE
C             INITIALIZED BY CALLING SUBROUTINE CFFTI(N,WSAVE) AND A
C             DIFFERENT WSAVE ARRAY MUST BE USED FOR EACH DIFFERENT
C             VALUE OF N. THIS INITIALIZATION DOES NOT HAVE TO BE
C             REPEATED SO LONG AS N REMAINS UNCHANGED THUS SUBSEQUENT
C             TRANSFORMS CAN BE OBTAINED FASTER THAN THE FIRST.
C             THE SAME WSAVE ARRAY CAN BE USED BY CFFTF AND CFFTB.
C
C     OUTPUT PARAMETERS
C
C     C      FOR J=1,...,N
C
C                C(J)=THE SUM FROM K=1,...,N OF
C
C                      C(K)*EXP(-I*J*K*2*PI/N)
C
C                            WHERE I=SQRT(-1)
C
C     WSAVE   CONTAINS INITIALIZATION CALCULATIONS WHICH MUST NOT BE
C             DESTROYED BETWEEN CALLS OF SUBROUTINE CFFTF OR CFFTB
C
C***REFERENCES
C***ROUTINES CALLED  CFFTF1
C***END PROLOGUE  CFFTF
      DIMENSION       C(*)       ,WSAVE(*)
C***FIRST EXECUTABLE STATEMENT  CFFTF

      IF (N .EQ. 1) RETURN
      IW1 = N+N+1
      IW2 = IW1+N+N
      CALL CFFTF1 (N,C,WSAVE,WSAVE(IW1),WSAVE(IW2))
      RETURN
      END

* -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+
*DECK CFFTF1
      SUBROUTINE CFFTF1 (N,C,CH,WA,IFAC)
C***BEGIN PROLOGUE  CFFTF1
C***REFER TO CFFTF
C***ROUTINES CALLED  PASSF,PASSF5,PASSF3,PASSF2,PASSF4
C***END PROLOGUE  CFFTF1
      DIMENSION       CH(*)      ,C(*)       ,WA(*)      ,IFAC(*)
C***FIRST EXECUTABLE STATEMENT  CFFTF1
      NF = IFAC(2)
      NA = 0
      L1 = 1
      IW = 1
      DO 116 K1=1,NF
         IP = IFAC(K1+2)
         L2 = IP*L1
         IDO = N/L2
         IDOT = IDO+IDO
         IDL1 = IDOT*L1
         IF (IP .NE. 4) GO TO 103
         IX2 = IW+IDOT
         IX3 = IX2+IDOT
         IF (NA .NE. 0) GO TO 101
         CALL PASSF4 (IDOT,L1,C,CH,WA(IW),WA(IX2),WA(IX3))
         GO TO 102
  101    CALL PASSF4 (IDOT,L1,CH,C,WA(IW),WA(IX2),WA(IX3))
  102    NA = 1-NA
         GO TO 115
  103    IF (IP .NE. 2) GO TO 106
         IF (NA .NE. 0) GO TO 104
         CALL PASSF2 (IDOT,L1,C,CH,WA(IW))
         GO TO 105
  104    CALL PASSF2 (IDOT,L1,CH,C,WA(IW))
  105    NA = 1-NA
         GO TO 115
  106    IF (IP .NE. 3) GO TO 109
         IX2 = IW+IDOT
         IF (NA .NE. 0) GO TO 107
         CALL PASSF3 (IDOT,L1,C,CH,WA(IW),WA(IX2))
         GO TO 108
  107    CALL PASSF3 (IDOT,L1,CH,C,WA(IW),WA(IX2))
  108    NA = 1-NA
         GO TO 115
  109    IF (IP .NE. 5) GO TO 112
         IX2 = IW+IDOT
         IX3 = IX2+IDOT
         IX4 = IX3+IDOT
         IF (NA .NE. 0) GO TO 110
         CALL PASSF5 (IDOT,L1,C,CH,WA(IW),WA(IX2),WA(IX3),WA(IX4))
         GO TO 111
  110    CALL PASSF5 (IDOT,L1,CH,C,WA(IW),WA(IX2),WA(IX3),WA(IX4))
  111    NA = 1-NA
         GO TO 115
  112    IF (NA .NE. 0) GO TO 113
         CALL PASSF (NAC,IDOT,IP,L1,IDL1,C,C,C,CH,CH,WA(IW))
         GO TO 114
  113    CALL PASSF (NAC,IDOT,IP,L1,IDL1,CH,CH,CH,C,C,WA(IW))
  114    IF (NAC .NE. 0) NA = 1-NA
  115    L1 = L2
         IW = IW+(IP-1)*IDOT
  116 CONTINUE
      IF (NA .EQ. 0) RETURN
      N2 = N+N
      DO 117 I=1,N2
         C(I) = CH(I)
  117 CONTINUE
      RETURN
      END

* -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+
*DECK CFFTI
      SUBROUTINE CFFTI (N,WSAVE)
C***BEGIN PROLOGUE  CFFTI
C***REVISION DATE  811015   (YYMMDD)
C***CATEGORY NO.  D6
C***KEYWORDS FFT,FAST FOURIER TRANSFORM,FOURIER TRANSFORM,COMPLEX
C***DATE WRITTEN  FEBRUARY 1978
C***AUTHOR  SWARZTRAUBER P.N. (NCAR)
C***PURPOSE
C   INITIALIZE FOR CFFTF AND CFFTB
C***DESCRIPTION
C     *****************************************************************
C
C     SUBROUTINE CFFTI(N,WSAVE)
C
C     *****************************************************************
C
C     SUBROUTINE CFFTI INITIALIZES THE ARRAY WSAVE WHICH IS USED IN
C     BOTH CFFTF AND CFFTB. THE PRIME FACTORIZATION OF N TOGETHER WITH
C     A TABULATION OF THE TRIGONOMETRIC FUNCTIONS ARE COMPUTED AND
C     STORED IN WSAVE.
C
C     INPUT PARAMETER
C
C     N       THE LENGTH OF THE SEQUENCE TO BE TRANSFORMED
C
C     OUTPUT PARAMETER
C
C     WSAVE   A WORK ARRAY WHICH MUST BE DIMENSIONED AT LEAST 4*N+15
C             THE SAME WORK ARRAY CAN BE USED FOR BOTH CFFTF AND CFFTB
C             AS LONG AS N REMAINS UNCHANGED. DIFFERENT WSAVE ARRAYS
C             ARE REQUIRED FOR DIFFERENT VALUES OF N. THE CONTENTS OF
C             WSAVE MUST NOT BE CHANGED BETWEEN CALLS OF CFFTF OR CFFTB
C
C***REFERENCES
C***ROUTINES CALLED  CFFTI1
C***END PROLOGUE  CFFTI
      DIMENSION       WSAVE(*)
C***FIRST EXECUTABLE STATEMENT  CFFTI
      IF (N .EQ. 1) RETURN
      IW1 = N+N+1
      IW2 = IW1+N+N
      CALL CFFTI1 (N,WSAVE(IW1),WSAVE(IW2))
      RETURN
      END

* -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+
*DECK CFFTI1
      SUBROUTINE CFFTI1 (N,WA,IFAC)
C***BEGIN PROLOGUE  CFFTI1
C***REFER TO CFFTI
C***ROUTINES CALLED  (NONE)
C***END PROLOGUE  CFFTI1
      DIMENSION       WA(*)      ,IFAC(*)    ,NTRYH(4)
      DATA NTRYH(1),NTRYH(2),NTRYH(3),NTRYH(4)/3,4,2,5/
C***FIRST EXECUTABLE STATEMENT  CFFTI1
      NL = N
      NF = 0
      J = 0
  101 J = J+1
      IF (J-4) 102,102,103
  102 NTRY = NTRYH(J)
      GO TO 104
  103 NTRY = NTRY+2
  104 NQ = NL/NTRY
      NR = NL-NTRY*NQ
      IF (NR) 101,105,101
  105 NF = NF+1
      IFAC(NF+2) = NTRY
      NL = NQ
      IF (NTRY .NE. 2) GO TO 107
      IF (NF .EQ. 1) GO TO 107
      DO 106 I=2,NF
         IB = NF-I+2
         IFAC(IB+2) = IFAC(IB+1)
  106 CONTINUE
      IFAC(3) = 2
  107 IF (NL .NE. 1) GO TO 104
      IFAC(1) = N
      IFAC(2) = NF
      TPI = 6.28318530717959
      ARGH = TPI/FLOAT(N)
      I = 2
      L1 = 1
      DO 110 K1=1,NF
         IP = IFAC(K1+2)
         LD = 0
         L2 = L1*IP
         IDO = N/L2
         IDOT = IDO+IDO+2
         IPM = IP-1
         WLDC = 1.
         WLDS = 0.
         ARG = FLOAT(L1)*ARGH
         DL1C = COS(ARG)
         DL1S = SIN(ARG)
         DO 109 J=1,IPM
            I1 = I
            WA(I-1) = 1.
            WA(I) = 0.
            LD = LD+L1
            WLDCH = WLDC
            WLDC = DL1C*WLDC-DL1S*WLDS
            WLDS = DL1S*WLDCH+DL1C*WLDS
            DC = WLDC
            DS = WLDS
            DO 108 II=4,IDOT,2
               I = I+2
               WA(I-1) = DC*WA(I-3)-DS*WA(I-2)
               WA(I) = DS*WA(I-3)+DC*WA(I-2)
  108       CONTINUE
            IF (IP .LE. 5) GO TO 109
            WA(I1-1) = WA(I-1)
            WA(I1) = WA(I)
  109    CONTINUE
         L1 = L2
  110 CONTINUE
      RETURN
      END

* -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+
*DECK EZFFTB
      SUBROUTINE EZFFTB (N,R,AZERO,A,B,WSAVE)
C***BEGIN PROLOGUE  EZFFTB
C***REVISION DATE  811015   (YYMMDD)
C***CATEGORY NO.  D6
C***KEYWORDS FFT,FAST FOURIER TRANSFORM,INVERSE,INVERSE FFT
C***DATE WRITTEN  FEBRUARY 1978
C***AUTHOR  SWARZTRAUBER P.N. (NCAR)
C***PURPOSE
C SIMPLIFIED VERSION OF INVERSE FFT OF A REAL PERIODIC SEQUENCE
C***DESCRIPTION
C     *****************************************************************
C
C     SUBROUTINE EZFFTB(N,R,AZERO,A,B,WSAVE)
C
C     *****************************************************************
C
C     SUBROUTINE EZFFTB COMPUTES A REAL PERODIC SEQUENCE FROM ITS
C     FOURIER COEFFICIENTS (FOURIER SYNTHESIS). THE TRANSFORM IS
C     DEFINED BELOW AT OUTPUT PARAMETER R. EZFFTB IS A SIMPLIFIED
C     VERSION OF RFFTB. IT IS NOT AS FAST AS RFFTB SINCE SCALING AND
C     INITIALIZATION ARE COMPUTED FOR EACH TRANSFORM. THE REPEATED
C     INITIALIZATION CAN BE SUPPRESSED BY REMOVEING THE STATMENT
C     ( CALL EZFFTI(N,WSAVE) ) FROM BOTH EZFFTF AND EZFFTB AND INSERTING
C     IT AT THE APPROPRIATE PLACE IN YOUR PROGRAM.
C
C     INPUT PARAMETERS
C
C     N       THE LENGTH OF THE ARRAY R. EZFFTB IS ABOUT TWICE AS FAST
C             FOR EVEN N AS IT IS FOR ODD N. ALSO EZFFTB IS MORE
C             EFFICIENT WHEN N IS A PRODUCT OF SMALL PRIMES.
C
C     AZERO   THE CONSTANT FOURIER COEFFICIENT
C
C     A,B     ARRAYS WHICH CONTAIN THE REMAINING FOURIER COEFFICIENTS
C             THESE ARRAYS ARE NOT DESTROYED.
C
C             THE LENGTH OF THESE ARRAYS DEPENDS ON WHETHER N IS EVEN OR
C             ODD.
C
C             IF N IS EVEN N/2    LOCATIONS ARE REQUIRED
C             IF N IS ODD (N-1)/2 LOCATIONS ARE REQUIRED
C
C     WSAVE   A WORK ARRAY WHOSE LENGTH DEPENDS ON WHETHER N IS EVEN OR
C             ODD.
C
C                  IF N IS EVEN 3.5*N+15 LOCATIONS ARE REQUIRED
C                  IF N IS ODD  6*N+15   LOCATIONS ARE REQUIRED
C
C
C     OUTPUT PARAMETERS
C
C     R       IF N IS EVEN DEFINE KMAX=N/2
C             IF N IS ODD  DEFINE KMAX=(N-1)/2
C
C             THEN FOR I=1,...,N
C
C                  R(I)=AZERO PLUS THE SUM FROM K=1 TO K=KMAX OF
C
C                  A(K)*COS(K*(I-1)*2*PI/N)+B(K)*SIN(K*(I-1)*2*PI/N)
C
C     ********************* COMPLEX NOTATION **************************
C
C             FOR J=1,...,N
C
C             R(J) EQUALS THE SUM FROM K=-KMAX TO K=KMAX OF
C
C                  C(K)*EXP(I*K*(J-1)*2*PI/N)
C
C             WHERE
C
C                  C(K) = .5*CMPLX(A(K),-B(K))   FOR K=1,...,KMAX
C
C                  C(-K) = CONJG(C(K))
C
C                  C(0) = AZERO
C
C                       AND I=SQRT(-1)
C
C     *************** AMPLITUDE - PHASE NOTATION **********************
C
C             FOR I=1,...,N
C
C             R(I) EQUALS AZERO PLUS THE SUM FROM K=1 TO K=KMAX OF
C
C                  ALPHA(K)*COS(K*(I-1)*2*PI/N+BETA(K))
C
C             WHERE
C
C                  ALPHA(K) = SQRT(A(K)*A(K)+B(K)*B(K))
C
C                  COS(BETA(K))=A(K)/ALPHA(K)
C
C                  SIN(BETA(K))=-B(K)/ALPHA(K)
C
C***REFERENCES
C***ROUTINES CALLED  CFFTB,RFFTB,EZFFTI
C***END PROLOGUE  EZFFTB
      DIMENSION       R(*)       ,A(*)       ,B(*)       ,WSAVE(*)
C***FIRST EXECUTABLE STATEMENT  EZFFTB
      IF (N .GT. 1) GO TO 101
      R(1) = AZERO
      RETURN
  101 IF (N .GT. 2) GO TO 102
      R(1) = AZERO+A(1)
      R(2) = AZERO-A(1)
      RETURN
  102 NS2 = N/2
C
C     TO SUPRESS REPEATED INITIALIZATION REMOVE THE FOLLOWING STATMENT
C     ( CALL EZFFTI(N,WSAVE)) FROM BOTH THIS PROGRAM AND EZFFTF AND
C     INSERT IT AT THE APPROPRIATE PLACE IN YOUR PROGRAM.
C
      CALL EZFFTI (N,WSAVE)
C
      MODN = N-NS2-NS2
      IF (MODN .NE. 0) GO TO 104
      IW1 = N+1
      NS2M = NS2-1
      DO 103 I=1,NS2M
         R(2*I+1) = .5*A(I)
         R(2*I+2) = .5*B(I)
  103 CONTINUE
      R(1) = AZERO
      R(2) = A(NS2)
      CALL RFFTB (N,R,WSAVE(IW1))
      RETURN
  104 IW1 = N+N+1
      NM1S2 = (N-1)/2
      DO 105 I=1,NM1S2
         IC = N-I
         WSAVE(2*I+1) = A(I)
         WSAVE(2*I+2) = -B(I)
         WSAVE(2*IC+1) = A(I)
         WSAVE(2*IC+2) = B(I)
  105 CONTINUE
      WSAVE(1) = AZERO+AZERO
      WSAVE(2) = 0.
      CALL CFFTB (N,WSAVE,WSAVE(IW1))
      DO 106 I=1,N
         R(I) = .5*WSAVE(2*I-1)
  106 CONTINUE
      RETURN
      END

* -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+
*DECK EZFFTF
      SUBROUTINE EZFFTF (N,R,AZERO,A,B,WSAVE)
C***BEGIN PROLOGUE  EZFFTF
C***REVISION DATE  811015   (YYMMDD)
C***CATEGORY NO.  D6
C***KEYWORDS FFT,FAST FOURIER TRANSFORM,FOURIER TRANSFORM
C***DATE WRITTEN  FEBRUARY 1978
C***AUTHOR  SWARZTRAUBER P.N. (NCAR)
C***PURPOSE
C  FORWARD FFT OF A REAL PERIODIC SEQUENCE (SIMPLIFED VERSION)
C***DESCRIPTION
C     *****************************************************************
C
C     SUBROUTINE EZFFTF(N,R,AZERO,A,B,WSAVE)
C
C     *****************************************************************
C
C     SUBROUTINE EZFFTF COMPUTES THE FOURIER COEFFICIENTS OF A REAL
C     PERODIC SEQUENCE (FOURIER ANALYSIS). THE TRANSFORM IS DEFINED
C     BELOW AT OUTPUT PARAMETERS AZERO,A AND B. EZFFTF IS A SIMPLIFIED
C     VERSION OF RFFTF. IT IS NOT AS FAST AS RFFTF SINCE SCALING
C     AND INITIALIZATION ARE COMPUTED FOR EACH TRANSFORM. THE REPEATED
C     INITIALIZATION CAN BE SUPPRESSED BY REMOVEING THE STATMENT
C     ( CALL EZFFTI(N,WSAVE) ) FROM BOTH EZFFTF AND EZFFTB AND INSERTING
C     IT AT THE APPROPRIATE PLACE IN YOUR PROGRAM.
C
C     INPUT PARAMETERS
C
C     N       THE LENGTH OF THE ARRAY R. EZFFTF IS ABOUT TWICE AS FAST
C             FOR EVEN N AS IT IS FOR ODD N. ALSO EZFFTF IS MORE
C             EFFICIENT WHEN N IS A PRODUCT OF SMALL PRIMES.
C
C     R       A REAL ARRAY OF LENGTH N WHICH CONTAINS THE SEQUENCE
C             TO BE TRANSFORMED. R IS NOT DESTROYED.
C
C     WSAVE   A WORK ARRAY WHOSE LENGTH DEPENDS ON WHETHER N IS EVEN OR
C             ODD.
C
C                  IF N IS EVEN 3.5*N+15 LOCATIONS ARE REQUIRED
C                  IF N IS ODD  6*N+15   LOCATIONS ARE REQUIRED
C
C
C     OUTPUT PARAMETERS
C
C     AZERO   THE SUM FROM I=1 TO I=N OF R(I)/N
C
C     A,B     FOR N EVEN B(N/2)=0. AND A(N/2) IS THE SUM FROM I=1 TO
C             I=N OF (-1)**(I-1)*R(I)/N
C
C             FOR N EVEN DEFINE KMAX=N/2-1
C             FOR N ODD  DEFINE KMAX=(N-1)/2
C
C             THEN FOR  K=1,...,KMAX
C
C                  A(K) EQUALS THE SUM FROM I=1 TO I=N OF
C
C                       2./N*R(I)*COS(K*(I-1)*2*PI/N)
C
C                  B(K) EQUALS THE SUM FROM I=1 TO I=N OF
C
C                       2./N*R(I)*SIN(K*(I-1)*2*PI/N)
C
C
C***REFERENCES
C***ROUTINES CALLED  CFFTF,RFFTF,EZFFTI
C***END PROLOGUE  EZFFTF
      DIMENSION       R(*)       ,A(*)       ,B(*)       ,WSAVE(*)
C***FIRST EXECUTABLE STATEMENT  EZFFTF
      IF (N .GT. 1) GO TO 101
      AZERO = R(1)
      RETURN
  101 IF (N .GT. 2) GO TO 102
      AZERO = .5*(R(1)+R(2))
      A(1) = .5*(R(1)-R(2))
      RETURN
  102 NS2 = N/2
C
C     TO SUPRESS REPEATED INITIALIZATION REMOVE THE FOLLOWING STATMENT
C     ( CALL EZFFTI(N,WSAVE)) FROM BOTH THIS PROGRAM AND EZFFTB AND
C     INSERT IT AT THE APPROPRIATE PLACE IN YOUR PROGRAM.
C
      CALL EZFFTI (N,WSAVE)
C
      MODN = N-NS2-NS2
      IF (MODN .NE. 0) GO TO 105
      IW1 = N+1
      DO 103 I=1,N
         WSAVE(I) = R(I)
  103 CONTINUE
      CALL RFFTF (N,WSAVE,WSAVE(IW1))
      CF = 1./FLOAT(N)
      AZERO = .5*CF*WSAVE(1)
      A(NS2) = .5*CF*WSAVE(2)
      B(NS2) = 0.
      NS2M = NS2-1
      DO 104 I=1,NS2M
         A(I) = CF*WSAVE(2*I+1)
         B(I) = CF*WSAVE(2*I+2)
  104 CONTINUE
      RETURN
  105 IW1 = N+N+1
      DO 106 I=1,N
         WSAVE(2*I-1) = R(I)
         WSAVE(2*I) = 0.
  106 CONTINUE
      CALL CFFTF (N,WSAVE,WSAVE(IW1))
      CF = 2./FLOAT(N)
      AZERO = .5*CF*WSAVE(1)
      NM1S2 = (N-1)/2
      DO 107 I=1,NM1S2
         A(I) = CF*WSAVE(2*I+1)
         B(I) = -CF*WSAVE(2*I+2)
  107 CONTINUE
      RETURN
      END

* -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+
*DECK EZFFTI
      SUBROUTINE EZFFTI(N,WSAVE)
C***BEGIN PROLOGUE  EZFFTI
C***REVISION DATE  811015   (YYMMDD)
C***CATEGORY NO.  D6
C***KEYWORDS FFT,FAST FOURIER TRANSFORM,FOURIER TRANSFORM
C***DATE WRITTEN  FEBRUARY 1978
C***AUTHOR  SWARZTRAUBER P.N. (NCAR)
C***PURPOSE
C   INITIALIZE FOR EZFFTF AND EZFFTB
C***DESCRIPTION
C     *****************************************************************
C
C     SUBROUTINE EZFFTI(N,WSAVE)
C
C     *****************************************************************
C
C     SUBROUTINE EZFFTI INITIALIZES THE ARRAY WSAVE WHICH IS USED IN
C     BOTH EZFFTF AND EZFFTB. THE PRIME FACTORIZATION OF N TOGETHER WITH
C     A TABULATION OF THE TRIGONOMETRIC FUNCTIONS ARE COMPUTED AND
C     STORED IN WSAVE.
C
C     INPUT PARAMETER
C
C     N       THE LENGTH OF THE SEQUENCE TO BE TRANSFORMED.
C
C     OUTPUT PARAMETER
C
C     WSAVE   A WORK ARRAY WHICH MUST BE DIMENSIONED AT LEAST 4*N+15
C             THE SAME WORK ARRAY CAN BE USED FOR BOTH EZFFTF AND EZFFTB
C             AS LONG AS N REMAINS UNCHANGED. DIFFERENT WSAVE ARRAYS
C             ARE REQUIRED FOR DIFFERENT VALUES OF N. THE CONTENTS OF
C             WSAVE MUST NOT BE CHANGED BETWEEN CALLS OF EZFFTF OR EZFFT
C
C***REFERENCES
C***ROUTINES CALLED  CFFTI,RFFTI
C***END PROLOGUE  EZFFTI
      DIMENSION       WSAVE(*)
C***FIRST EXECUTABLE STATEMENT  EZFFTI
      IF (N .LE. 2) RETURN
      NS2 = N/2
      MODN = N-NS2-NS2
      IF (MODN .NE. 0) GO TO 101
      IW1 = N+1
      CALL RFFTI (N,WSAVE(IW1))
      RETURN
  101 IW1 = N+N+1
      CALL CFFTI (N,WSAVE(IW1))
      RETURN
      END

* -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+
*DECK FFTDOC
      SUBROUTINE FFTDOC
C***BEGIN PROLOGUE  FFTDOC
C***DATE WRITTEN   780201   (YYMMDD)
C***REVISION DATE  811015   (YYMMDD)
C***PURPOSE  DOCUMENTATION FOR FFT PACKAGE
C***DESCRIPTION
C     * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
C
C                       VERSION 2  FEBRUARY 1978
C
C          A PACKAGE OF FORTRAN SUBPROGRAMS FOR THE FAST FOURIER
C           TRANSFORM OF PERIODIC AND OTHER SYMMETRIC SEQUENCES
C
C                              BY
C
C                       PAUL N SWARZTRAUBER
C
C       NATIONAL CENTER FOR ATMOSPHERIC RESEARCH  BOULDER,COLORADO 8030
C
C        WHICH IS SPONSORED BY THE NATIONAL SCIENCE FOUNDATION
C
C     * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
C
C
C     THIS PACKAGE CONSISTS OF PROGRAMS WHICH PERFORM FAST FOURIER
C     TRANSFORMS FOR BOTH COMPLEX AND REAL PERIODIC SEQUENCES AND
C     CERTAIN OTHER SYMMETRIC SEQUENCES THAT ARE LISTED BELOW.
C
C     1.   RFFTI     INITIALIZE  RFFTF AND RFFTB
C     2.   RFFTF     FORWARD TRANSFORM OF A REAL PERIODIC SEQUENCE
C     3.   RFFTB     BACKWARD TRANSFORM OF A REAL COEFFICIENT ARRAY
C
C     4.   EZFFTF    A SIMPLIFIED REAL PERIODIC FORWARD TRANSFORM
C     5.   EZFFTB    A SIMPLIFIED REAL PERIODIC BACKWARD TRANSFORM
C
C     6.   SINTI     INITIALIZE SINT
C     7.   SINT      SINE TRANSFORM OF A REAL ODD SEQUENCE
C
C     8.   COSTI     INITIALIZE COST
C     9.   COST      COSINE TRANSFORM OF A REAL EVEN SEQUENCE
C
C     10.  SINQI     INITIALIZE SINQF AND SINQB
C     11.  SINQF     FORWARD SINE TRANSFORM WITH ODD WAVE NUMBERS
C     12.  SINQB     UNNORMALIZED INVERSE OF SINQF
C
C     13.  COSQI     INITIALIZE COSQF AND COSQB
C     14.  COSQF     FORWARD COSINE TRANSFORM WITH ODD WAVE NUMBERS
C     15.  COSQB     UNNORMALIZED INVERSE OF COSQF
C
C     16.  CFFTI     INITIALIZE CFFTF AND CFFTB
C     17.  CFFTF     FORWARD TRANSFORM OF A COMPLEX PERIODIC SEQUENCE
C     18.  CFFTB     UNNORMALIZED INVERSE OF CFFTF
C
C***ROUTINES CALLED  (NONE)
C***END PROLOGUE  FFTDOC
C***FIRST EXECUTABLE STATEMENT  FFTDOC
       RETURN
       END

* -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+
*DECK PASSB
      SUBROUTINE PASSB (NAC,IDO,IP,L1,IDL1,CC,C1,C2,CH,CH2,WA)
C***BEGIN PROLOGUE  PASSB
C***REFER TO CFFTB
C***ROUTINES CALLED  (NONE)
C***END PROLOGUE  PASSB
      DIMENSION       CH(IDO,L1,IP)          ,CC(IDO,IP,L1)          ,
     1                C1(IDO,L1,IP)          ,WA(*)      ,C2(IDL1,IP),
     2                CH2(IDL1,IP)
C***FIRST EXECUTABLE STATEMENT  PASSB
      IDOT = IDO/2
      NT = IP*IDL1
      IPP2 = IP+2
      IPPH = (IP+1)/2
      IDP = IP*IDO
C
      IF (IDO .LT. L1) GO TO 106
      DO 103 J=2,IPPH
         JC = IPP2-J
         DO 102 K=1,L1
            DO 101 I=1,IDO
               CH(I,K,J) = CC(I,J,K)+CC(I,JC,K)
               CH(I,K,JC) = CC(I,J,K)-CC(I,JC,K)
  101       CONTINUE
  102    CONTINUE
  103 CONTINUE
      DO 105 K=1,L1
         DO 104 I=1,IDO
            CH(I,K,1) = CC(I,1,K)
  104    CONTINUE
  105 CONTINUE
      GO TO 112
  106 DO 109 J=2,IPPH
         JC = IPP2-J
         DO 108 I=1,IDO
            DO 107 K=1,L1
               CH(I,K,J) = CC(I,J,K)+CC(I,JC,K)
               CH(I,K,JC) = CC(I,J,K)-CC(I,JC,K)
  107       CONTINUE
  108    CONTINUE
  109 CONTINUE
      DO 111 I=1,IDO
         DO 110 K=1,L1
            CH(I,K,1) = CC(I,1,K)
  110    CONTINUE
  111 CONTINUE
  112 IDL = 2-IDO
      INC = 0
      DO 116 L=2,IPPH
         LC = IPP2-L
         IDL = IDL+IDO
         DO 113 IK=1,IDL1
            C2(IK,L) = CH2(IK,1)+WA(IDL-1)*CH2(IK,2)
            C2(IK,LC) = WA(IDL)*CH2(IK,IP)
  113    CONTINUE
         IDLJ = IDL
         INC = INC+IDO
         DO 115 J=3,IPPH
            JC = IPP2-J
            IDLJ = IDLJ+INC
            IF (IDLJ .GT. IDP) IDLJ = IDLJ-IDP
            WAR = WA(IDLJ-1)
            WAI = WA(IDLJ)
            DO 114 IK=1,IDL1
               C2(IK,L) = C2(IK,L)+WAR*CH2(IK,J)
               C2(IK,LC) = C2(IK,LC)+WAI*CH2(IK,JC)
  114       CONTINUE
  115    CONTINUE
  116 CONTINUE
      DO 118 J=2,IPPH
         DO 117 IK=1,IDL1
            CH2(IK,1) = CH2(IK,1)+CH2(IK,J)
  117    CONTINUE
  118 CONTINUE
      DO 120 J=2,IPPH
         JC = IPP2-J
         DO 119 IK=2,IDL1,2
            CH2(IK-1,J) = C2(IK-1,J)-C2(IK,JC)
            CH2(IK-1,JC) = C2(IK-1,J)+C2(IK,JC)
            CH2(IK,J) = C2(IK,J)+C2(IK-1,JC)
            CH2(IK,JC) = C2(IK,J)-C2(IK-1,JC)
  119    CONTINUE
  120 CONTINUE
      NAC = 1
      IF (IDO .EQ. 2) RETURN
      NAC = 0
      DO 121 IK=1,IDL1
         C2(IK,1) = CH2(IK,1)
  121 CONTINUE
      DO 123 J=2,IP
         DO 122 K=1,L1
            C1(1,K,J) = CH(1,K,J)
            C1(2,K,J) = CH(2,K,J)
  122    CONTINUE
  123 CONTINUE
      IF (IDOT .GT. L1) GO TO 127
      IDIJ = 0
      DO 126 J=2,IP
         IDIJ = IDIJ+2
         DO 125 I=4,IDO,2
            IDIJ = IDIJ+2
            DO 124 K=1,L1
               C1(I-1,K,J) = WA(IDIJ-1)*CH(I-1,K,J)-WA(IDIJ)*CH(I,K,J)
               C1(I,K,J) = WA(IDIJ-1)*CH(I,K,J)+WA(IDIJ)*CH(I-1,K,J)
  124       CONTINUE
  125    CONTINUE
  126 CONTINUE
      RETURN
  127 IDJ = 2-IDO
      DO 130 J=2,IP
         IDJ = IDJ+IDO
         DO 129 K=1,L1
            IDIJ = IDJ
            DO 128 I=4,IDO,2
               IDIJ = IDIJ+2
               C1(I-1,K,J) = WA(IDIJ-1)*CH(I-1,K,J)-WA(IDIJ)*CH(I,K,J)
               C1(I,K,J) = WA(IDIJ-1)*CH(I,K,J)+WA(IDIJ)*CH(I-1,K,J)
  128       CONTINUE
  129    CONTINUE
  130 CONTINUE
      RETURN
      END

* -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+ -+

*DECK PASSB2
      SUBROUTINE PASSB2 (IDO,L1,CC,CH,WA1)
C***BEGIN PROLOGUE  PASSB2
C***REFER TO CFFTB
C***ROUTINES CALLED  (NONE)
C***END PROLOGUE  PASSB2
      DIMENSION       CC(IDO,2,L1)           ,CH(IDO,L1,2)           ,
     1                WA1(*)
C***FIRST EXECUTABLE STATEMENT  PASSB2
      IF (IDO .GT. 2) GO TO 102
      DO 101 K=1,L1
         CH(1,K,1) = CC(1,1,K)+CC(1,2,K)
         CH(1,K,2) = CC(1,1,K)-CC(1,2,K)
         CH(2,K,1) = CC(2,1,K)+CC(2,2,K)
         CH(2,K,2) = CC(2,1,K)-CC(2,2,K)
  101 CONTINUE
      RETURN
  102 DO 104 K=1,L1
         DO 103 I=2,IDO,2
            CH(I-1,K,1) = CC(I-1,1,K)+CC(I-1,2,K)
            TR2 = CC(I-1,1,K)-CC(I-1,2,K)
            CH(I,K,1) = CC(I,1,K)+CC(I,2,K)
            TI2 = CC(I,1,K)-CC(I,2,K)
            CH(I,K,2) = WA1(I-1)*TI2+WA1(I)*TR2
            CH(I-1,K,2) = WA1(I-1)*TR2-WA1(I)*TI2
  103    CONTINUE
  104 CONTINUE
      RETURN
      END

* --+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+

*DECK PASSB3
      SUBROUTINE PASSB3 (IDO,L1,CC,CH,WA1,WA2)
C***BEGIN PROLOGUE  PASSB3
C***REFER TO CFFTB
C***ROUTINES CALLED  (NONE)
C***END PROLOGUE  PASSB3
      DIMENSION       CC(IDO,3,L1)           ,CH(IDO,L1,3)           ,
     1                WA1(*)     ,WA2(*)
      DATA TAUR,TAUI /-.5,.866025403784439/
C***FIRST EXECUTABLE STATEMENT  PASSB3
      IF (IDO .NE. 2) GO TO 102
      DO 101 K=1,L1
         TR2 = CC(1,2,K)+CC(1,3,K)
         CR2 = CC(1,1,K)+TAUR*TR2
         CH(1,K,1) = CC(1,1,K)+TR2
         TI2 = CC(2,2,K)+CC(2,3,K)
         CI2 = CC(2,1,K)+TAUR*TI2
         CH(2,K,1) = CC(2,1,K)+TI2
         CR3 = TAUI*(CC(1,2,K)-CC(1,3,K))
         CI3 = TAUI*(CC(2,2,K)-CC(2,3,K))
         CH(1,K,2) = CR2-CI3
         CH(1,K,3) = CR2+CI3
         CH(2,K,2) = CI2+CR3
         CH(2,K,3) = CI2-CR3
  101 CONTINUE
      RETURN
  102 DO 104 K=1,L1
         DO 103 I=2,IDO,2
            TR2 = CC(I-1,2,K)+CC(I-1,3,K)
            CR2 = CC(I-1,1,K)+TAUR*TR2
            CH(I-1,K,1) = CC(I-1,1,K)+TR2
            TI2 = CC(I,2,K)+CC(I,3,K)
            CI2 = CC(I,1,K)+TAUR*TI2
            CH(I,K,1) = CC(I,1,K)+TI2
            CR3 = TAUI*(CC(I-1,2,K)-CC(I-1,3,K))
            CI3 = TAUI*(CC(I,2,K)-CC(I,3,K))
            DR2 = CR2-CI3
            DR3 = CR2+CI3
            DI2 = CI2+CR3
            DI3 = CI2-CR3
            CH(I,K,2) = WA1(I-1)*DI2+WA1(I)*DR2
            CH(I-1,K,2) = WA1(I-1)*DR2-WA1(I)*DI2
            CH(I,K,3) = WA2(I-1)*DI3+WA2(I)*DR3
            CH(I-1,K,3) = WA2(I-1)*DR3-WA2(I)*DI3
  103    CONTINUE
  104 CONTINUE
      RETURN
      END

* --+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+

*DECK PASSB4
      SUBROUTINE PASSB4 (IDO,L1,CC,CH,WA1,WA2,WA3)
C***BEGIN PROLOGUE  PASSB4
C***REFER TO CFFTB
C***ROUTINES CALLED (NONE)
C***END PROLOGUE  PASSB4
      DIMENSION       CC(IDO,4,L1)           ,CH(IDO,L1,4)           ,
     1                WA1(*)     ,WA2(*)     ,WA3(*)
C***FIRST EXECUTABLE STATEMENT  PASSB4
      IF (IDO .NE. 2) GO TO 102
      DO 101 K=1,L1
         TI1 = CC(2,1,K)-CC(2,3,K)
         TI2 = CC(2,1,K)+CC(2,3,K)
         TR4 = CC(2,4,K)-CC(2,2,K)
         TI3 = CC(2,2,K)+CC(2,4,K)
         TR1 = CC(1,1,K)-CC(1,3,K)
         TR2 = CC(1,1,K)+CC(1,3,K)
         TI4 = CC(1,2,K)-CC(1,4,K)
         TR3 = CC(1,2,K)+CC(1,4,K)
         CH(1,K,1) = TR2+TR3
         CH(1,K,3) = TR2-TR3
         CH(2,K,1) = TI2+TI3
         CH(2,K,3) = TI2-TI3
         CH(1,K,2) = TR1+TR4
         CH(1,K,4) = TR1-TR4
         CH(2,K,2) = TI1+TI4
         CH(2,K,4) = TI1-TI4
  101 CONTINUE
      RETURN
  102 DO 104 K=1,L1
         DO 103 I=2,IDO,2
            TI1 = CC(I,1,K)-CC(I,3,K)
            TI2 = CC(I,1,K)+CC(I,3,K)
            TI3 = CC(I,2,K)+CC(I,4,K)
            TR4 = CC(I,4,K)-CC(I,2,K)
            TR1 = CC(I-1,1,K)-CC(I-1,3,K)
            TR2 = CC(I-1,1,K)+CC(I-1,3,K)
            TI4 = CC(I-1,2,K)-CC(I-1,4,K)
            TR3 = CC(I-1,2,K)+CC(I-1,4,K)
            CH(I-1,K,1) = TR2+TR3
            CR3 = TR2-TR3
            CH(I,K,1) = TI2+TI3
            CI3 = TI2-TI3
            CR2 = TR1+TR4
            CR4 = TR1-TR4
            CI2 = TI1+TI4
            CI4 = TI1-TI4
            CH(I-1,K,2) = WA1(I-1)*CR2-WA1(I)*CI2
            CH(I,K,2) = WA1(I-1)*CI2+WA1(I)*CR2
            CH(I-1,K,3) = WA2(I-1)*CR3-WA2(I)*CI3
            CH(I,K,3) = WA2(I-1)*CI3+WA2(I)*CR3
            CH(I-1,K,4) = WA3(I-1)*CR4-WA3(I)*CI4
            CH(I,K,4) = WA3(I-1)*CI4+WA3(I)*CR4
  103    CONTINUE
  104 CONTINUE
      RETURN
      END

* --+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+

*DECK PASSB5
      SUBROUTINE PASSB5 (IDO,L1,CC,CH,WA1,WA2,WA3,WA4)
C***BEGIN PROLOGUE  PASSB5
C***REFER TO CFFTB
C***ROUTINES CALLED  (NONE)
C***END PROLOGUE  PASSB5
      DIMENSION       CC(IDO,5,L1)           ,CH(IDO,L1,5)           ,
     1                WA1(*)     ,WA2(*)     ,WA3(*)     ,WA4(*)
      DATA TR11,TI11,TR12,TI12 /.309016994374947,.951056516295154,
     1-.809016994374947,.587785252292473/
C***FIRST EXECUTABLE STATEMENT  PASSB5
      IF (IDO .NE. 2) GO TO 102
      DO 101 K=1,L1
         TI5 = CC(2,2,K)-CC(2,5,K)
         TI2 = CC(2,2,K)+CC(2,5,K)
         TI4 = CC(2,3,K)-CC(2,4,K)
         TI3 = CC(2,3,K)+CC(2,4,K)
         TR5 = CC(1,2,K)-CC(1,5,K)
         TR2 = CC(1,2,K)+CC(1,5,K)
         TR4 = CC(1,3,K)-CC(1,4,K)
         TR3 = CC(1,3,K)+CC(1,4,K)
         CH(1,K,1) = CC(1,1,K)+TR2+TR3
         CH(2,K,1) = CC(2,1,K)+TI2+TI3
         CR2 = CC(1,1,K)+TR11*TR2+TR12*TR3
         CI2 = CC(2,1,K)+TR11*TI2+TR12*TI3
         CR3 = CC(1,1,K)+TR12*TR2+TR11*TR3
         CI3 = CC(2,1,K)+TR12*TI2+TR11*TI3
         CR5 = TI11*TR5+TI12*TR4
         CI5 = TI11*TI5+TI12*TI4
         CR4 = TI12*TR5-TI11*TR4
         CI4 = TI12*TI5-TI11*TI4
         CH(1,K,2) = CR2-CI5
         CH(1,K,5) = CR2+CI5
         CH(2,K,2) = CI2+CR5
         CH(2,K,3) = CI3+CR4
         CH(1,K,3) = CR3-CI4
         CH(1,K,4) = CR3+CI4
         CH(2,K,4) = CI3-CR4
         CH(2,K,5) = CI2-CR5
  101 CONTINUE
      RETURN
  102 DO 104 K=1,L1
         DO 103 I=2,IDO,2
            TI5 = CC(I,2,K)-CC(I,5,K)
            TI2 = CC(I,2,K)+CC(I,5,K)
            TI4 = CC(I,3,K)-CC(I,4,K)
            TI3 = CC(I,3,K)+CC(I,4,K)
            TR5 = CC(I-1,2,K)-CC(I-1,5,K)
            TR2 = CC(I-1,2,K)+CC(I-1,5,K)
            TR4 = CC(I-1,3,K)-CC(I-1,4,K)
            TR3 = CC(I-1,3,K)+CC(I-1,4,K)
            CH(I-1,K,1) = CC(I-1,1,K)+TR2+TR3
            CH(I,K,1) = CC(I,1,K)+TI2+TI3
            CR2 = CC(I-1,1,K)+TR11*TR2+TR12*TR3
            CI2 = CC(I,1,K)+TR11*TI2+TR12*TI3
            CR3 = CC(I-1,1,K)+TR12*TR2+TR11*TR3
            CI3 = CC(I,1,K)+TR12*TI2+TR11*TI3
            CR5 = TI11*TR5+TI12*TR4
            CI5 = TI11*TI5+TI12*TI4
            CR4 = TI12*TR5-TI11*TR4
            CI4 = TI12*TI5-TI11*TI4
            DR3 = CR3-CI4
            DR4 = CR3+CI4
            DI3 = CI3+CR4
            DI4 = CI3-CR4
            DR5 = CR2+CI5
            DR2 = CR2-CI5
            DI5 = CI2-CR5
            DI2 = CI2+CR5
            CH(I-1,K,2) = WA1(I-1)*DR2-WA1(I)*DI2
            CH(I,K,2) = WA1(I-1)*DI2+WA1(I)*DR2
            CH(I-1,K,3) = WA2(I-1)*DR3-WA2(I)*DI3
            CH(I,K,3) = WA2(I-1)*DI3+WA2(I)*DR3
            CH(I-1,K,4) = WA3(I-1)*DR4-WA3(I)*DI4
            CH(I,K,4) = WA3(I-1)*DI4+WA3(I)*DR4
            CH(I-1,K,5) = WA4(I-1)*DR5-WA4(I)*DI5
            CH(I,K,5) = WA4(I-1)*DI5+WA4(I)*DR5
  103    CONTINUE
  104 CONTINUE
      RETURN
      END

* --+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+

*DECK PASSF
      SUBROUTINE PASSF (NAC,IDO,IP,L1,IDL1,CC,C1,C2,CH,CH2,WA)
C***BEGIN PROLOGUE  PASSF
C***REFER TO CFFTF
C***ROUTINES CALLED  (NONE)
C***END PROLOGUE  PASSF
      DIMENSION       CH(IDO,L1,IP)          ,CC(IDO,IP,L1)          ,
     1                C1(IDO,L1,IP)          ,WA(*)      ,C2(IDL1,IP),
     2                CH2(IDL1,IP)
C***FIRST EXECUTABLE STATEMENT  PASSF
      IDOT = IDO/2
      NT = IP*IDL1
      IPP2 = IP+2
      IPPH = (IP+1)/2
      IDP = IP*IDO
C
      IF (IDO .LT. L1) GO TO 106
      DO 103 J=2,IPPH
         JC = IPP2-J
         DO 102 K=1,L1
            DO 101 I=1,IDO
               CH(I,K,J) = CC(I,J,K)+CC(I,JC,K)
               CH(I,K,JC) = CC(I,J,K)-CC(I,JC,K)
  101       CONTINUE
  102    CONTINUE
  103 CONTINUE
      DO 105 K=1,L1
         DO 104 I=1,IDO
            CH(I,K,1) = CC(I,1,K)
  104    CONTINUE
  105 CONTINUE
      GO TO 112
  106 DO 109 J=2,IPPH
         JC = IPP2-J
         DO 108 I=1,IDO
            DO 107 K=1,L1
               CH(I,K,J) = CC(I,J,K)+CC(I,JC,K)
               CH(I,K,JC) = CC(I,J,K)-CC(I,JC,K)
  107       CONTINUE
  108    CONTINUE
  109 CONTINUE
      DO 111 I=1,IDO
         DO 110 K=1,L1
            CH(I,K,1) = CC(I,1,K)
  110    CONTINUE
  111 CONTINUE
  112 IDL = 2-IDO
      INC = 0
      DO 116 L=2,IPPH
         LC = IPP2-L
         IDL = IDL+IDO
         DO 113 IK=1,IDL1
            C2(IK,L) = CH2(IK,1)+WA(IDL-1)*CH2(IK,2)
            C2(IK,LC) = -WA(IDL)*CH2(IK,IP)
  113    CONTINUE
         IDLJ = IDL
         INC = INC+IDO
         DO 115 J=3,IPPH
            JC = IPP2-J
            IDLJ = IDLJ+INC
            IF (IDLJ .GT. IDP) IDLJ = IDLJ-IDP
            WAR = WA(IDLJ-1)
            WAI = WA(IDLJ)
            DO 114 IK=1,IDL1
               C2(IK,L) = C2(IK,L)+WAR*CH2(IK,J)
               C2(IK,LC) = C2(IK,LC)-WAI*CH2(IK,JC)
  114       CONTINUE
  115    CONTINUE
  116 CONTINUE
      DO 118 J=2,IPPH
         DO 117 IK=1,IDL1
            CH2(IK,1) = CH2(IK,1)+CH2(IK,J)
  117    CONTINUE
  118 CONTINUE
      DO 120 J=2,IPPH
         JC = IPP2-J
         DO 119 IK=2,IDL1,2
            CH2(IK-1,J) = C2(IK-1,J)-C2(IK,JC)
            CH2(IK-1,JC) = C2(IK-1,J)+C2(IK,JC)
            CH2(IK,J) = C2(IK,J)+C2(IK-1,JC)
            CH2(IK,JC) = C2(IK,J)-C2(IK-1,JC)
  119    CONTINUE
  120 CONTINUE
      NAC = 1
      IF (IDO .EQ. 2) RETURN
      NAC = 0
      DO 121 IK=1,IDL1
         C2(IK,1) = CH2(IK,1)
  121 CONTINUE
      DO 123 J=2,IP
         DO 122 K=1,L1
            C1(1,K,J) = CH(1,K,J)
            C1(2,K,J) = CH(2,K,J)
  122    CONTINUE
  123 CONTINUE
      IF (IDOT .GT. L1) GO TO 127
      IDIJ = 0
      DO 126 J=2,IP
         IDIJ = IDIJ+2
         DO 125 I=4,IDO,2
            IDIJ = IDIJ+2
            DO 124 K=1,L1
               C1(I-1,K,J) = WA(IDIJ-1)*CH(I-1,K,J)+WA(IDIJ)*CH(I,K,J)
               C1(I,K,J) = WA(IDIJ-1)*CH(I,K,J)-WA(IDIJ)*CH(I-1,K,J)
  124       CONTINUE
  125    CONTINUE
  126 CONTINUE
      RETURN
  127 IDJ = 2-IDO
      DO 130 J=2,IP
         IDJ = IDJ+IDO
         DO 129 K=1,L1
            IDIJ = IDJ
            DO 128 I=4,IDO,2
               IDIJ = IDIJ+2
               C1(I-1,K,J) = WA(IDIJ-1)*CH(I-1,K,J)+WA(IDIJ)*CH(I,K,J)
               C1(I,K,J) = WA(IDIJ-1)*CH(I,K,J)-WA(IDIJ)*CH(I-1,K,J)
  128       CONTINUE
  129    CONTINUE
  130 CONTINUE
      RETURN
      END

* --+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+

*DECK PASSF2
      SUBROUTINE PASSF2 (IDO,L1,CC,CH,WA1)
C***BEGIN PROLOGUE  PASSF2
C***REFER TO CFFTF
C***ROUTINES CALLED  (NONE)
C***END PROLOGUE  PASSF2
      DIMENSION       CC(IDO,2,L1)           ,CH(IDO,L1,2)           ,
     1                WA1(*)
C***FIRST EXECUTABLE STATEMENT  PASSF2
      IF (IDO .GT. 2) GO TO 102
      DO 101 K=1,L1
         CH(1,K,1) = CC(1,1,K)+CC(1,2,K)
         CH(1,K,2) = CC(1,1,K)-CC(1,2,K)
         CH(2,K,1) = CC(2,1,K)+CC(2,2,K)
         CH(2,K,2) = CC(2,1,K)-CC(2,2,K)
  101 CONTINUE
      RETURN
  102 DO 104 K=1,L1
         DO 103 I=2,IDO,2
            CH(I-1,K,1) = CC(I-1,1,K)+CC(I-1,2,K)
            TR2 = CC(I-1,1,K)-CC(I-1,2,K)
            CH(I,K,1) = CC(I,1,K)+CC(I,2,K)
            TI2 = CC(I,1,K)-CC(I,2,K)
            CH(I,K,2) = WA1(I-1)*TI2-WA1(I)*TR2
            CH(I-1,K,2) = WA1(I-1)*TR2+WA1(I)*TI2
  103    CONTINUE
  104 CONTINUE
      RETURN
      END

* --+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+

*DECK PASSF3
      SUBROUTINE PASSF3 (IDO,L1,CC,CH,WA1,WA2)
C***BEGIN PROLOGUE  PASSF3
C***REFER TO CFFTF
C***ROUTINES CALLED  (NONE)
C***END PROLOGUE  PASSF3
      DIMENSION       CC(IDO,3,L1)           ,CH(IDO,L1,3)           ,
     1                WA1(*)     ,WA2(*)
      DATA TAUR,TAUI /-.5,-.866025403784439/
C***FIRST EXECUTABLE STATEMENT  PASSF3
      IF (IDO .NE. 2) GO TO 102
      DO 101 K=1,L1
         TR2 = CC(1,2,K)+CC(1,3,K)
         CR2 = CC(1,1,K)+TAUR*TR2
         CH(1,K,1) = CC(1,1,K)+TR2
         TI2 = CC(2,2,K)+CC(2,3,K)
         CI2 = CC(2,1,K)+TAUR*TI2
         CH(2,K,1) = CC(2,1,K)+TI2
         CR3 = TAUI*(CC(1,2,K)-CC(1,3,K))
         CI3 = TAUI*(CC(2,2,K)-CC(2,3,K))
         CH(1,K,2) = CR2-CI3
         CH(1,K,3) = CR2+CI3
         CH(2,K,2) = CI2+CR3
         CH(2,K,3) = CI2-CR3
  101 CONTINUE
      RETURN
  102 DO 104 K=1,L1
         DO 103 I=2,IDO,2
            TR2 = CC(I-1,2,K)+CC(I-1,3,K)
            CR2 = CC(I-1,1,K)+TAUR*TR2
            CH(I-1,K,1) = CC(I-1,1,K)+TR2
            TI2 = CC(I,2,K)+CC(I,3,K)
            CI2 = CC(I,1,K)+TAUR*TI2
            CH(I,K,1) = CC(I,1,K)+TI2
            CR3 = TAUI*(CC(I-1,2,K)-CC(I-1,3,K))
            CI3 = TAUI*(CC(I,2,K)-CC(I,3,K))
            DR2 = CR2-CI3
            DR3 = CR2+CI3
            DI2 = CI2+CR3
            DI3 = CI2-CR3
            CH(I,K,2) = WA1(I-1)*DI2-WA1(I)*DR2
            CH(I-1,K,2) = WA1(I-1)*DR2+WA1(I)*DI2
            CH(I,K,3) = WA2(I-1)*DI3-WA2(I)*DR3
            CH(I-1,K,3) = WA2(I-1)*DR3+WA2(I)*DI3
  103    CONTINUE
  104 CONTINUE
      RETURN
      END

* --+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+

*DECK PASSF4
      SUBROUTINE PASSF4 (IDO,L1,CC,CH,WA1,WA2,WA3)
C***BEGIN PROLOGUE  PASSF4
C***REFER TO CFFTF
C***ROUTINES CALLED  (NONE)
C***END PROLOGUE  PASSF4
      DIMENSION       CC(IDO,4,L1)           ,CH(IDO,L1,4)           ,
     1                WA1(*)     ,WA2(*)     ,WA3(*)
C***FIRST EXECUTABLE STATEMENT  PASSF4
      IF (IDO .NE. 2) GO TO 102
      DO 101 K=1,L1
         TI1 = CC(2,1,K)-CC(2,3,K)
         TI2 = CC(2,1,K)+CC(2,3,K)
         TR4 = CC(2,2,K)-CC(2,4,K)
         TI3 = CC(2,2,K)+CC(2,4,K)
         TR1 = CC(1,1,K)-CC(1,3,K)
         TR2 = CC(1,1,K)+CC(1,3,K)
         TI4 = CC(1,4,K)-CC(1,2,K)
         TR3 = CC(1,2,K)+CC(1,4,K)
         CH(1,K,1) = TR2+TR3
         CH(1,K,3) = TR2-TR3
         CH(2,K,1) = TI2+TI3
         CH(2,K,3) = TI2-TI3
         CH(1,K,2) = TR1+TR4
         CH(1,K,4) = TR1-TR4
         CH(2,K,2) = TI1+TI4
         CH(2,K,4) = TI1-TI4
  101 CONTINUE
      RETURN
  102 DO 104 K=1,L1
         DO 103 I=2,IDO,2
            TI1 = CC(I,1,K)-CC(I,3,K)
            TI2 = CC(I,1,K)+CC(I,3,K)
            TI3 = CC(I,2,K)+CC(I,4,K)
            TR4 = CC(I,2,K)-CC(I,4,K)
            TR1 = CC(I-1,1,K)-CC(I-1,3,K)
            TR2 = CC(I-1,1,K)+CC(I-1,3,K)
            TI4 = CC(I-1,4,K)-CC(I-1,2,K)
            TR3 = CC(I-1,2,K)+CC(I-1,4,K)
            CH(I-1,K,1) = TR2+TR3
            CR3 = TR2-TR3
            CH(I,K,1) = TI2+TI3
            CI3 = TI2-TI3
            CR2 = TR1+TR4
            CR4 = TR1-TR4
            CI2 = TI1+TI4
            CI4 = TI1-TI4
            CH(I-1,K,2) = WA1(I-1)*CR2+WA1(I)*CI2
            CH(I,K,2) = WA1(I-1)*CI2-WA1(I)*CR2
            CH(I-1,K,3) = WA2(I-1)*CR3+WA2(I)*CI3
            CH(I,K,3) = WA2(I-1)*CI3-WA2(I)*CR3
            CH(I-1,K,4) = WA3(I-1)*CR4+WA3(I)*CI4
            CH(I,K,4) = WA3(I-1)*CI4-WA3(I)*CR4
  103    CONTINUE
  104 CONTINUE
      RETURN
      END

* --+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+

*DECK PASSF5
      SUBROUTINE PASSF5 (IDO,L1,CC,CH,WA1,WA2,WA3,WA4)
C***BEGIN PROLOGUE  PASSF5
C***REFER TO CFFTF
C***ROUTINES CALLED  (NONE)
C***END PROLOGUE  PASSF5
      DIMENSION       CC(IDO,5,L1)           ,CH(IDO,L1,5)           ,
     1                WA1(*)     ,WA2(*)     ,WA3(*)     ,WA4(*)
      DATA TR11,TI11,TR12,TI12 /.309016994374947,-.951056516295154,
     1-.809016994374947,-.587785252292473/
C***FIRST EXECUTABLE STATEMENT  PASSF5
      IF (IDO .NE. 2) GO TO 102
      DO 101 K=1,L1
         TI5 = CC(2,2,K)-CC(2,5,K)
         TI2 = CC(2,2,K)+CC(2,5,K)
         TI4 = CC(2,3,K)-CC(2,4,K)
         TI3 = CC(2,3,K)+CC(2,4,K)
         TR5 = CC(1,2,K)-CC(1,5,K)
         TR2 = CC(1,2,K)+CC(1,5,K)
         TR4 = CC(1,3,K)-CC(1,4,K)
         TR3 = CC(1,3,K)+CC(1,4,K)
         CH(1,K,1) = CC(1,1,K)+TR2+TR3
         CH(2,K,1) = CC(2,1,K)+TI2+TI3
         CR2 = CC(1,1,K)+TR11*TR2+TR12*TR3
         CI2 = CC(2,1,K)+TR11*TI2+TR12*TI3
         CR3 = CC(1,1,K)+TR12*TR2+TR11*TR3
         CI3 = CC(2,1,K)+TR12*TI2+TR11*TI3
         CR5 = TI11*TR5+TI12*TR4
         CI5 = TI11*TI5+TI12*TI4
         CR4 = TI12*TR5-TI11*TR4
         CI4 = TI12*TI5-TI11*TI4
         CH(1,K,2) = CR2-CI5
         CH(1,K,5) = CR2+CI5
         CH(2,K,2) = CI2+CR5
         CH(2,K,3) = CI3+CR4
         CH(1,K,3) = CR3-CI4
         CH(1,K,4) = CR3+CI4
         CH(2,K,4) = CI3-CR4
         CH(2,K,5) = CI2-CR5
  101 CONTINUE
      RETURN
  102 DO 104 K=1,L1
         DO 103 I=2,IDO,2
            TI5 = CC(I,2,K)-CC(I,5,K)
            TI2 = CC(I,2,K)+CC(I,5,K)
            TI4 = CC(I,3,K)-CC(I,4,K)
            TI3 = CC(I,3,K)+CC(I,4,K)
            TR5 = CC(I-1,2,K)-CC(I-1,5,K)
            TR2 = CC(I-1,2,K)+CC(I-1,5,K)
            TR4 = CC(I-1,3,K)-CC(I-1,4,K)
            TR3 = CC(I-1,3,K)+CC(I-1,4,K)
            CH(I-1,K,1) = CC(I-1,1,K)+TR2+TR3
            CH(I,K,1) = CC(I,1,K)+TI2+TI3
            CR2 = CC(I-1,1,K)+TR11*TR2+TR12*TR3
            CI2 = CC(I,1,K)+TR11*TI2+TR12*TI3
            CR3 = CC(I-1,1,K)+TR12*TR2+TR11*TR3
            CI3 = CC(I,1,K)+TR12*TI2+TR11*TI3
            CR5 = TI11*TR5+TI12*TR4
            CI5 = TI11*TI5+TI12*TI4
            CR4 = TI12*TR5-TI11*TR4
            CI4 = TI12*TI5-TI11*TI4
            DR3 = CR3-CI4
            DR4 = CR3+CI4
            DI3 = CI3+CR4
            DI4 = CI3-CR4
            DR5 = CR2+CI5
            DR2 = CR2-CI5
            DI5 = CI2-CR5
            DI2 = CI2+CR5
            CH(I-1,K,2) = WA1(I-1)*DR2+WA1(I)*DI2
            CH(I,K,2) = WA1(I-1)*DI2-WA1(I)*DR2
            CH(I-1,K,3) = WA2(I-1)*DR3+WA2(I)*DI3
            CH(I,K,3) = WA2(I-1)*DI3-WA2(I)*DR3
            CH(I-1,K,4) = WA3(I-1)*DR4+WA3(I)*DI4
            CH(I,K,4) = WA3(I-1)*DI4-WA3(I)*DR4
            CH(I-1,K,5) = WA4(I-1)*DR5+WA4(I)*DI5
            CH(I,K,5) = WA4(I-1)*DI5-WA4(I)*DR5
  103    CONTINUE
  104 CONTINUE
      RETURN
      END

* --+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+

*DECK RFFTB
      SUBROUTINE RFFTB (N,R,WSAVE)
C***BEGIN PROLOGUE  RFFTB
C***REVISION DATE  811015   (YYMMDD)
C***CATEGORY NO.  D6
C***KEYWORDS FFT,FAST FOURIER TRANSFORM,INVERSE,INVERSE FFT
C***DATE WRITTEN  FEBRUARY 1978
C***AUTHOR  SWARZTRAUBER P.N. (NCAR)
C***PURPOSE
C  INVERSE FFT OF A REAL PERIODIC SEQUENCE
C***DESCRIPTION
C     *****************************************************************
C
C     SUBROUTINE RFFTB(N,R,WSAVE)
C
C     *****************************************************************
C
C     SUBROUTINE RFFTB COMPUTES THE REAL PERODIC SEQUENCE FROM ITS
C     FOURIER COEFFICIENTS (FOURIER SYNTHESIS). THE TRANSFORM IS DEFINED
C     BELOW AT OUTPUT PARAMETER R.
C
C     INPUT PARAMETERS
C
C     N       THE LENGTH OF THE ARRAY R. N MUST BE EVEN AND THE METHOD
C             IS MOST EFFICIENT WHEN N IS A PRODUCT OF SMALL PRIMES.
C             N MAY CHANGE SO LONG AS DIFFERENT WORK ARRAYS ARE PROVIDED
C
C     R       A REAL ARRAY OF LENGTH N WHICH CONTAINS THE SEQUENCE
C             TO BE TRANSFORMED
C
C     WSAVE   A WORK ARRAY WHICH MUST BE DIMENSIONED AT LEAST 2.5*N+15
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
C     R       FOR I=1,...,N
C
C                  R(I)=X(1)+(-1)**(I+1)*X(2)
C
C                       PLUS THE SUM FROM K=2 TO K=N/2 OF
C
C                         2*R(2K-1)*COS((K-1)*(I-1)*2*PI/N)
C
C                        +2*R(2K)*SIN((K-1)*(I-1)*2*PI/N)
C
C      *****  NOTE
C                  THIS TRANSFORM IS UNNORMALIZED SINCE A CALL OF RFFTF
C                  FOLLOWED BY A CALL OF RFFTB WILL MULTIPLY THE INPUT
C                  SEQUENCE BY 2*N.
C
C     WSAVE   CONTAINS RESULTS WHICH MUST NOT BE DESTROYED BETWEEN
C             CALLS OF RFFTB OR RFFTF.
C
C
C***REFERENCES
C***ROUTINES CALLED  RFFTB1,XERROR
C***END PROLOGUE  RFFTB
      DIMENSION       R(2,*)     ,WSAVE(*)
C***FIRST EXECUTABLE STATEMENT  RFFTB
      IF (N .GT. 2) GO TO 101
      R1 = R(1,1)+R(2,1)
      R(2,1) = R(1,1)-R(2,1)
      R(1,1) = R1
      RETURN
  101 IW1 = N/2+1
      CALL RFFTB1 (N,R,WSAVE(IW1),WSAVE)
      RETURN
      END

* --+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+

*DECK RFFTB1
      SUBROUTINE RFFTB1 (N,X,XH,W)
C***BEGIN PROLOGUE  RFFTB1
C***REFER TO RFFTB
C***ROUTINES CALLED  CFFTB
C***END PROLOGUE  RFFTB1
      DIMENSION       X(2,*)     ,XH(2,*)    ,W(*)
C***FIRST EXECUTABLE STATEMENT  RFFTB1
      NS2 = N/2
      NS2P2 = NS2+2
      NQ = NS2/2
      IPAR = NS2-NQ-NQ
      NQM = NQ
      IF (IPAR .EQ. 0) NQM = NQM-1
      NQP = NQM+1
      XHOLD1 = X(1,1)-X(2,1)
      X(1,1) = X(2,1)+X(1,1)
      X(2,1) = XHOLD1
      IF (IPAR .NE. 0) GO TO 101
      X(1,NQP+1) = X(1,NQP+1)+X(1,NQP+1)
      X(2,NQP+1) = X(2,NQP+1)+X(2,NQP+1)
  101 IF (NQP .LT. 2) GO TO 108
      DO 102 K=2,NQP
         KC = NS2P2-K
         XH(1,K) = X(1,K)+X(1,KC)
         XH(2,K) = X(2,KC)-X(2,K)
  102 CONTINUE
      DO 103 K=2,NQP
         KC = NS2P2-K
         XH(1,KC) = X(1,K)-X(1,KC)
         XH(2,KC) = -X(2,K)-X(2,KC)
  103 CONTINUE
      DO 104 K=2,NQP
         KC = NS2P2-K
         X(1,KC) = XH(1,KC)
         X(2,KC) = XH(2,KC)
  104 CONTINUE
      DO 105 K=2,NQP
         KC = NS2P2-K
         XH(1,KC) = W(K-1)*X(1,KC)-W(KC-1)*X(2,KC)
         XH(2,KC) = W(K-1)*X(2,KC)+W(KC-1)*X(1,KC)
  105 CONTINUE
      DO 106 K=2,NQP
         KC = NS2P2-K
         X(1,K) = XH(1,K)-XH(2,KC)
         X(2,K) = XH(2,K)+XH(1,KC)
  106 CONTINUE
      DO 107 K=2,NQP
         KC = NS2P2-K
         X(1,KC) = XH(1,K)+XH(2,KC)
         X(2,KC) = XH(1,KC)-XH(2,K)
  107 CONTINUE
  108 CALL CFFTB (NS2,X,XH)
      RETURN
      END

* --+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+

*DECK RFFTF
      SUBROUTINE RFFTF (N,R,WSAVE)
C***BEGIN PROLOGUE  RFFTF
C***REVISION DATE  811015   (YYMMDD)
C***CATEGORY NO.  D6
C***KEYWORDS FFT,FAST FOURIER TRANSFORM,FOURIER TRANSFORM
C***DATE WRITTEN  FEBRUARY 1978
C***AUTHOR  SWARZTRAUBER P.N. (NCAR)
C***PURPOSE
C  FORWARD FFT OF A REAL PERIODIC SEQUENCE
C***DESCRIPTION
C     *****************************************************************
C
C     SUBROUTINE RFFTF(N,R,WSAVE)
C
C     *****************************************************************
C
C     SUBROUTINE RFFTF COMPUTES THE FOURIER COEFFICIENTS OF A REAL
C     PERODIC SEQUENCE (FOURIER ANALYSIS). THE TRANSFORM IS DEFINED
C     BELOW AT OUTPUT PARAMETER R.
C
C     INPUT PARAMETERS
C
C     N       THE LENGTH OF THE ARRAY R. N MUST BE EVEN AND THE METHOD
C             IS MOST EFFICIENT WHEN N IS A PRODUCT OF SMALL PRIMES.
C             N MAY CHANGE SO LONG AS DIFFERENT WORK ARRAYS ARE PROVIDED
C
C     R       A REAL ARRAY OF LENGTH N WHICH CONTAINS THE SEQUENCE
C             TO BE TRANSFORMED
C
C     WSAVE   A WORK ARRAY WHICH MUST BE DIMENSIONED AT LEAST 2.5*N+15
C             IN THE PROGRAM THAT CALLS RFFTF. THE WSAVE ARRAY MUST BE
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
C     R       FOR K=2,...,N/2
C
C                  R(2*K-1)= THE SUM FROM I=1 TO I=N OF
C
C                       2.*R(I)*COS((K-1)*(I-1)*2*PI/N)
C
C                  R(2*K)= THE SUM FROM I=1 TO I=N OF
C
C                       2.*R(I)*SIN((K-1)*(I-1)*2*PI/N)
C
C             ALSO
C
C                  R(1)= THE SUM FROM I=1 TO I=N OF 2.*R(I)
C
C                  R(2)= THE SUM FROM I=1 TO I=N OF 2.*(-1)**(I+1)*R(I)
C
C      *****  NOTE
C                  THIS TRANSFORM IS UNNORMALIZED SINCE A CALL OF RFFTF
C                  FOLLOWED BY A CALL OF RFFTB WILL MULTIPLY THE INPUT
C                  SEQUENCE BY 2*N.
C
C     WSAVE   CONTAINS RESULTS WHICH MUST NOT BE DESTROYED BETWEEN
C             CALLS OF RFFTF OR RFFTB.
C
C
C***REFERENCES
C***ROUTINES CALLED  RFFTF1,XERROR
C***END PROLOGUE  RFFTF
      DIMENSION       R(2,*)     ,WSAVE(*)
C***FIRST EXECUTABLE STATEMENT  RFFTF
      IF (N .GT. 2) GO TO 101
      R1 = 2.*(R(1,1)+R(2,1))
      R(2,1) = 2.*(R(1,1)-R(2,1))
      R(1,1) = R1
      RETURN
  101 IW1 = N/2+1
      CALL RFFTF1 (N,R,WSAVE(IW1),WSAVE)
      RETURN
      END

* --+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+

*DECK RFFTF1
      SUBROUTINE RFFTF1 (N,X,XH,W)
C***BEGIN PROLOGUE  RFFTF1
C***REFER TO RFFTF
C***ROUTINES CALLED  CFFTF
C***END PROLOGUE  RFFTF1
      DIMENSION       X(2,*)     ,XH(2,*)    ,W(*)
C***FIRST EXECUTABLE STATEMENT  RFFTF1
      NS2 = N/2
      NS2P2 = NS2+2
      NQ = NS2/2
      IPAR = NS2-NQ-NQ
      NQM = NQ
      IF (IPAR .EQ. 0) NQM = NQM-1
      NQP = NQM+1
      CALL CFFTF (NS2,X,XH)
      IF (NQP .LT. 2) GO TO 107
      DO 101 K=2,NQP
         KC = NS2P2-K
         XH(1,KC) = X(2,K)+X(2,KC)
         XH(2,KC) = X(1,KC)-X(1,K)
  101 CONTINUE
      DO 102 K=2,NQP
         KC = NS2P2-K
         XH(1,K) = X(1,K)+X(1,KC)
         XH(2,K) = X(2,K)-X(2,KC)
  102 CONTINUE
      DO 103 K=2,NQP
         KC = NS2P2-K
         X(1,KC) = W(K-1)*XH(1,KC)+W(KC-1)*XH(2,KC)
         X(2,KC) = W(K-1)*XH(2,KC)-W(KC-1)*XH(1,KC)
  103 CONTINUE
      DO 104 K=2,NQP
         KC = NS2P2-K
         XH(1,KC) = X(1,KC)
         XH(2,KC) = X(2,KC)
  104 CONTINUE
      DO 105 K=2,NQP
         KC = NS2P2-K
         X(1,KC) = XH(1,K)-XH(1,KC)
         X(2,KC) = XH(2,K)-XH(2,KC)
  105 CONTINUE
      DO 106 K=2,NQP
         KC = NS2P2-K
         X(1,K) = XH(1,K)+XH(1,KC)
         X(2,K) = -XH(2,K)-XH(2,KC)
  106 CONTINUE
      IF (IPAR .NE. 0) GO TO 108
  107 X(1,NQP+1) = X(1,NQP+1)+X(1,NQP+1)
      X(2,NQP+1) = X(2,NQP+1)+X(2,NQP+1)
  108 XHOLD1 = X(2,1)+X(1,1)
      XHOLD2 = X(1,1)-X(2,1)
      X(2,1) = XHOLD2+XHOLD2
      X(1,1) = XHOLD1+XHOLD1
      RETURN
      END

* --+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+

*DECK RFFTI
      SUBROUTINE RFFTI (N,WSAVE)
C***BEGIN PROLOGUE  RFFTI
C***REVISION DATE  811015   (YYMMDD)
C***CATEGORY NO.  D6
C***KEYWORDS FFT,FAST FOURIER TRANSFORM,FOURIER TRANSFORM
C***DATE WRITTEN  FEBRUARY 1978
C***AUTHOR  SWARZTRAUBER P.N. (NCAR)
C***PURPOSE
C   INITIALIZE FOR RFFTF AND RFFTB
C***DESCRIPTION
C     *****************************************************************
C
C     SUBROUTINE RFFTI(N,WSAVE)
C
C       ***************************************************************
C
C     SUBROUTINE RFFTI INITIALIZES THE ARRAY WSAVE WHICH IS USED IN
C     BOTH RFFTF AND RFFTB. THE PRIME FACTORIZATION OF N TOGETHER WITH
C     A TABULATION OF THE TRIGONOMETRIC FUNCTIONS ARE COMPUTED AND
C     STORED IN WSAVE.
C
C     INPUT PARAMETER
C
C     N       THE LENGTH OF THE SEQUENCE TO BE TRANSFORMED. N MUST BE
C             EVEN
C
C     OUTPUT PARAMETER
C
C     WSAVE   A WORK ARRAY WHICH MUST BE DIMENSIONED AT LEAST 2.5*N+15.
C             THE SAME WORK ARRAY CAN BE USED FOR BOTH RFFTF AND RFFTB
C             AS LONG AS N REMAINS UNCHANGED. DIFFERENT WSAVE ARRAYS
C             ARE REQUIRED FOR DIFFERENT VALUES OF N. THE CONTENTS OF
C             WSAVE MUST NOT BE CHANGED BETWEEN CALLS OF RFFTF OR RFFTB
C
C***REFERENCES
C***ROUTINES CALLED  CFFTI,XERROR
C***END PROLOGUE  RFFTI
      DIMENSION       WSAVE(*)
C***FIRST EXECUTABLE STATEMENT  RFFTI
      N = 2* (N/2)  ! N must be even

      NS2 = N/2
      NQM = (NS2-1)/2
      TPI = 8.*ATAN(1.)
      DT = TPI/FLOAT(N)
      DC = COS(DT)
      DS = SIN(DT)
      WSAVE(1) = DC
      WSAVE(NS2-1) = DS
      IF (NQM .LT. 2) GO TO 102
      DO 101 K=2,NQM
         KC = NS2-K
         WSAVE(K) = DC*WSAVE(K-1)-DS*WSAVE(KC+1)
         WSAVE(KC) = DS*WSAVE(K-1)+DC*WSAVE(KC+1)
  101 CONTINUE
  102 IW1 = NS2+1
      CALL CFFTI (NS2,WSAVE(IW1))
      RETURN
      END
      SUBROUTINE COSWINDOW (W, ND, NP)

c   Cosine tapers end values for NP/100 percent of ND values for start and end
c   values

c   written by Jim Larsen
c   revised 1/24/96

c   Calls: no subroutines

      DIMENSION W(*)
      REAL*8 PI, XM, CC, S1, S2, S3

      DO I = 1, ND
        W(I) = 1.0
      ENDDO

      PI = 3.1415926535897932384

      MD = 0.01* NP* ND + 0.5
      XM = PI/ (MD* 2)

      W(1)  = 0.0
      W(ND) = 0.0

      CC = 2.D0* COS (XM)
      S1 = 0.D0
      S2 = SIN (XM)

      J = ND
      DO I = 2, MD
        J = J - 1
        W(I) = S2
        W(J) = S2 
        S3 = CC* S2 - S1
        S1 = S2
        S2 = S3
      ENDDO

      S = 0.0
      DO I = 1, ND
        S = S + W(I)**2
      ENDDO
      S = SQRT (ND/ S)
      DO I = 1, ND
        W(I) = S* W(I)
      ENDDO

      RETURN
      END
      SUBROUTINE PROLATE (W, ND, ISW)
C  CALCULATES PROLATE SPHEROIDAL WAVEFUNCTION DATA WINDOW FOR
C  HIGH RESOLUTION FOURIER ANALYSIS
C  REF: D.J.THOMSON, BELL SYST. TECH. J. 56,1769-1815 (1977)
C
C  W IS A SINGLE PRECISON REAL ARRAY OF DATA WINDOW VALUES
C  ND IS THE NUMBER OF POINTS IN W
C  ISW IS A SWITCH--ISW=4 MEANS USE A 4-PI WINDOW, ISW=1 MEANS USE
C      THE HIGHER RESOLUTION PI WINDOW

C  SCALE FACTORS=INTEGRAL(BOXCAR)/INTEGRAL(PROLATE WINDOW) ARE:
C    4 PI PROLATE WINDOW--1.425658520238489
C    PI PROLATE WINDOW--1.057568010371401
C  THESE ARE THE NUMBERS TO MULTIPLY THE SPECTRUM BY FOR COMPARISON
C  WITH OTHER WINDOWS

c     revise 4/20/98

      REAL W(*)
      REAL*8 UN, D, U, WI
      UN = DBLE(4)/(DBLE(ND-1)* DBLE(ND-1))
      IF (ISW .EQ. 4) THEN
        D = SQRT (DBLE(2)/ 0.508125548147497D0)
        DO I = 1, ND
          U  = DBLE(ND-I)* DBLE(I-1)* UN
          WI = D*(((((((((((((((((((((
     $        2.6197747176990866D-11*U+2.9812025862125737D-10)*U+
     $        3.0793023552299688D-9)*U+2.8727486379692354D-8)*U+
     $        2.4073904863499725D-7)*U+1.8011359410323110D-6)*U+
     $        1.1948784162527709D-5)*U+6.9746276641509466D-5)*U+
     $        3.5507361197109845D-4)*U+1.5607376779150113D-3)*U+
     $        5.8542015072142441D-3)*U+1.8482388295519675D-2)*U+
     $        4.8315671140720506D-2)*U+1.0252816895203814D-1)*U+
     $        1.7233583271499150D-1)*U+2.2242525852102708D-1)*U+
     $        2.1163435697968192D-1)*U+1.4041394473085307D-1)*U+
     $        5.9923940532892353D-2)*U+1.4476509897632850D-2)*U+
     $        1.5672417352380246D-3)*U+4.2904633140034110D-5)
          W(I) = WI
        ENDDO
        RETURN
      ELSE
        D = SQRT (DBLE(2))
        DO I = 1, ND
          U  = DBLE(ND-I)* DBLE(I-1)* UN
          WI = D*((((((((((
     $        5.3476939016920851D-11*U+2.2654256220146656D-9)*U+
     $        7.8075102004229667D-8)*U+2.1373409644281953D-6)*U+
     $        4.5094847544714943D-5)*U+7.0498957221483167D-4)*U+
     $        7.7412693304064753D-3)*U+5.5280627452077586D-2)*U+
     $        2.2753754228751827D-1)*U+4.3433904277546202D-1)*U+
     $        2.2902051859068017D-1)
          W(I) = WI
        ENDDO
        RETURN
      ENDIF
      END
