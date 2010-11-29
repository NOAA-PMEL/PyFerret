	SUBROUTINE CALC_CVAR( memory, cx, mrl, mres, status )

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
*
* calculate one or more variables based on the components supplied

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  7/24/89 - based on CALC_XYGRID
*	10/16/89 - adjustable array dimensioning via NON_ARRAY_SUBSC
*	 2/27/90 - added v2.0 CAIR
*		 - and made it able to handle unspecified result region limits
*	 3/13/90 - fixed subsc. err. in QAD (et.al.) with unspec lims
*	 5/15/90 - special unknown subscript checks for AIR components
* V312: 5/94 - array "memory" as a calling argument
* V320: 12/29/94 - use SPLIT_LIST to redirect tty output for GUI

* calling arguments:
* inp	cx	- "context" of the desired result (variable,data set and region)
* inp	mrl	- list of memory variable numbers for the 
*			  component variables of the calculation
* out	mres	- memory variable number of requested result

#ifdef unix
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'instance_table.parm'
	include	'gfdl.parm'
	include	'xvariables.cmn'
	include	'xcalc_vars.cmn'
	include	'xprog_state.cmn'
	include	'xcontext.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
	INCLUDE	'FERRET_CMN:INSTANCE_TABLE.PARM'
	INCLUDE	'FERRET_CMN:GFDL.PARM'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XCALC_VARS.CMN'
	INCLUDE	'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
#endif

	INTEGER		max_components
	PARAMETER     ( max_components = 20 )

* calling argument declarations:
	INTEGER		cx, mrl( max_components ), mres, status
	REAL	memory( mem_blk_size, max_mem_blks )

* internal variable declarations:
	LOGICAL		HOURLY_DATA, chg(4), unk(4), reg, two_results
	INTEGER		CGRID_SIZE, MGRID_SIZE, CGRID_AXIS, MR_DIM_LEN,
     .			nmr, mxtra, cx_xtra, cx_lrg, grid_size, temp_grid_size,
     .			var, ncomp, icomp, iend, k, i, idim, mr, fp, grid,
     .			temp_start_blk, temp_nblks, dlo, dhi,
     .			loest_ss, hiest_ss, res_ss_lo(4), res_ss_hi(4)
	REAL		dt
	REAL*8		TM_WORLD

* for diagnostic output:
	CHARACTER*4	VAR_CODE
	CHARACTER*132	diag_buff
	DATA		diag_buff(1:9) / '-->using:' /

* remove special codes from variable to compute
	var = cx_variable( cx )
	IF ( var .GT. pcv_spec_flg ) var = var - pcv_spec_flg
	cx_variable( cx ) = var

* initialize some stuff
	two_results = cvar_num_results( var ) .GT. 1	! only 2 or 1 allowed
	ncomp  = cvar_frame_length( var )
	grid   = cx_grid( cx )
	DO 1 idim = 1, 4
	res_ss_lo(idim) = 0
	res_ss_hi(idim) = 999999
	unk( idim ) = .FALSE.
 1	chg( idim ) = .FALSE.

* diagnostic output ?
	IF ( mode_diagnostic ) THEN
	   diag_buff = '--> '//VAR_CODE( cat_calc_var, var )//' from: '
	   iend = 15
	   DO 2 icomp = 1, cvar_frame_length( var )
	      iend = iend + 5
	      mr = mrl( icomp )
	      WRITE ( diag_buff( iend-4:iend ), '(1X,A)' )
     .			VAR_CODE(  mr_category(mr),mr_variable(mr) )
 2	   CONTINUE
	   CALL SPLIT_LIST( pttmode_ops, ttout_lun, diag_buff, iend )
	ENDIF

* shrink requested dimensions to match smallest available component
	fp = cvar_frame_start (var)		! frame pointer
	DO 6 icomp = 1, ncomp
	   fp = fp + 1
	   mr = mrl( icomp )
	   IF ( mr .EQ. dummy_mr ) GOTO 6
	   DO 4 idim = 1, 4
	      IF ( CGRID_AXIS( idim, cx ) .EQ. mnormal ) GOTO 4
	      IF ( mr_lo_ss( mr, idim ) .EQ. unspecified_int4 ) GOTO 4
	      CALL GRID_SUBSCRIPT_EXTREMES ( loest_ss, hiest_ss, grid, idim )
	      dlo = cv_lo_ds(fp,idim)
	      dhi = cv_hi_ds(fp,idim)
	      IF ( idim.EQ.T_dim .AND. dlo.GT.pcv_hourly_flg ) THEN
	         dlo = 0
	         dhi = 0
	         IF ( HOURLY_DATA( grid ) ) dlo = -1
	      ENDIF
* ... 5/90: air temp is special because it sometimes uses regridded CAIR
* ... if it gets here with t_dim unspecified in the SST component
* ... it must be a non-regridded case (note: mr(cair)-->dummy in this case)
	      IF ( var .EQ. pair ) THEN
	         IF ( idim .EQ. t_dim ) THEN
	            IF ( cx_lo_ss(cx,idim) .EQ. unspecified_int4 ) THEN
	               res_ss_lo(idim) = mr_lo_ss(mr,idim)
	               res_ss_hi(idim) = mr_hi_ss(mr,idim)
	               chg( idim ) = .TRUE.
	               unk( idim ) = .TRUE.
	               GOTO 4
	            ENDIF
	         ENDIF
	      ENDIF
* ... fill in unspecified limits (2/90) - fixed QAD error 3/90
	      IF ( cx_lo_ss(cx,idim) .EQ. unspecified_int4 ) THEN
	         res_ss_lo(idim) = MAX( res_ss_lo(idim),
     .					mr_lo_ss(mr,idim) - dlo )
	         res_ss_hi(idim) = MIN( res_ss_hi(idim),
     .					mr_hi_ss(mr,idim) - dhi )
	         chg( idim ) = .TRUE.
	         unk( idim ) = .TRUE.
	         GOTO 4
	      ENDIF
* ... trim limits to available source data
	      IF ( cx_lo_ss(cx,idim) .NE. loest_ss
     .	     .AND. cx_lo_ss(cx,idim)+dlo .LT. mr_lo_ss(mr,idim) ) THEN
	         cx_lo_ss(cx,idim) = mr_lo_ss(mr,idim) - dlo
	         chg( idim ) = .TRUE.
	      ENDIF
	      IF ( cx_hi_ss(cx,idim) .NE. hiest_ss
     .	     .AND. cx_hi_ss(cx,idim)+dhi .GT. mr_hi_ss(mr,idim) ) THEN
	         cx_hi_ss(cx,idim) = mr_hi_ss(mr,idim) - dhi
	         chg( idim ) = .TRUE.
	      ENDIF
 4	   CONTINUE
 6	CONTINUE
	DO 8 idim = 1, 4
	   IF ( unk(idim) ) THEN
* ... fill unspecified result limits with the amount of component available
	      cx_lo_ss(cx,idim) = res_ss_lo(idim)
	      cx_hi_ss(cx,idim) = res_ss_hi(idim)
	   ENDIF
	   IF ( chg(idim) ) THEN
	      cx_by_ss(idim,cx) = .TRUE.
	      CALL CONFINE_AXIS( idim, cx, status )
	      IF ( status .NE. ferr_ok ) RETURN
	      CALL FLESH_OUT_AXIS( idim, cx, status )	
	      IF ( status .NE. ferr_ok ) GOTO 5000
	   ENDIF
 8	CONTINUE

* create another context in case result must be larger that requested size
* (for example baroclinic pressure must always be computed from Z=0 even if
*  it is requested only at depth)
	CALL STACK_PTR_UP( cx_stack_ptr, max_context, status )
	IF ( status .NE. ferr_ok ) RETURN
	cx_lrg = cx_stack_ptr
	CALL TRANSFER_CONTEXT( cx, cx_lrg )

* Z range to start at K=1 for components of baroclinic pres. and W
	IF ( var.EQ.pdpdx  .OR. var.EQ.pdpdy
     .	.OR. var.EQ.pw_u   .OR. var.EQ.pw_t  ) THEN
	   DO 9 icomp = 1, ncomp
	      mr = mrl( icomp )
	      IF ( mr_lo_ss( mr, z_dim ) .EQ. unspecified_int4 ) GOTO 9
	      IF ( mr_lo_ss( mr, z_dim ) .NE. 1 ) CALL ERRMSG
     .		( ferr_out_of_range, status,
     .		'baroclinic pressure requires surface data', *5000 )
 9	   CONTINUE
* ... result will go to k=1, also
	   cx_lo_ss( cx_lrg, z_dim ) = 1
	   cx_lo_ww(z_dim, cx_lrg) = TM_WORLD( 1, cx_grid(cx_lrg),
     .					       z_dim, box_lo_lim )
	ENDIF
	grid_size = CGRID_SIZE( cx_lrg )

* reserve memory space for result
	CALL CREATE_MEM_VAR( cx_lrg, mres, status )
	IF ( status .NE. ferr_ok ) RETURN
	nmr = ncomp + 1
	IF ( isp+nmr .GT. max_intrp ) CALL ERRMSG
     .				( ferr_stack_ovfl, status, ' ',*5200 )
	mrl(nmr) = mres	! put it in the list fot NON_ARRAY_SUBSC

* create another context in case 2 results are computed at once
	IF ( two_results ) THEN
	   CALL STACK_PTR_UP( cx_stack_ptr, max_context, status )
	   IF ( status .NE. ferr_ok ) RETURN
	   cx_xtra = cx_stack_ptr
	   CALL TRANSFER_CONTEXT( cx_lrg, cx_xtra )
	   CALL CREATE_MEM_VAR( cx_xtra, mxtra, status )
	   IF ( status .NE. ferr_ok ) RETURN
	   nmr = nmr + 1
	   IF ( isp+nmr .GT. max_intrp ) CALL ERRMSG
     .				( ferr_stack_ovfl, status, ' ',*5100 )
	   mrl(nmr) = mxtra	! put it in the list fot NON_ARRAY_SUBSC
	ENDIF

* extract all the necessary dimensions to unsubscripted variables
	CALL NON_ARRAY_SUBSC( mrl, nmr )

* see BLOCK DATA XVARIABLES_DATA for lists of components in each calculation
	GOTO (	 10, 20, 30, 40, 50, 60, 70, 80, 90,100,
     .		110,120,130,140,150,160,170,180,190,200,
     .		210,220,230,240,250,260,270,280,290,300,
     .		310,320,330,340,350,360,370,380,390,400,
     .		410,420,430,440,450,460,470,480,490,500,
     .		510,520,530,540,550,560,570,580,590		 ) var

* QCDZ - vertical heat diffussion coefficient
* ... also computes and stores PCDZ ...
 10	cx_variable( cx_xtra ) = ppcdz
	cx_grid( cx_xtra ) = cvar_grid( ppcdz, cx_data_set(cx) )
	CALL RE_ASSIGN_VARIABLE( mxtra, cx_xtra )
	CALL PROF (
     .		km					, ! k max
     .		memory( 1, mr_blk1( mrl(1) ) ), mrl(1)	, ! u
     .		memory( 1, mr_blk1( mrl(2) ) ), mrl(2)	, ! v
     .		memory( 1, mr_blk1( mrl(3) ) ), mrl(3)	, ! rho0
     .		memory( 1, mr_blk1( mxtra  ) ), mxtra	, ! pcdz
     .		memory( 1, mr_blk1( mres   ) ), mres	) ! qcdz
	GOTO 1900

* PCDZ - vertical momentum diffussion coefficient
* ... also computes and stores QCDZ ...
 20	cx_variable( cx_xtra ) = pqcdz
	cx_grid( cx_xtra ) = cvar_grid( pqcdz, cx_data_set(cx) )
	CALL RE_ASSIGN_VARIABLE( mxtra, cx_xtra )
	CALL PROF (
     .		km					, ! k max
     .		memory( 1, mr_blk1( mrl(1) ) ), mrl(1)	, ! u
     .		memory( 1, mr_blk1( mrl(2) ) ), mrl(2)	, ! v
     .		memory( 1, mr_blk1( mrl(3) ) ), mrl(3)	, ! rho0
     .		memory( 1, mr_blk1( mres   ) ), mres	, ! pcdz
     .		memory( 1, mr_blk1( mxtra  ) ), mxtra	) ! qcdz
	GOTO 1900

* RHO0 - density referenced to surface ( compression effects neglected )
 30	CALL STATED (
     .		memory( 1, mr_blk1( mrl(1) ) ), mrl(1)	,	  ! temp
     .		memory( 1, mr_blk1( mrl(2) ) ), mrl(2)	, 	  ! salt
     .		memory( 1, mr_blk1( mres   ) )	, mres		) ! rho0
	GOTO 2000

* QDFZ - vertical heat diffussion
 40	CALL Z_HDIFFUS (
     .		memory( 1, mr_blk1( mrl(1) ) ), mrl(1)	, ! temp
     .		memory( 1, mr_blk1( mrl(2) ) ), mrl(2)	, ! qvdc
     .		memory( 1, mr_blk1( mrl(3) ) ), mrl(3)	, ! qflx
     .		memory( 1, mr_blk1( mres   ) )	, mres	) ! qdfz
	GOTO 2000

* QDFH - horizontal heat diffussion
 50	CALL XY_HDIFFUS (
     .		memory( 1, mr_blk1( mrl(1) ) ), mrl(1)	, ! temp
     .		memory( 1, mr_blk1( mres   ) ), mres	) ! qdfh
	GOTO 2000

* QADX - zonal heat advection
 60	CALL EW_HADVEC (
     .		memory( 1, mr_blk1( mrl(1) ) ), mrl(1)	, ! temp
     .		memory( 1, mr_blk1( mrl(2) ) ), mrl(2)	, ! u   
     .		memory( 1, mr_blk1( mres   ) ), mres	) ! qadx
	GOTO 2000

* QADY - meridianal heat advection
 70	CALL SN_HADVEC (
     .		memory( 1, mr_blk1( mrl(1) ) ), mrl(1)	, ! temp
     .		memory( 1, mr_blk1( mrl(2) ) ), mrl(2)	, ! v   
     .		memory( 1, mr_blk1( mres   ) ), mres	) ! qady
	GOTO 2000

* QADZ - vertical heat advection
 80	CALL Z_HADVEC (
     .		memory( 1, mr_blk1( mrl(1) ) ), mrl(1)	, ! temp
     .		memory( 1, mr_blk1( mrl(2) ) ), mrl(2)	, ! w   
     .		memory( 1, mr_blk1( mres   ) ), mres	) ! qadz
	GOTO 2000

* QAD  - total heat advection
 90	CALL COPY_GRID(
     .	  memory( 1, mr_blk1(mrl(1)) ) , mrl(1),		! qadx
     .	  memory( 1, mr_blk1( mres   )), mres	)		! qad
	CALL ADD_2_GRIDS(
     .	  memory( 1, mr_blk1(mrl(2)) ) , mrl(2),		! qady
     .	  memory( 1, mr_blk1( mres   )), mres	 )		! qad
	CALL ADD_2_GRIDS(
     .	  memory( 1, mr_blk1(mrl(3)) ) , mrl(3),		! qadz
     .	  memory( 1, mr_blk1( mres   )), mres 	 )		! qad
	GOTO 2000	

* QDF  - total heat diffussion
 100	CALL COPY_GRID(
     .	  memory( 1, mr_blk1(mrl(1)) ) , mrl(1),		! qdfh
     .	  memory( 1, mr_blk1( mres   )), mres 	 )		! qdf
	CALL ADD_2_GRIDS(
     .	  memory( 1, mr_blk1(mrl(2)) ),  mrl(2),		! qdfz
     .	  memory( 1, mr_blk1( mres   )), mres	 )		! qdf
	GOTO 2000

* QEVA - surface evaporative cooling
* ... also computes and stores QSEN ...
 110	cx_variable( cx_xtra ) = pqsen
	CALL RE_ASSIGN_VARIABLE( mxtra, cx_xtra )
	CALL HFLUX (
     .		memory( 1, mr_blk1( mrl(1) ) ), mrl(1)	, ! tau
     .		memory( 1, mr_blk1( mrl(2) ) ), mrl(2)	, ! sst
     .		memory( 1, mr_blk1( mrl(3) ) ), mrl(3)	, ! air
     .		memory( 1, mr_blk1( mxtra  ) ), mxtra	, ! qsen
     .		memory( 1, mr_blk1( mres   ) ), mres	) ! qeva
* ... convert units ...
	CALL MULT_GRID( grid_size, memory(1,mr_blk1( mxtra  )), q2wattm2 )
	CALL MULT_GRID( grid_size, memory(1,mr_blk1( mres   )), q2wattm2 )
	GOTO 1900

* TAU  - wind stress magnitude
 120	CALL WIND_STRESS (
     .		memory( 1, mr_blk1( mrl(1) ) ), mrl(1)	,	! taux
     .		memory( 1, mr_blk1( mrl(2) ) ), mrl(2)	,	! tauy
     .		memory( 1, mr_blk1( mres   ) ), mres	)	! tau
	GOTO 2000

* QSEN - surface sensible heat
* ... also computes and stores QEVA ...
 130	cx_variable( cx_xtra ) = pqeva
	CALL RE_ASSIGN_VARIABLE( mxtra, cx_xtra )
	CALL HFLUX (
     .		memory( 1, mr_blk1( mrl(1) ) ), mrl(1)	, ! tau
     .		memory( 1, mr_blk1( mrl(2) ) ), mrl(2)	, ! sst
     .		memory( 1, mr_blk1( mrl(3) ) ), mrl(3)	, ! air
     .		memory( 1, mr_blk1( mres   ) ), mres	, ! qsen
     .		memory( 1, mr_blk1( mxtra  ) ), mxtra	) ! qeva
* ... convert units ...
	CALL MULT_GRID( grid_size, memory(1,mr_blk1( mxtra  )), q2wattm2 )
	CALL MULT_GRID( grid_size, memory(1,mr_blk1( mres   )), q2wattm2 )
	GOTO 1900

* QRAD - surface radiative heat
 140	CALL HRADIATION( 
     .		memory( 1, mr_blk1( mrl(1) ) ), mrl(1)	,	! sst
     .		memory( 1, mr_blk1( mres   ) ), mres	)	! qrad
*    ... convert units
	CALL MULT_GRID( grid_size, memory(1, mr_blk1( mres   )), q2wattm2 )
	GOTO 2000

* QFLX  - total heat flux
 150	CALL COPY_GRID(
     .	  memory( 1, mr_blk1(mrl(1)) )  , mrl(1) , 			! qeva
     .	  memory( 1, mr_blk1( mres   ) ), mres	  )			! qflx
	CALL ADD_2_GRIDS(
     .	  memory( 1, mr_blk1(mrl(2)) )  , mrl(2) , 			! qsen
     .	  memory( 1, mr_blk1( mres   ) ), mres	  )			! qflx
	CALL ADD_2_GRIDS(
     .	  memory( 1, mr_blk1(mrl(3)) )  , mrl(3) , 			! qrad
     .	  memory( 1, mr_blk1( mres   ) ), mres	  )			! qflx
	GOTO 2000

* RHO - true density
 160	CALL STATE (
     .		memory( 1, mr_blk1( mrl(1) ) ), mrl(1)	,		! temp
     .		memory( 1, mr_blk1( mrl(2) ) ), mrl(2)	,		! salt
     .		memory( 1, mr_blk1( mres   ) ), mres		)	! rho
	GOTO 2000

* DPDX - x derivative of baroclinic pressure
* ... also computes and stores DPDY ...
 170	cx_variable( cx_xtra ) = pdpdy
	CALL RE_ASSIGN_VARIABLE( mxtra, cx_xtra )
	CALL GRAD_P_CLINIC (
     .		memory( 1, mr_blk1( mrl(1) ) ), mrl(1)	, ! rho 
     .		memory( 1, mr_blk1( mres   ) ), mres	, ! dpdx
     .		memory( 1, mr_blk1( mxtra  ) ), mxtra	) ! dpdy
	GOTO 1900

* DPDY - y derivative of baroclinic pressure
* ... also computes and stores DPDX ...
 180	cx_variable( cx_xtra ) = pdpdx
	CALL RE_ASSIGN_VARIABLE( mxtra, cx_xtra )
	CALL GRAD_P_CLINIC (
     .		memory( 1, mr_blk1( mrl(1) ) ), mrl(1)	, ! rho 
     .		memory( 1, mr_blk1( mxtra  ) ), mxtra	, ! dpdx
     .		memory( 1, mr_blk1( mres   ) ), mres	) ! dpdy
	GOTO 1900

* W_U - vertical velocity computed on the uv grid (still on w grid vertically)
 190	CALL W_ON_UV (
     .		memory( 1, mr_blk1( mrl(1) ) ), mrl(1)	, ! fuw
     .		memory( 1, mr_blk1( mrl(2) ) ), mrl(2)	, ! fvn
     .		memory( 1, mr_blk1( mres   ) ), mres	) ! w_u
	GOTO 2000

* UAVZ - vertically averaged zonal velocity calculated from PSI
* ... also computes and stores VAVZ ...
 200	cx_variable( cx_xtra ) = pvavz
	CALL RE_ASSIGN_VARIABLE( mxtra, cx_xtra )
	CALL U_FROM_PSI (
     .		memory( 1, mr_blk1( mrl(1) ) ), mrl(1)	, ! psi
     .		memory( 1, mr_blk1( mres   ) ), mres	, ! uavz
     .		memory( 1, mr_blk1( mxtra  ) ), mxtra	) ! vavz
	GOTO 1900

* VAVZ - vertically averaged meridional velocity calculated from PSI
* ... also computes and stores UAVZ ...
 210	cx_variable( cx_xtra ) = puavz
	CALL RE_ASSIGN_VARIABLE( mxtra, cx_xtra )
	CALL U_FROM_PSI (
     .		memory( 1, mr_blk1( mrl(1) ) ), mrl(1)	, ! psi
     .		memory( 1, mr_blk1( mxtra  ) ), mxtra	, ! uavz
     .		memory( 1, mr_blk1( mres   ) ), mres	) ! vavz
	GOTO 1900

* FUW - zonal advection coefficient ( internal variable for W_U and *ADX )
 220	CALL FUW (
     .		memory( 1, mr_blk1( mrl(1) ) ), mrl(1)	, ! psi
     .		memory( 1, mr_blk1( mrl(2) ) ), mrl(2)	, ! uavz
     .		memory( 1, mr_blk1( mrl(3) ) ), mrl(3)	, ! u
     .		memory( 1, mr_blk1( mres   ) ), mres	) ! fuw
	GOTO 2000

* FVN - meridional advection coefficient ( internal variable for W_U and *ADY )
 230	CALL FVN (
     .		memory( 1, mr_blk1( mrl(1) ) ), mrl(1)	, ! psi
     .		memory( 1, mr_blk1( mrl(2) ) ), mrl(2)	, ! vavz
     .		memory( 1, mr_blk1( mrl(3) ) ), mrl(3)	, ! v
     .		memory( 1, mr_blk1( mres   ) ), mres	) ! fvn
	GOTO 2000

* UADX - EW advection of U
 240	CALL VEL_ADV_EW (
     .		memory( 1, mr_blk1( mrl(1) ) ), mrl(1)	, ! fuw
     .		memory( 1, mr_blk1( mrl(2) ) ), mrl(2)	, ! u
     .		memory( 1, mr_blk1( mres   ) ), mres	) ! uadx
	GOTO 2000

* UADY - NS advection of U
 250	CALL VEL_ADV_NS (
     .		memory( 1, mr_blk1( mrl(1) ) ), mrl(1)	, ! fvn
     .		memory( 1, mr_blk1( mrl(2) ) ), mrl(2)	, ! u
     .		memory( 1, mr_blk1( mres   ) ), mres	) ! uady
	GOTO 2000

* UADZ - vertical advection of U
 260	CALL VEL_ADV_Z(
     .		memory( 1, mr_blk1( mrl(1) ) ), mrl(1)	, ! u
     .		memory( 1, mr_blk1( mrl(2) ) ), mrl(2)	, ! w
     .		memory( 1, mr_blk1( mres   ) ), mres	) ! uadz
	GOTO 2000

* VADX - EW advection of V
* ... calculation is identical to UADX but with different components
 270	GOTO 240

* VADY - NS advection of V
* ... calculation is identical to UADY but with different components
 280	GOTO 250

* VADZ - vertical advection of V
* ... calculation is identical to UADZ but with different components
 290	GOTO 260

* UDFH - horizontal diffusion of U
 300	CALL VEL_DIF_XY(
     .		memory( 1, mr_blk1( mrl(1) ) ), mrl(1)	, ! u
     .		memory( 1, mr_blk1( mrl(2) ) ), mrl(2)	, ! v
     .		memory( 1, mr_blk1( mres   ) ), mres	) ! udfh
	GOTO 2000

* UDFZ - vertical diffusion of U
 310	CALL VEL_DIF_Z(
     .		memory( 1, mr_blk1( mrl(1) ) ), mrl(1)	, ! pcdz
     .		memory( 1, mr_blk1( mrl(2) ) ), mrl(2)	, ! u
     .		memory( 1, mr_blk1( mrl(3) ) ), mrl(3)	, ! ubwi
     .		memory( 1, mr_blk1( mres   ) ), mres	) ! udfz
	GOTO 2000

* VDFH - horizontal advection of V
* ... calculation is identical to UDFH but with different components
 320	GOTO 300

* VDFZ - vertical advection of V
* ... calculation is identical to UDFZ but with different components
 330	GOTO 310

* UCOI - implicit calculation of Coriolis effect on U
 340	CALL ERRMSG(ferr_unknown_variable,status,
     .				'UCOI no longer implemented',*5000)
* VCOI - implicit calculation of Coriolis effect on V
 350	CALL ERRMSG(ferr_unknown_variable,status,
     .					'UCOI no longer implemented',*5000)

* UBPI - Grad( Baroclinic pressure ) effect on U
 360	CALL VEL_BY_GRAD_P (
     .		memory( 1, mr_blk1( mrl(1) ) ), mrl(1)	, ! dpdx
     .		memory( 1, mr_blk1( mres   ) ), mres	) ! ubpi
	GOTO 2000

* VBPI - Grad( Baroclinic pressure ) effect on V
* ... calculation is identical to UBPI but with different components
 370	GOTO 360

* UBWI - wind stress effect on U
 380	CALL VEL_BY_WIND (
     .		memory( 1, mr_blk1( mrl(1) ) ), mrl(1)	, ! taux
     .		memory( 1, mr_blk1( mres   ) ), mres	) ! ubwi
	GOTO 2000

* VBWI - wind stress effect on V
* ... calculation is identical to UBWI but with different components
 390	GOTO 380

* UCOE - explicit calculation of Coriolis effect on U
* ... also computes and stores VCOE ...
 400	cx_variable( cx_xtra ) = pvcoe
	CALL RE_ASSIGN_VARIABLE( mxtra, cx_xtra )
	temp_grid_size = MGRID_SIZE( mrl(1) )	! working space for "AIR"
	CALL GET_MEMORY( temp_grid_size,temp_start_blk,temp_nblks,status )
	IF ( status .NE. ferr_ok ) GOTO 5100
	CALL EXPL_CORIOLIS( 
     .		memory( 1, mr_blk1( mrl(1) ) ), mrl(1)	, ! u
     .		memory( 1, mr_blk1( mrl(2) ) ), mrl(2)	, ! v
     .		memory( 1, mr_blk1( mres   ) ), mres	, ! ucoe
     .		memory( 1, mr_blk1( mxtra  ) ), mxtra	) ! vcoe
	CALL FREE_MEMORY( temp_start_blk, temp_nblks )
	GOTO 1900

* VCOE - explicit calculation of Coriolis effect on V
* ... also computes and stores UCOE ...
 410	cx_variable( cx_xtra ) = pucoe
	CALL RE_ASSIGN_VARIABLE( mxtra, cx_xtra )
	temp_grid_size = MGRID_SIZE( mrl(1) )	! working space for "AIR"
	CALL GET_MEMORY( temp_grid_size,temp_start_blk,temp_nblks,status )
	IF ( status .NE. ferr_ok ) GOTO 5100
	CALL EXPL_CORIOLIS( 
     .		memory( 1, mr_blk1( mrl(1) ) ), mrl(1)	, ! u
     .		memory( 1, mr_blk1( mrl(2) ) ), mrl(2)	, ! v
     .		memory( 1, mr_blk1( mxtra  ) ), mxtra	, ! ucoe
     .		memory( 1, mr_blk1( mres   ) ), mres	) ! vcoe
	CALL FREE_MEMORY( temp_start_blk, temp_nblks )
	GOTO 1900

* UAD - total advection of zonal velocity
* ... uses identical code to QAD ...
 420	GOTO 90

* VAD - total advection of merional velocity
* ... uses identical code to QAD ...
 430	GOTO 90

* UDF - total diffusion of zonal velocity
* ... uses identical code to QDF ...
 440	GOTO 100

* VDF - total diffusion of merional velocity
* ... uses identical code to QDF ...
 450	GOTO 100

* DUDT - centered time derivative of U
 460	IF ( MR_DIM_LEN(t_dim, mrl(1)).LE.2 ) CALL ERRMSG
     .	( ferr_limits, status, 'insufficient T axis data for '
     .				//VAR_CODE(cat_calc_var,var), *5200 )
	CALL T_DERIV( 
     .	  memory( 1, mr_blk1(mrl(1)) ) , mrl(1),		! u
     .	  memory( 1, mr_blk1( mres   )), mres  ,		! difference
     .    dt, reg )
* ... convert per hour to per month
	CALL MULT_GRID( grid_size, memory(1, mr_blk1(mres)), 730. )
* ... store information about accuracy of calculation
	IF ( reg ) THEN
	   mr_trans_arg( t_dim, mres ) = dt / 2.
	ELSE
	   mr_trans_arg( t_dim, mres ) = bad_val4
	ENDIF
	GOTO 2000	

* DVDT - time derivative of V
* ... uses identical code to DUDT ...
 470	GOTO 460

* DWDT - time derivative of W
* ... uses identical code to DUDT ...
 480	GOTO 460

* DTDT - time derivative of TEMP
* ... uses identical code to DUDT ...
 490	GOTO 460

* UBPS - monthly dU due to surface pressure effects
* ... UBPS = DUDT - ( UDF + UAD + UCOE + UBPI )
* ... we will use a kludge to get around 3D components to a 2D result
 500    mr_lo_ss( mres, z_dim ) = 1	! kludge - make it appear 3D
	mr_hi_ss( mres, z_dim ) = 1	! kludge
	CALL COPY_GRID(
     .	  memory( 1, mr_blk1(mrl(1)) ) , mrl(1) , 		! udf
     .	  memory( 1, mr_blk1( mres   )), mres	  )		! udf
	CALL ADD_2_GRIDS(
     .	  memory( 1, mr_blk1(mrl(2)) ) , mrl(2) , 		! uad
     .	  memory( 1, mr_blk1( mres   )), mres	  )		! udf+uad
	CALL ADD_2_GRIDS(
     .	  memory( 1, mr_blk1(mrl(3)) ) , mrl(3),		! ucoe
     .	  memory( 1, mr_blk1( mres   )), mres 	 )		! udf+uad+ucoe
	CALL ADD_2_GRIDS(
     .	  memory( 1, mr_blk1(mrl(4)) ) , mrl(4),		! ubpi
     .	  memory( 1, mr_blk1( mres   )), mres 	 )		! udf+uad+u...
	CALL SUB_2_GRIDS(
     .	  memory( 1, mr_blk1(mrl(5)) ) , mrl(5),		! dudt 
     .	  memory( 1, mr_blk1( mres   )), mres	,		! udf+uad+u...
     .	  memory( 1, mr_blk1( mres   )), mres	 )		! ubps
	mr_lo_ss( mres, z_dim ) = unspecified_int4	! kludge - back to 2D
	mr_hi_ss( mres, z_dim ) = unspecified_int4	! kludge
* ... store information about accuracy of derivative in calculation
	mr_trans_arg( t_dim, mres ) = 	mr_trans_arg( t_dim, mrl(5) )
	GOTO 2000	

* VBPS - monthly dV due to surface pressure effects
* ... uses identical code to UBPS ...
 510	GOTO 500

* SALI - salinity in parts per thousand
* ... from "SALT" in (PPT-35)/1000
 520	CALL COPY_GRID(
     .	  memory( 1, mr_blk1(mrl(1)) ) , mrl(1),		! salt
     .	  memory( 1, mr_blk1( mres   )), mres  )
	CALL MULT_GRID   ( grid_size, memory(1, mr_blk1( mres )), 1000. )
	CALL ADD_C_2_GRID( grid_size, memory(1, mr_blk1( mres )), 35. )
	GOTO 2000

* DENS - density in gm-cm**3
* ... from "RHO" which is "normalized"
 530	CALL KNUDSEN(
     .	  memory( 1, mr_blk1(mrl(1)) ) , mrl(1),		! temp
     .	  memory( 1, mr_blk1(mrl(2)) ) , mrl(2),		! salt
     .	  memory( 1, mr_blk1( mres   )), mres	 )		! dens

	GOTO 2000

* UBP  - U change caused by baroclinic and barotropic pressure
* ... special routine used to replicate 2D UBPS into 3D result
 540	CALL COPY_2D_TO_3D(
     .	  memory( 1, mr_blk1(mrl(2)) ) , mrl(2),		! ubps
     .	  memory( 1, mr_blk1( mres   )), mres	 )		! ubp
	CALL ADD_2_GRIDS(
     .	  memory( 1, mr_blk1(mrl(1)) ) , mrl(1),		! ubpi
     .	  memory( 1, mr_blk1( mres   )), mres	 )		! ubp
	GOTO 2000

* VBP  - V change caused by baroclinic and barotropic pressure
* ... uses identical code to UBP ...
 550	GOTO 540

* CMSK - convective adjustment mask
 560	CALL CONV_ADJ_MASK (
     .		memory( 1, mr_blk1( mrl(1) ) ), mrl(1)	, ! temp
     .		memory( 1, mr_blk1( mrl(2) ) ), mrl(2)	, ! salt
     .		memory( 1, mr_blk1( mres   ) ), mres	) ! w_u
	GOTO 2000

* W_T - vertical velocity computed on the ts grid (still on w grid vertically)
 570	CALL W_ON_TS (
     .		memory( 1, mr_blk1( mrl(1) ) ), mrl(1)	, ! u
     .		memory( 1, mr_blk1( mrl(2) ) ), mrl(2)	, ! v
     .		memory( 1, mr_blk1( mres   ) ), mres	) ! w_u
	GOTO 2000

* CAIR - climatological air temperature regridded from monthly data
 580	CALL COPY_GRID(
     .	  memory( 1, mr_blk1(mrl(1)) ) , mrl(1),	! regridded airt
     .	  memory( 1, mr_blk1( mres   )), mres	)	! cair
	GOTO 2000

* AIR - air temperature taylored to data set
 590	CALL AIR_TEMP( 
     .		memory( 1, mr_blk1( mrl(1) ) ), mrl(1)	, ! sst
     .		memory( 1, mr_blk1( mrl(2) ) ), mrl(2)	, ! cair
     .		memory( 1, mr_blk1( mres   ) ), mres	) ! air
	GOTO 2000

* ----------------------------------------------------------------------
* 2 results were computed.  Catalog the extra one
 1900	mr_protected( mxtra ) = mr_not_protected
	CALL MR_NOT_IN_USE( mxtra )
	cx_stack_ptr = cx_stack_ptr - 1

* free cx_lrg and cx_xtra space
 2000	cx_stack_ptr = cx_stack_ptr - 1

* flag that the components are no longer in use
	DO 2010 icomp = 1, ncomp
 2010	CALL MR_NOT_IN_USE( mrl(icomp) )

* success
	RETURN

* error exit
 5000	RETURN 
 5100	CALL DELETE_VARIABLE( mxtra )
 5200	CALL DELETE_VARIABLE( mres )
	RETURN
	END
