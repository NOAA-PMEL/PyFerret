	BLOCK DATA XCALC_VARS_DATA

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

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  7/21/89 - extracted from XVARIABLES_DTA with extensive changes
*	10/12/89 - added cvar_num_results data
*	 2/26/90 - CAIR requires only one component in the 4D version 2.00
* Linux Port 1/97 *kob* - Added a preprocessor include for tmap_dset.parm
*			  because it needed a preprocessor.

#ifdef unix
#include "tmap_format/tmap_dset.parm"
	include 'tmap_format/tmap_dims.parm'
	include 'tmap_format/xtm_grid.cmn_text'	! .cmn_text has an external
!	external xgt_grid_data	! xtm_grid --> xgt  (DEC f77 3.2 not allowed)
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/instance_table.parm'
	include 'ferret_cmn/xcalc_vars.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN_TEXT'	! .CMN has an EXTERNAL
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:INSTANCE_TABLE.PARM'
	INCLUDE 'FERRET_CMN:XCALC_VARS.CMN'
#endif

	INTEGER	i

* local parameter declarations:
	INTEGER     ptstep_1hr_b4, pcair_special, pairt_special, 
     .              ptemp_special
	PARAMETER ( ptstep_1hr_b4 = pcv_hourly_0 - 1,
     .		    pcair_special = pcair  + pcv_spec_flg,
     .		    pairt_special = pairt + pcv_spec_flg,
     .		    ptemp_special = ptemp + pcv_spec_flg )

* number of calculatable variables defined in this BLOCK DATA
	DATA num_calc_vars	/ 59 /

* computable variables pre-defined in program
	DATA cvar_primary_name	( pqcdz )
     .				/'VERTICAL HEAT DIFFUSION COEFFICIENT'/
	DATA cvar_name_code	( pqcdz ) / 'QCDZ'	/
	DATA cvar_unit_code	( pqcdz ) / pudiffusion /
	DATA cvar_frame_start	( pqcdz ) / to_qcdz	/
	DATA cvar_frame_length	( pqcdz ) / qcdz_frame	/
	DATA cvar_num_results	( pqcdz ) / 2	/	! also PCDZ
	DATA ( cvar_axes(i,pqcdz),i=1,3 ) / mxt, myt, mzw/

	DATA cvar_primary_name	( ppcdz )
     .				/'VERTICAL MOMENTUM DIFFUSION COEFFICIENT'/
	DATA cvar_name_code	( ppcdz ) / 'PCDZ'	/
	DATA cvar_unit_code	( ppcdz ) / pudiffusion /
	DATA cvar_frame_start	( ppcdz ) / to_pcdz	/
	DATA cvar_frame_length	( ppcdz ) / pcdz_frame	/
	DATA cvar_num_results	( ppcdz ) / 2	/	! also QCDZ
	DATA ( cvar_axes(i,ppcdz),i=1,3 ) / mxu, myu, mzw /

	DATA cvar_primary_name	( prho0 ) / 'DENSITY REFERENCED TO SURFACE'/
	DATA cvar_name_code	( prho0 ) / 'RHO0'	/
	DATA cvar_unit_code	( prho0 ) / punot_app	/
	DATA cvar_frame_start	( prho0 ) / to_rho0	/
	DATA cvar_frame_length	( prho0 ) / rho0_frame	/
	DATA cvar_num_results	( prho0 ) / 1	/
	DATA ( cvar_axes(i,prho0),i=1,3 ) / mxt, myt, mzt /

	DATA cvar_primary_name	( pqdfz ) 
     .				/ 'VERTICAL HEAT DIFFUSION' /
	DATA cvar_name_code	( pqdfz ) / 'QDFZ'	/
	DATA cvar_unit_code	( pqdfz ) / putemp_dt	/
	DATA cvar_frame_start	( pqdfz ) / to_qdfz	/
	DATA cvar_frame_length	( pqdfz ) / qdfz_frame	/
	DATA cvar_num_results	( pqdfz ) / 1	/
	DATA ( cvar_axes(i,pqdfz),i=1,3 ) / mxt, myt, mzt /

	DATA cvar_primary_name	( pqdfh ) / 'HORIZONTAL HEAT DIFFUSION' /
	DATA cvar_name_code	( pqdfh ) / 'QDFH'	/
	DATA cvar_unit_code	( pqdfh ) / putemp_dt	/
	DATA cvar_frame_start	( pqdfh ) / to_qdfh	/
	DATA cvar_frame_length	( pqdfh ) / qdfh_frame	/
	DATA cvar_num_results	( pqdfh ) / 1	/
	DATA ( cvar_axes(i,pqdfh),i=1,3 ) / mxt, myt, mzt /

	DATA cvar_primary_name	( pqadx ) / 'ZONAL HEAT ADVECTION' /
	DATA cvar_name_code	( pqadx ) / 'QADX'	/
	DATA cvar_unit_code	( pqadx ) / putemp_dt	/
	DATA cvar_frame_start	( pqadx ) / to_qadx	/
	DATA cvar_frame_length	( pqadx ) / qadx_frame	/
	DATA cvar_num_results	( pqadx ) / 1	/
	DATA ( cvar_axes(i,pqadx),i=1,3 ) / mxt, myt, mzt /

	DATA cvar_primary_name	( pqady ) / 'MERIDIONAL HEAT ADVECTION' /
	DATA cvar_name_code	( pqady ) / 'QADY'	/
	DATA cvar_unit_code	( pqady ) / putemp_dt	/
	DATA cvar_frame_start	( pqady ) / to_qady	/
	DATA cvar_frame_length	( pqady ) / qady_frame	/
	DATA cvar_num_results	( pqady ) / 1	/
	DATA ( cvar_axes(i,pqady),i=1,3 ) / mxt, myt, mzt /

	DATA cvar_primary_name	( pqadz ) / 'VERTICAL HEAT ADVECTION' /
	DATA cvar_name_code	( pqadz ) / 'QADZ'	/
	DATA cvar_unit_code	( pqadz ) / putemp_dt	/
	DATA cvar_frame_start	( pqadz ) / to_qadz	/
	DATA cvar_frame_length	( pqadz ) / qadz_frame	/
	DATA cvar_num_results	( pqadz ) / 1	/
	DATA ( cvar_axes(i,pqadz),i=1,3 ) / mxt, myt, mzt /

	DATA cvar_primary_name	( pqad  ) / 'TOTAL HEAT ADVECTION' /
	DATA cvar_name_code	( pqad  ) / 'QAD '	/
	DATA cvar_unit_code	( pqad  ) / putemp_dt	/
	DATA cvar_frame_start	( pqad  ) / to_qad 	/
	DATA cvar_frame_length	( pqad  ) / qad _frame	/
	DATA cvar_num_results	( pqad  ) / 1	/
	DATA ( cvar_axes(i,pqad ),i=1,3 ) / mxt, myt, mzt /

	DATA cvar_primary_name	( pqdf  ) / 'TOTAL HEAT DIFFUSION' /
	DATA cvar_name_code	( pqdf  ) / 'QDF '	/
	DATA cvar_unit_code	( pqdf  ) / putemp_dt	/
	DATA cvar_frame_start	( pqdf  ) / to_qdf 	/
	DATA cvar_frame_length	( pqdf  ) / qdf _frame	/
	DATA cvar_num_results	( pqdf  ) / 1	/
	DATA ( cvar_axes(i,pqdf ),i=1,3 ) / mxt, myt, mzt /

	DATA cvar_primary_name	( pqeva ) / 'EVAPORATIVE HEAT' /
	DATA cvar_name_code	( pqeva ) / 'QEVA'	/
	DATA cvar_unit_code	( pqeva ) / puheat_flux	/
	DATA cvar_frame_start	( pqeva ) / to_qeva	/
	DATA cvar_frame_length	( pqeva ) / qeva_frame	/
	DATA cvar_num_results	( pqeva ) / 2	/	! also computes QSEN
	DATA ( cvar_axes(i,pqeva),i=1,3 ) / mxt, myt, mnormal /

	DATA cvar_primary_name	( ptau  ) / 'MAGNITUDE OF WIND STRESS' /
	DATA cvar_name_code	( ptau  ) / 'TAU '	/
	DATA cvar_unit_code	( ptau  ) / pupressure	/
	DATA cvar_frame_start	( ptau  ) / to_tau 	/
	DATA cvar_frame_length	( ptau  ) / tau _frame	/
	DATA cvar_num_results	( ptau  ) / 1	/
	DATA ( cvar_axes(i,ptau ),i=1,3 ) / mxt, myt, mnormal /

	DATA cvar_primary_name	( pqsen ) / 'SENSIBLE HEAT'/
	DATA cvar_name_code	( pqsen ) / 'QSEN'	/
	DATA cvar_unit_code	( pqsen ) / puheat_flux	/
	DATA cvar_frame_start	( pqsen ) / to_qsen	/
	DATA cvar_frame_length	( pqsen ) / qsen_frame	/
	DATA cvar_num_results	( pqsen ) / 2	/	! also computes QEVA
	DATA ( cvar_axes(i,pqsen),i=1,3 ) / mxt, myt, mnormal /

	DATA cvar_primary_name	( pqrad ) / 'RADIATIVE HEAT' /
	DATA cvar_name_code	( pqrad ) / 'QRAD'	/
	DATA cvar_unit_code	( pqrad ) / puheat_flux	/
	DATA cvar_frame_start	( pqrad ) / to_qrad	/
	DATA cvar_frame_length	( pqrad ) / qrad_frame	/
	DATA cvar_num_results	( pqrad ) / 1	/
	DATA ( cvar_axes(i,pqrad),i=1,3 ) / mxt, myt, mnormal /

	DATA cvar_primary_name	( pqflx ) / 'SURFACE HEAT FLUX' /
	DATA cvar_name_code	( pqflx ) / 'QFLX'	/
	DATA cvar_unit_code	( pqflx ) / puheat_flux	/
	DATA cvar_frame_start	( pqflx ) / to_qflx	/
	DATA cvar_frame_length	( pqflx ) / qflx_frame	/
	DATA cvar_num_results	( pqflx ) / 1	/
	DATA ( cvar_axes(i,pqflx),i=1,3 ) / mxt, myt, mnormal /

	DATA cvar_primary_name	( prho  ) / 'NORMALIZED DENSITY'/
	DATA cvar_name_code	( prho  ) / 'RHO'	/
	DATA cvar_unit_code	( prho  ) / punot_app	/
	DATA cvar_frame_start	( prho  ) / to_rho	/
	DATA cvar_frame_length	( prho  ) / rho_frame	/
	DATA cvar_num_results	( prho  ) / 1	/
	DATA ( cvar_axes(i,prho ),i=1,3 ) / mxt, myt, mzt /

	DATA cvar_primary_name	( pdpdx ) 
     .				/ 'd/dX( BAROCLINIC PRESSURE )'/
	DATA cvar_name_code	( pdpdx ) / 'DPDX'	/
	DATA cvar_unit_code	( pdpdx ) / pupres_dx	/
	DATA cvar_frame_start	( pdpdx ) / to_dpdx	/
	DATA cvar_frame_length	( pdpdx ) / dpdx_frame	/
	DATA cvar_num_results	( pdpdx ) / 2	/	! also DPDY
	DATA ( cvar_axes(i,pdpdx),i=1,3 ) / mxu, myu, mzt /

	DATA cvar_primary_name	( pdpdy ) 
     .				/ 'd/dY( BAROCLINIC PRESSURE )'/
	DATA cvar_name_code	( pdpdy ) / 'DPDY'	/
	DATA cvar_unit_code	( pdpdy ) / pupres_dx	/
	DATA cvar_frame_start	( pdpdy ) / to_dpdy	/
	DATA cvar_frame_length	( pdpdy ) / dpdy_frame	/
	DATA cvar_num_results	( pdpdy ) / 2	/	! also DPDX
	DATA ( cvar_axes(i,pdpdy),i=1,3 ) / mxu, myu, mzt /

	DATA cvar_primary_name	( pw_u  ) 
     .				 / 'VERTICAL VELOCITY COMPUTED ON UV GRID' /
	DATA cvar_name_code	( pw_u  ) / 'W_U '	/
	DATA cvar_unit_code	( pw_u  ) / puvelocity	/
	DATA cvar_frame_start	( pw_u  ) / to_w_u	/
	DATA cvar_frame_length	( pw_u  ) / w_u_frame	/
	DATA cvar_num_results	( pw_u  ) / 1	/
	DATA ( cvar_axes(i,pw_u ),i=1,3 ) / mxu, myu, mzw /

	DATA cvar_primary_name	( puavz ) 
     .				 / 'VERTICALLY AVERAGED ZONAL VELOCITY'	/
	DATA cvar_name_code	( puavz ) / 'UAVZ'	/
	DATA cvar_unit_code	( puavz ) / puvelocity	/
	DATA cvar_frame_start	( puavz ) / to_uavz	/
	DATA cvar_frame_length	( puavz ) / uavz_frame	/
	DATA cvar_num_results	( puavz ) / 2	/	! also VAVZ
	DATA ( cvar_axes(i,puavz),i=1,3 ) / mxu, myu, mnormal /

	DATA cvar_primary_name	( pvavz ) 
     .				 / 'VERTICALLY AVERAGED MERIDIONAL VELOCITY' /
	DATA cvar_name_code	( pvavz ) / 'VAVZ'	/
	DATA cvar_unit_code	( pvavz ) / puvelocity	/
	DATA cvar_frame_start	( pvavz ) / to_vavz	/
	DATA cvar_frame_length	( pvavz ) / vavz_frame	/
	DATA cvar_num_results	( pvavz ) / 2	/	! also UAVZ
	DATA ( cvar_axes(i,pvavz),i=1,3 ) / mxu, myu, mnormal /

	DATA cvar_primary_name	( pfuw  ) 
     .					/ 'ZONAL ADVECTIVE COEFFICIENT, FUW' /
	DATA cvar_name_code	( pfuw  ) / 'FUW ' 	/
	DATA cvar_unit_code	( pfuw  ) / puvelocity	/
	DATA cvar_frame_start	( pfuw  ) / to_fuw 	/
	DATA cvar_frame_length	( pfuw  ) / fuw_frame	/
	DATA cvar_num_results	( pfuw  ) / 1	/
	DATA ( cvar_axes(i,pfuw ),i=1,3 ) / mxu, myu, mzt /

	DATA cvar_primary_name	( pfvn  ) 
     .				/ 'MERIDIONAL ADVECTIVE COEFFICIENT, FVN' /
	DATA cvar_name_code	( pfvn  ) / 'FVN '	/
	DATA cvar_unit_code	( pfvn  ) / puvelocity	/
	DATA cvar_frame_start	( pfvn  ) / to_fvn 	/
	DATA cvar_frame_length	( pfvn  ) / fvn_frame	/
	DATA cvar_num_results	( pfvn  ) / 1	/
	DATA ( cvar_axes(i,pfvn ),i=1,3 ) / mxu, myu, mzt /

	DATA cvar_primary_name	( puadx ) / 'ZONAL ADVECTION OF U' /
	DATA cvar_name_code	( puadx ) / 'UADX'	/
	DATA cvar_unit_code	( puadx ) / puvel_dt	/
	DATA cvar_frame_start	( puadx ) / to_uadx	/
	DATA cvar_frame_length	( puadx ) / uadx_frame	/
	DATA cvar_num_results	( puadx ) / 1	/
	DATA ( cvar_axes(i,puadx),i=1,3 ) / mxu, myu, mzt /

	DATA cvar_primary_name	( puady ) / 'MERIDIONAL ADVECTION OF U' /
	DATA cvar_name_code	( puady ) / 'UADY'	/
	DATA cvar_unit_code	( puady ) / puvel_dt	/
	DATA cvar_frame_start	( puady ) / to_uady	/
	DATA cvar_frame_length	( puady ) / uady_frame	/
	DATA cvar_num_results	( puady ) / 1	/
	DATA ( cvar_axes(i,puady),i=1,3 ) / mxu, myu, mzt /

	DATA cvar_primary_name	( puadz ) / 'VERTICAL ADVECTION OF U' /
	DATA cvar_name_code	( puadz ) / 'UADZ'	/
	DATA cvar_unit_code	( puadz ) / puvel_dt	/
	DATA cvar_frame_start	( puadz ) / to_uadz	/
	DATA cvar_frame_length	( puadz ) / uadz_frame	/
	DATA cvar_num_results	( puadz ) / 1	/
	DATA ( cvar_axes(i,puadz),i=1,3 ) / mxu, myu, mzt /

	DATA cvar_primary_name	( pvadx ) / 'ZONAL ADVECTION OF V' /
	DATA cvar_name_code	( pvadx ) / 'VADX'	/
	DATA cvar_unit_code	( pvadx ) / puvel_dt	/
	DATA cvar_frame_start	( pvadx ) / to_vadx	/
	DATA cvar_frame_length	( pvadx ) / vadx_frame	/
	DATA cvar_num_results	( pvadx ) / 1	/
	DATA ( cvar_axes(i,pvadx),i=1,3 ) / mxu, myu, mzt /

	DATA cvar_primary_name	( pvady ) / 'MERIDIONAL ADVECTION OF V' /
	DATA cvar_name_code	( pvady ) / 'VADY'	/
	DATA cvar_unit_code	( pvady ) / puvel_dt	/
	DATA cvar_frame_start	( pvady ) / to_vady	/
	DATA cvar_frame_length	( pvady ) / vady_frame	/
	DATA cvar_num_results	( pvady ) / 1	/
	DATA ( cvar_axes(i,pvady),i=1,3 ) / mxu, myu, mzt /

	DATA cvar_primary_name	( pvadz ) / 'VERTICAL ADVECTION OF V' /
	DATA cvar_name_code	( pvadz ) / 'VADZ'	/
	DATA cvar_unit_code	( pvadz ) / puvel_dt	/
	DATA cvar_frame_start	( pvadz ) / to_vadz	/
	DATA cvar_frame_length	( pvadz ) / vadz_frame	/
	DATA cvar_num_results	( pvadz ) / 1	/
	DATA ( cvar_axes(i,pvadz),i=1,3 ) / mxu, myu, mzt /

	DATA cvar_primary_name	( pudfh ) / 'HORIZONTAL DIFFUSION OF U' /
	DATA cvar_name_code	( pudfh ) / 'UDFH'	/
	DATA cvar_unit_code	( pudfh ) / puvel_dt	/
	DATA cvar_frame_start	( pudfh ) / to_udfh	/
	DATA cvar_frame_length	( pudfh ) / udfh_frame	/
	DATA cvar_num_results	( pudfh ) / 1	/
	DATA ( cvar_axes(i,pudfh),i=1,3 ) / mxu, myu, mzt /

	DATA cvar_primary_name	( pudfz ) / 'VERTICAL DIFFUSION OF U' /
	DATA cvar_name_code	( pudfz ) / 'UDFZ'	/
	DATA cvar_unit_code	( pudfz ) / puvel_dt	/
	DATA cvar_frame_start	( pudfz ) / to_udfz	/
	DATA cvar_frame_length	( pudfz ) / udfz_frame	/
	DATA cvar_num_results	( pudfz ) / 1	/
	DATA ( cvar_axes(i,pudfz),i=1,3 ) / mxu, myu, mzt /

	DATA cvar_primary_name	( pvdfh ) / 'HORIZONTAL DIFFUSION OF V' /
	DATA cvar_name_code	( pvdfh ) / 'VDFH'	/
	DATA cvar_unit_code	( pvdfh ) / puvel_dt	/
	DATA cvar_frame_start	( pvdfh ) / to_vdfh	/
	DATA cvar_frame_length	( pvdfh ) / vdfh_frame	/
	DATA cvar_num_results	( pvdfh ) / 1	/
	DATA ( cvar_axes(i,pvdfh),i=1,3 ) / mxu, myu, mzt /

	DATA cvar_primary_name	( pvdfz ) / 'VERTICAL DIFFUSION OF V' /
	DATA cvar_name_code	( pvdfz ) / 'VDFZ'	/
	DATA cvar_unit_code	( pvdfz ) / puvel_dt	/
	DATA cvar_frame_start	( pvdfz ) / to_vdfz	/
	DATA cvar_frame_length	( pvdfz ) / vdfz_frame	/
	DATA cvar_num_results	( pvdfz ) / 1	/
	DATA ( cvar_axes(i,pvdfz),i=1,3 ) / mxu, myu, mzt /

	DATA cvar_primary_name	( pucoi ) 
     .				/ 'U DUE TO IMPLICIT CORIOLIS CORRECTION' /
	DATA cvar_name_code	( pucoi ) / 'UCOI'	/
	DATA cvar_unit_code	( pucoi ) / puvel_dt	/
	DATA cvar_frame_start	( pucoi ) / to_ucoi	/
	DATA cvar_frame_length	( pucoi ) / ucoi_frame	/
	DATA cvar_num_results	( pucoi ) / -9999	/
	DATA ( cvar_axes(i,pucoi),i=1,3 ) / mxu, myu, mzt /

	DATA cvar_primary_name	( pvcoi )  
     .				/ 'V DUE TO IMPLICIT CORIOLIS CORRECTION' /
	DATA cvar_name_code	( pvcoi ) / 'VCOI'	/
	DATA cvar_unit_code	( pvcoi ) / puvel_dt	/
	DATA cvar_frame_start	( pvcoi ) / to_vcoi	/
	DATA cvar_frame_length	( pvcoi ) / vcoi_frame	/
	DATA cvar_num_results	( pvcoi ) / -9999	/
	DATA ( cvar_axes(i,pvcoi),i=1,3 ) / mxu, myu, mzt /

	DATA cvar_primary_name	( pubpi ) 
     .				/ 'U DUE TO BAROCLINIC PRESSURE EFFECTS' /
	DATA cvar_name_code	( pubpi ) / 'UBPI'	/
	DATA cvar_unit_code	( pubpi ) / puvel_dt	/
	DATA cvar_frame_start	( pubpi ) / to_ubpi	/
	DATA cvar_frame_length	( pubpi ) / ubpi_frame	/
	DATA cvar_num_results	( pubpi ) / 1	/
	DATA ( cvar_axes(i,pubpi),i=1,3 ) / mxu, myu, mzt /

	DATA cvar_primary_name	( pvbpi ) 
     .				/ 'V DUE TO BAROCLINIC PRESSURE EFFECTS' /
	DATA cvar_name_code	( pvbpi ) / 'VBPI'	/
	DATA cvar_unit_code	( pvbpi ) / puvel_dt	/
	DATA cvar_frame_start	( pvbpi ) / to_vbpi	/
	DATA cvar_frame_length	( pvbpi ) / vbpi_frame	/
	DATA cvar_num_results	( pvbpi ) / 1	/
	DATA ( cvar_axes(i,pvbpi),i=1,3 ) / mxu, myu, mzt /

	DATA cvar_primary_name	( pubwi ) / 'U DUE TO WIND STRESS EFFECTS' /
	DATA cvar_name_code	( pubwi ) / 'UBWI'	/
	DATA cvar_unit_code	( pubwi ) / puvel_dt	/
	DATA cvar_frame_start	( pubwi ) / to_ubwi	/
	DATA cvar_frame_length	( pubwi ) / ubwi_frame	/
	DATA cvar_num_results	( pubwi ) / 1	/
	DATA ( cvar_axes(i,pubwi),i=1,3 ) / mxu, myu, mnormal /

	DATA cvar_primary_name	( pvbwi ) / 'V DUE TO WIND STRESS EFFECTS' /
	DATA cvar_name_code	( pvbwi ) / 'VBWI'	/
	DATA cvar_unit_code	( pvbwi ) / puvel_dt	/
	DATA cvar_frame_start	( pvbwi ) / to_vbwi	/
	DATA cvar_frame_length	( pvbwi ) / vbwi_frame	/
	DATA cvar_num_results	( pvbwi ) / 1	/
	DATA ( cvar_axes(i,pvbwi),i=1,3 ) / mxu, myu, mnormal /

	DATA cvar_primary_name	( pucoe ) 
     .				/ 'U DUE TO EXPLICIT CORIOLIS CALCULATION' /
	DATA cvar_name_code	( pucoe ) / 'UCOE'	/
	DATA cvar_unit_code	( pucoe ) / puvel_dt	/
	DATA cvar_frame_start	( pucoe ) / to_ucoe	/
	DATA cvar_frame_length	( pucoe ) / ucoe_frame	/
	DATA cvar_num_results	( pucoe ) / 2	/	! also VCOE
	DATA ( cvar_axes(i,pucoe),i=1,3 ) / mxu, myu, mzt /

	DATA cvar_primary_name	( pvcoe ) 
     .				/ 'V DUE TO EXPLICIT CORIOLIS CALCULATION' /
	DATA cvar_name_code	( pvcoe ) / 'VCOE'	/
	DATA cvar_unit_code	( pvcoe ) / puvel_dt	/
	DATA cvar_frame_start	( pvcoe ) / to_vcoe	/
	DATA cvar_frame_length	( pvcoe ) / vcoe_frame	/
	DATA cvar_num_results	( pvcoe ) / 2	/	! also UCOE
	DATA ( cvar_axes(i,pvcoe),i=1,3 ) / mxu, myu, mzt /

	DATA cvar_primary_name	( puad  ) / 'TOTAL ADVECTION OF U' /
	DATA cvar_name_code	( puad  ) / 'UAD '	/
	DATA cvar_unit_code	( puad  ) / puvel_dt	/
	DATA cvar_frame_start	( puad  ) / to_uad 	/
	DATA cvar_frame_length	( puad  ) / uad _frame	/
	DATA cvar_num_results	( puad  ) / 1	/
	DATA ( cvar_axes(i,puad ),i=1,3 ) / mxu, myu, mzt /

	DATA cvar_primary_name	( pvad  ) / 'TOTAL ADVECTION OF V' /
	DATA cvar_name_code	( pvad  ) / 'VAD '	/
	DATA cvar_unit_code	( pvad  ) / puvel_dt	/
	DATA cvar_frame_start	( pvad  ) / to_vad 	/
	DATA cvar_frame_length	( pvad  ) / vad _frame	/
	DATA cvar_num_results	( pvad  ) / 1	/
	DATA ( cvar_axes(i,pvad ),i=1,3 ) / mxu, myu, mzt /

	DATA cvar_primary_name	( pudf  ) / 'TOTAL DIFFUSION OF U' /
	DATA cvar_name_code	( pudf  ) / 'UDF '	/
	DATA cvar_unit_code	( pudf  ) / puvel_dt	/
	DATA cvar_frame_start	( pudf  ) / to_udf 	/
	DATA cvar_frame_length	( pudf  ) / udf _frame	/
	DATA cvar_num_results	( pudf  ) / 1	/
	DATA ( cvar_axes(i,pudf ),i=1,3 ) / mxu, myu, mzt /

	DATA cvar_primary_name	( pvdf  ) / 'TOTAL DIFFUSION OF V' /
	DATA cvar_name_code	( pvdf  ) / 'VDF '	/
	DATA cvar_unit_code	( pvdf  ) / puvel_dt	/
	DATA cvar_frame_start	( pvdf  ) / to_vdf 	/
	DATA cvar_frame_length	( pvdf  ) / vdf _frame	/
	DATA cvar_num_results	( pvdf  ) / 1	/
	DATA ( cvar_axes(i,pvdf ),i=1,3 ) / mxu, myu, mzt /

	DATA cvar_primary_name	( pdudt ) / 'd/dt(U)'	/
	DATA cvar_name_code	( pdudt ) / 'DUDT'	/
	DATA cvar_unit_code	( pdudt ) / puvel_dt	/
	DATA cvar_frame_start	( pdudt ) / to_dudt	/
	DATA cvar_frame_length	( pdudt ) / dudt_frame	/
	DATA cvar_num_results	( pdudt ) / 1	/
	DATA ( cvar_axes(i,pdudt),i=1,3 ) / mxu, myu, mzt /

	DATA cvar_primary_name	( pdvdt ) / 'd/dt(V)'/
	DATA cvar_name_code	( pdvdt ) / 'DVDT'	/
	DATA cvar_unit_code	( pdvdt ) / puvel_dt	/
	DATA cvar_frame_start	( pdvdt ) / to_dvdt	/
	DATA cvar_frame_length	( pdvdt ) / dvdt_frame	/
	DATA cvar_num_results	( pdvdt ) / 1	/
	DATA ( cvar_axes(i,pdvdt),i=1,3 ) / mxu, myu, mzt /

	DATA cvar_primary_name	( pdwdt ) / 'd/dt(W)'/
	DATA cvar_name_code	( pdwdt ) / 'DWDT'	/
	DATA cvar_unit_code	( pdwdt ) / puvel_dt	/
	DATA cvar_frame_start	( pdwdt ) / to_dwdt	/
	DATA cvar_frame_length	( pdwdt ) / dwdt_frame	/
	DATA cvar_num_results	( pdwdt ) / 1	/
	DATA ( cvar_axes(i,pdwdt),i=1,3 ) / mxu, myu, mzw /

	DATA cvar_primary_name	( pdtdt ) / 'd/dt(TEMPERATURE)'/
	DATA cvar_name_code	( pdtdt ) / 'DTDT'	/
	DATA cvar_unit_code	( pdtdt ) / putemp_dt	/
	DATA cvar_frame_start	( pdtdt ) / to_dtdt	/
	DATA cvar_frame_length	( pdtdt ) / dtdt_frame	/
	DATA cvar_num_results	( pdtdt ) / 1	/
	DATA ( cvar_axes(i,pdtdt),i=1,3 ) / mxt, myt, mzt /

	DATA cvar_primary_name	( pubps ) 
     .				/ 'U DUE TO SURFACE PRESSURE EFFECTS' /
	DATA cvar_name_code	( pubps ) / 'UBPS'	/
	DATA cvar_unit_code	( pubps ) / puvel_dt	/
	DATA cvar_frame_start	( pubps ) / to_ubps	/
	DATA cvar_frame_length	( pubps ) / ubps_frame	/
	DATA cvar_num_results	( pubps ) / 1	/
	DATA ( cvar_axes(i,pubps),i=1,3 ) / mxu, myu, mnormal /

	DATA cvar_primary_name	( pvbps ) 
     .				/ 'V DUE TO SURFACE PRESSURE EFFECTS' /
	DATA cvar_name_code	( pvbps ) / 'VBPS'	/
	DATA cvar_unit_code	( pvbps ) / puvel_dt	/
	DATA cvar_frame_start	( pvbps ) / to_vbps	/
	DATA cvar_frame_length	( pvbps ) / vbps_frame	/
	DATA cvar_num_results	( pvbps ) / 1	/
	DATA ( cvar_axes(i,pvbps),i=1,3 ) / mxu, myu, mnormal /

	DATA cvar_primary_name	( psali ) 
     .				/ 'SALINITY'/
	DATA cvar_name_code	( psali ) / 'SALI'	/
	DATA cvar_unit_code	( psali ) / puppt	/
	DATA cvar_frame_start	( psali ) / to_sali	/
	DATA cvar_frame_length	( psali ) / sali_frame	/
	DATA cvar_num_results	( psali ) / 1	/
	DATA ( cvar_axes(i,psali),i=1,3 ) / mxt, myt, mzt /

	DATA cvar_primary_name	( pdens ) 
     .				/ '(DENSITY-1)*1000'		/
	DATA cvar_name_code	( pdens ) / 'DENS'	/
	DATA cvar_unit_code	( pdens ) / pudensity	/
	DATA cvar_frame_start	( pdens ) / to_dens	/
	DATA cvar_frame_length	( pdens ) / dens_frame	/
	DATA cvar_num_results	( pdens ) / 1	/
	DATA ( cvar_axes(i,pdens),i=1,3 ) / mxt, myt, mzt /

	DATA cvar_primary_name	( pubp  ) 
     .				/ 'U DUE TO TOTAL PRESSURE' /
	DATA cvar_name_code	( pubp  ) / 'UBP '	/
	DATA cvar_unit_code	( pubp  ) / puvel_dt	/
	DATA cvar_frame_start	( pubp  ) / to_ubp 	/
	DATA cvar_frame_length	( pubp  ) /  ubp_frame	/
	DATA cvar_num_results	( pubp  ) / 1	/
	DATA ( cvar_axes(i,pubp ),i=1,3 ) / mxu, myu, mzt /

	DATA cvar_primary_name	( pvbp  ) 
     .				/ 'V DUE TO TOTAL PRESSURE' /
	DATA cvar_name_code	( pvbp  ) / 'VBP '	/
	DATA cvar_unit_code	( pvbp  ) / puvel_dt	/
	DATA cvar_frame_start	( pvbp  ) / to_vbp 	/
	DATA cvar_frame_length	( pvbp  ) /  vbp_frame	/
	DATA cvar_num_results	( pvbp  ) / 1	/
	DATA ( cvar_axes(i,pvbp ),i=1,3 ) / mxu, myu, mzt /

	DATA cvar_primary_name	( pcmsk ) / 'CONVECTIVE ADJUSTMENT MASK'/
	DATA cvar_name_code	( pcmsk ) / 'CMSK'	/
	DATA cvar_unit_code	( pcmsk ) / punot_app	/
	DATA cvar_frame_start	( pcmsk ) / to_cmsk	/
	DATA cvar_frame_length	( pcmsk ) / cmsk_frame	/
	DATA cvar_num_results	( pcmsk ) / 1	/
	DATA ( cvar_axes(i,pcmsk),i=1,3 ) / mxt, myt, mzt /

	DATA cvar_primary_name	( pw_t  ) 
     .				/ 'VERTICAL VELOCITY COMPUTED ON TS GRID'/
	DATA cvar_name_code	( pw_t  ) / 'W_T '	/
	DATA cvar_unit_code	( pw_t  ) / puvelocity	/
	DATA cvar_frame_start	( pw_t  ) / to_w_t 	/
	DATA cvar_frame_length	( pw_t  ) /  w_t_frame	/
	DATA cvar_num_results	( pw_t  ) / 1	/
	DATA ( cvar_axes(i,pw_t ),i=1,3 ) / mxt, myt, mzw /

	DATA cvar_primary_name	( pcair ) / 'CLIMATOLOGICAL AIR TEMPERATURE'/
	DATA cvar_name_code	( pcair ) / 'CAIR'	/
	DATA cvar_unit_code	( pcair ) / putemperature /
	DATA cvar_frame_start	( pcair ) / to_cair	/
	DATA cvar_frame_length	( pcair ) / cair_frame	/
	DATA cvar_num_results	( pcair ) / 1	/
	DATA ( cvar_axes(i,pcair),i=1,3 ) / mxt, myt, mnormal /

	DATA cvar_primary_name	( pair  ) / 'AIR TEMPERATURE'/
	DATA cvar_name_code	( pair  ) / 'AIR '	/
	DATA cvar_unit_code	( pair  ) / putemperature /
	DATA cvar_frame_start	( pair  ) / to_air 	/
	DATA cvar_frame_length	( pair  ) / air_frame	/
	DATA cvar_num_results	( pair  ) / 1	/
	DATA ( cvar_axes(i,pair ),i=1,3 ) / mxt, myt, mnormal /

********************************************************************************
* variable instances needed to  compute cvar variables

	DATA cv_category (to_qcdz+1)/ cat_file_var	/
	DATA cv_variable (to_qcdz+1)/ pu		/
	DATA cv_lo_d1  (to_qcdz+1)/ 0		/
	DATA cv_hi_d1  (to_qcdz+1)/ 0		/
	DATA cv_lo_d2  (to_qcdz+1)/ 0		/
	DATA cv_hi_d2  (to_qcdz+1)/ 0		/
	DATA cv_lo_d3  (to_qcdz+1)/ 0		/
	DATA cv_hi_d3  (to_qcdz+1)/ 1		/
	DATA cv_lo_d4  (to_qcdz+1)/ 0		/
	DATA cv_hi_d4  (to_qcdz+1)/ 0		/

	DATA cv_category (to_qcdz+2)/ cat_file_var	/
	DATA cv_variable (to_qcdz+2)/ pv		/
	DATA cv_lo_d1  (to_qcdz+2)/ 0		/
	DATA cv_hi_d1  (to_qcdz+2)/ 0		/
	DATA cv_lo_d2  (to_qcdz+2)/ 0		/
	DATA cv_hi_d2  (to_qcdz+2)/ 0		/
	DATA cv_lo_d3  (to_qcdz+2)/ 0		/
	DATA cv_hi_d3  (to_qcdz+2)/ 1		/
	DATA cv_lo_d4  (to_qcdz+2)/ 0		/
	DATA cv_hi_d4  (to_qcdz+2)/ 0		/

	DATA cv_category (to_qcdz+3)/ cat_calc_var	/
	DATA cv_variable (to_qcdz+3)/ prho0 		/
	DATA cv_lo_d1  (to_qcdz+3)/ 0		/
	DATA cv_hi_d1  (to_qcdz+3)/ 0		/
	DATA cv_lo_d2  (to_qcdz+3)/ 0		/
	DATA cv_hi_d2  (to_qcdz+3)/ 0		/
	DATA cv_lo_d3  (to_qcdz+3)/ 0		/
	DATA cv_hi_d3  (to_qcdz+3)/ 1		/
	DATA cv_lo_d4  (to_qcdz+3)/ 0		/
	DATA cv_hi_d4  (to_qcdz+3)/ 0		/

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	
	DATA cv_category (to_pcdz+1)/ cat_file_var	/
	DATA cv_variable (to_pcdz+1)/ pu		/
	DATA cv_lo_d1  (to_pcdz+1)/ 0		/
	DATA cv_hi_d1  (to_pcdz+1)/ 0		/
	DATA cv_lo_d2  (to_pcdz+1)/ 0		/
	DATA cv_hi_d2  (to_pcdz+1)/ 0		/
	DATA cv_lo_d3  (to_pcdz+1)/ 0		/
	DATA cv_hi_d3  (to_pcdz+1)/ 1		/
	DATA cv_lo_d4  (to_pcdz+1)/ 0		/
	DATA cv_hi_d4  (to_pcdz+1)/ 0		/

	DATA cv_category (to_pcdz+2)/ cat_file_var	/
	DATA cv_variable (to_pcdz+2)/ pv		/
	DATA cv_lo_d1  (to_pcdz+2)/ 0		/
	DATA cv_hi_d1  (to_pcdz+2)/ 0		/
	DATA cv_lo_d2  (to_pcdz+2)/ 0		/
	DATA cv_hi_d2  (to_pcdz+2)/ 0		/
	DATA cv_lo_d3  (to_pcdz+2)/ 0		/
	DATA cv_hi_d3  (to_pcdz+2)/ 1		/
	DATA cv_lo_d4  (to_pcdz+2)/ 0		/
	DATA cv_hi_d4  (to_pcdz+2)/ 0		/

	DATA cv_category (to_pcdz+3)/ cat_calc_var	/
	DATA cv_variable (to_pcdz+3)/ prho0 		/
	DATA cv_lo_d1  (to_pcdz+3)/ 0		/
	DATA cv_hi_d1  (to_pcdz+3)/ 0		/
	DATA cv_lo_d2  (to_pcdz+3)/ 0		/
	DATA cv_hi_d2  (to_pcdz+3)/ 0		/
	DATA cv_lo_d3  (to_pcdz+3)/ 0		/
	DATA cv_hi_d3  (to_pcdz+3)/ 1		/
	DATA cv_lo_d4  (to_pcdz+3)/ 0		/
	DATA cv_hi_d4  (to_pcdz+3)/ 0		/

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	
	DATA cv_category (to_rho0+1)/ cat_file_var	/
	DATA cv_variable (to_rho0+1)/ ptemp		/
	DATA cv_lo_d1  (to_rho0+1)/ 0		/
	DATA cv_hi_d1  (to_rho0+1)/ 0		/
	DATA cv_lo_d2  (to_rho0+1)/ 0		/
	DATA cv_hi_d2  (to_rho0+1)/ 0		/
	DATA cv_lo_d3  (to_rho0+1)/ 0		/
	DATA cv_hi_d3  (to_rho0+1)/ 0		/
	DATA cv_lo_d4  (to_rho0+1)/ 0		/
	DATA cv_hi_d4  (to_rho0+1)/ 0		/
	
	DATA cv_category (to_rho0+2)/ cat_file_var	/
	DATA cv_variable (to_rho0+2)/ psalt		/
	DATA cv_lo_d1  (to_rho0+2)/ 0		/
	DATA cv_hi_d1  (to_rho0+2)/ 0		/
	DATA cv_lo_d2  (to_rho0+2)/ 0		/
	DATA cv_hi_d2  (to_rho0+2)/ 0		/
	DATA cv_lo_d3  (to_rho0+2)/ 0		/
	DATA cv_hi_d3  (to_rho0+2)/ 0		/
	DATA cv_lo_d4  (to_rho0+2)/ 0		/
	DATA cv_hi_d4  (to_rho0+2)/ 0		/

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	
	DATA cv_category (to_qdfz+1)/ cat_file_var	/
	DATA cv_variable (to_qdfz+1)/ ptemp		/
	DATA cv_lo_d1  (to_qdfz+1)/ 0		/
	DATA cv_hi_d1  (to_qdfz+1)/ 0		/
	DATA cv_lo_d2  (to_qdfz+1)/ 0		/
	DATA cv_hi_d2  (to_qdfz+1)/ 0		/
	DATA cv_lo_d3  (to_qdfz+1)/ -1		/
	DATA cv_hi_d3  (to_qdfz+1)/ +1		/
	DATA cv_lo_d4  (to_qdfz+1)/ ptstep_1hr_b4	/
	DATA cv_hi_d4  (to_qdfz+1)/ ptstep_1hr_b4	/

	DATA cv_category (to_qdfz+2)/ cat_calc_var	/
	DATA cv_variable (to_qdfz+2)/ pqcdz		/
	DATA cv_lo_d1  (to_qdfz+2)/ 0		/
	DATA cv_hi_d1  (to_qdfz+2)/ 0		/
	DATA cv_lo_d2  (to_qdfz+2)/ 0		/
	DATA cv_hi_d2  (to_qdfz+2)/ 0		/
	DATA cv_lo_d3  (to_qdfz+2)/ -1		/
	DATA cv_hi_d3  (to_qdfz+2)/ 0		/
	DATA cv_lo_d4  (to_qdfz+2)/ ptstep_1hr_b4	/
	DATA cv_hi_d4  (to_qdfz+2)/ ptstep_1hr_b4	/

	DATA cv_category (to_qdfz+3)/ cat_calc_var	/
	DATA cv_variable (to_qdfz+3)/ pqflx		/
	DATA cv_lo_d1  (to_qdfz+3)/ 0		/
	DATA cv_hi_d1  (to_qdfz+3)/ 0		/
	DATA cv_lo_d2  (to_qdfz+3)/ 0		/
	DATA cv_hi_d2  (to_qdfz+3)/ 0		/
	DATA cv_lo_d3  (to_qdfz+3)/ 0		/
	DATA cv_hi_d3  (to_qdfz+3)/ 0		/
	DATA cv_lo_d4  (to_qdfz+3)/ ptstep_1hr_b4	/
	DATA cv_hi_d4  (to_qdfz+3)/ ptstep_1hr_b4	/

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	
	DATA cv_category (to_qdfh+1)/ cat_file_var	/
	DATA cv_variable (to_qdfh+1)/ ptemp		/
	DATA cv_lo_d1  (to_qdfh+1)/ -1		/
	DATA cv_hi_d1  (to_qdfh+1)/ +1		/
	DATA cv_lo_d2  (to_qdfh+1)/ -1		/
	DATA cv_hi_d2  (to_qdfh+1)/ +1		/
	DATA cv_lo_d3  (to_qdfh+1)/ 0		/
	DATA cv_hi_d3  (to_qdfh+1)/ 0		/
	DATA cv_lo_d4  (to_qdfh+1)/ ptstep_1hr_b4	/
	DATA cv_hi_d4  (to_qdfh+1)/ ptstep_1hr_b4	/

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	
	DATA cv_category (to_qadx+1)/ cat_file_var	/
	DATA cv_variable (to_qadx+1)/ ptemp		/
	DATA cv_lo_d1  (to_qadx+1)/ -1		/
	DATA cv_hi_d1  (to_qadx+1)/ +1		/
	DATA cv_lo_d2  (to_qadx+1)/ -1		/
	DATA cv_hi_d2  (to_qadx+1)/ 0		/
	DATA cv_lo_d3  (to_qadx+1)/ 0		/
	DATA cv_hi_d3  (to_qadx+1)/ 0		/
	DATA cv_lo_d4  (to_qadx+1)/ 0		/
	DATA cv_hi_d4  (to_qadx+1)/ 0		/

	DATA cv_category (to_qadx+2)/ cat_file_var	/
	DATA cv_variable (to_qadx+2)/ pu		/
	DATA cv_lo_d1  (to_qadx+2)/ -1		/
	DATA cv_hi_d1  (to_qadx+2)/ 0		/
	DATA cv_lo_d2  (to_qadx+2)/ -1		/
	DATA cv_hi_d2  (to_qadx+2)/ 0		/
	DATA cv_lo_d3  (to_qadx+2)/ 0		/
	DATA cv_hi_d3  (to_qadx+2)/ 0		/
	DATA cv_lo_d4  (to_qadx+2)/ 0		/
	DATA cv_hi_d4  (to_qadx+2)/ 0		/

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	
	DATA cv_category (to_qady+1)/ cat_file_var	/
	DATA cv_variable (to_qady+1)/ ptemp		/
	DATA cv_lo_d1  (to_qady+1)/ 0		/
	DATA cv_hi_d1  (to_qady+1)/ 0		/
	DATA cv_lo_d2  (to_qady+1)/ -1		/
	DATA cv_hi_d2  (to_qady+1)/ +1		/
	DATA cv_lo_d3  (to_qady+1)/ 0		/
	DATA cv_hi_d3  (to_qady+1)/ 0		/
	DATA cv_lo_d4  (to_qady+1)/ 0		/
	DATA cv_hi_d4  (to_qady+1)/ 0		/

	DATA cv_category (to_qady+2)/ cat_file_var	/
	DATA cv_variable (to_qady+2)/ pv		/
	DATA cv_lo_d1  (to_qady+2)/ -1		/
	DATA cv_hi_d1  (to_qady+2)/ 0		/
	DATA cv_lo_d2  (to_qady+2)/ -1		/
	DATA cv_hi_d2  (to_qady+2)/ 0		/
	DATA cv_lo_d3  (to_qady+2)/ 0		/
	DATA cv_hi_d3  (to_qady+2)/ 0		/
	DATA cv_lo_d4  (to_qady+2)/ 0		/
	DATA cv_hi_d4  (to_qady+2)/ 0		/

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	
	DATA cv_category (to_qadz+1)/ cat_file_var	/
	DATA cv_variable (to_qadz+1)/ ptemp		/
	DATA cv_lo_d1  (to_qadz+1)/ 0		/
	DATA cv_hi_d1  (to_qadz+1)/ 0		/
	DATA cv_lo_d2  (to_qadz+1)/ 0		/
	DATA cv_hi_d2  (to_qadz+1)/ 0		/
	DATA cv_lo_d3  (to_qadz+1)/ -1		/
	DATA cv_hi_d3  (to_qadz+1)/ +1		/
	DATA cv_lo_d4  (to_qadz+1)/ 0		/
	DATA cv_hi_d4  (to_qadz+1)/ 0		/

	DATA cv_category (to_qadz+2)/ cat_file_var	/
	DATA cv_variable (to_qadz+2)/ pw		/
	DATA cv_lo_d1  (to_qadz+2)/ 0		/
	DATA cv_hi_d1  (to_qadz+2)/ 0		/
	DATA cv_lo_d2  (to_qadz+2)/ 0		/
	DATA cv_hi_d2  (to_qadz+2)/ 0		/
	DATA cv_lo_d3  (to_qadz+2)/ -1		/
	DATA cv_hi_d3  (to_qadz+2)/ 0		/
	DATA cv_lo_d4  (to_qadz+2)/ 0		/
	DATA cv_hi_d4  (to_qadz+2)/ 0		/

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	
	DATA cv_category (to_qad +1)/ cat_calc_var	/
	DATA cv_variable (to_qad +1)/ pqadx		/
	DATA cv_lo_d1  (to_qad +1)/ 0		/
	DATA cv_hi_d1  (to_qad +1)/ 0		/
	DATA cv_lo_d2  (to_qad +1)/ 0		/
	DATA cv_hi_d2  (to_qad +1)/ 0		/
	DATA cv_lo_d3  (to_qad +1)/ 0		/
	DATA cv_hi_d3  (to_qad +1)/ 0		/
	DATA cv_lo_d4  (to_qad +1)/ 0		/
	DATA cv_hi_d4  (to_qad +1)/ 0		/

	DATA cv_category (to_qad +2)/ cat_calc_var	/
	DATA cv_variable (to_qad +2)/ pqady		/
	DATA cv_lo_d1  (to_qad +2)/ 0		/
	DATA cv_hi_d1  (to_qad +2)/ 0		/
	DATA cv_lo_d2  (to_qad +2)/ 0		/
	DATA cv_hi_d2  (to_qad +2)/ 0		/
	DATA cv_lo_d3  (to_qad +2)/ 0		/
	DATA cv_hi_d3  (to_qad +2)/ 0		/
	DATA cv_lo_d4  (to_qad +2)/ 0		/
	DATA cv_hi_d4  (to_qad +2)/ 0		/

	DATA cv_category (to_qad +3)/ cat_calc_var	/
	DATA cv_variable (to_qad +3)/ pqadz		/
	DATA cv_lo_d1  (to_qad +3)/ 0		/
	DATA cv_hi_d1  (to_qad +3)/ 0		/
	DATA cv_lo_d2  (to_qad +3)/ 0		/
	DATA cv_hi_d2  (to_qad +3)/ 0		/
	DATA cv_lo_d3  (to_qad +3)/ 0		/
	DATA cv_hi_d3  (to_qad +3)/ 0		/
	DATA cv_lo_d4  (to_qad +3)/ 0		/
	DATA cv_hi_d4  (to_qad +3)/ 0		/

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	
	DATA cv_category (to_qdf +1)/ cat_calc_var	/
	DATA cv_variable (to_qdf +1)/ pqdfh		/
	DATA cv_lo_d1  (to_qdf +1)/ 0		/
	DATA cv_hi_d1  (to_qdf +1)/ 0		/
	DATA cv_lo_d2  (to_qdf +1)/ 0		/
	DATA cv_hi_d2  (to_qdf +1)/ 0		/
	DATA cv_lo_d3  (to_qdf +1)/ 0		/
	DATA cv_hi_d3  (to_qdf +1)/ 0		/
	DATA cv_lo_d4  (to_qdf +1)/ 0		/
	DATA cv_hi_d4  (to_qdf +1)/ 0		/

	DATA cv_category (to_qdf +2)/ cat_calc_var	/
	DATA cv_variable (to_qdf +2)/ pqdfz		/
	DATA cv_lo_d1  (to_qdf +2)/ 0		/
	DATA cv_hi_d1  (to_qdf +2)/ 0		/
	DATA cv_lo_d2  (to_qdf +2)/ 0		/
	DATA cv_hi_d2  (to_qdf +2)/ 0		/
	DATA cv_lo_d3  (to_qdf +2)/ 0		/
	DATA cv_hi_d3  (to_qdf +2)/ 0		/
	DATA cv_lo_d4  (to_qdf +2)/ 0		/
	DATA cv_hi_d4  (to_qdf +2)/ 0		/

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	
	DATA cv_category (to_qeva+1)/ cat_calc_var	/
	DATA cv_variable (to_qeva+1)/ ptau		/
	DATA cv_lo_d1  (to_qeva+1)/ 0		/
	DATA cv_hi_d1  (to_qeva+1)/ 0		/
	DATA cv_lo_d2  (to_qeva+1)/ 0		/
	DATA cv_hi_d2  (to_qeva+1)/ 0		/
	DATA cv_lo_d3  (to_qeva+1)/ 0		/
	DATA cv_hi_d3  (to_qeva+1)/ 0		/
	DATA cv_lo_d4  (to_qeva+1)/ 0		/
	DATA cv_hi_d4  (to_qeva+1)/ 0		/

	DATA cv_category (to_qeva+2)/ cat_file_var	/
	DATA cv_variable (to_qeva+2)/ ptemp 		/
	DATA cv_lo_d1  (to_qeva+2)/ 0		/
	DATA cv_hi_d1  (to_qeva+2)/ 0		/
	DATA cv_lo_d2  (to_qeva+2)/ 0		/
	DATA cv_hi_d2  (to_qeva+2)/ 0		/
	DATA cv_lo_d3  (to_qeva+2)/ 0		/
	DATA cv_hi_d3  (to_qeva+2)/ 0		/
	DATA cv_lo_d4  (to_qeva+2)/ 0		/
	DATA cv_hi_d4  (to_qeva+2)/ 0		/

	DATA cv_category (to_qeva+3)/ cat_calc_var	/
	DATA cv_variable (to_qeva+3)/ pair		/
	DATA cv_lo_d1  (to_qeva+3)/ 0		/
	DATA cv_hi_d1  (to_qeva+3)/ 0		/
	DATA cv_lo_d2  (to_qeva+3)/ 0		/
	DATA cv_hi_d2  (to_qeva+3)/ 0		/
	DATA cv_lo_d3  (to_qeva+3)/ 0		/
	DATA cv_hi_d3  (to_qeva+3)/ 0		/
	DATA cv_lo_d4  (to_qeva+3)/ 0		/
	DATA cv_hi_d4  (to_qeva+3)/ 0		/

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	
	DATA cv_category (to_tau +1)/ cat_file_var	/
	DATA cv_variable (to_tau +1)/ ptaux		/
	DATA cv_lo_d1  (to_tau +1)/ -1		/
	DATA cv_hi_d1  (to_tau +1)/ 0		/
	DATA cv_lo_d2  (to_tau +1)/ -1		/
	DATA cv_hi_d2  (to_tau +1)/ 0		/
	DATA cv_lo_d3  (to_tau +1)/ 0		/
	DATA cv_hi_d3  (to_tau +1)/ 0		/
	DATA cv_lo_d4  (to_tau +1)/ 0		/
	DATA cv_hi_d4  (to_tau +1)/ 0		/

	DATA cv_category (to_tau +2)/ cat_file_var	/
	DATA cv_variable (to_tau +2)/ ptauy		/
	DATA cv_lo_d1  (to_tau +2)/ -1		/
	DATA cv_hi_d1  (to_tau +2)/ 0		/
	DATA cv_lo_d2  (to_tau +2)/ -1		/
	DATA cv_hi_d2  (to_tau +2)/ 0		/
	DATA cv_lo_d3  (to_tau +2)/ 0		/
	DATA cv_hi_d3  (to_tau +2)/ 0		/
	DATA cv_lo_d4  (to_tau +2)/ 0		/
	DATA cv_hi_d4  (to_tau +2)/ 0		/

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	
	DATA cv_category (to_qsen+1)/ cat_calc_var	/
	DATA cv_variable (to_qsen+1)/ ptau 		/
	DATA cv_lo_d1  (to_qsen+1)/ 0		/
	DATA cv_hi_d1  (to_qsen+1)/ 0		/
	DATA cv_lo_d2  (to_qsen+1)/ 0		/
	DATA cv_hi_d2  (to_qsen+1)/ 0		/
	DATA cv_lo_d3  (to_qsen+1)/ 0		/
	DATA cv_hi_d3  (to_qsen+1)/ 0		/
	DATA cv_lo_d4  (to_qsen+1)/ 0		/
	DATA cv_hi_d4  (to_qsen+1)/ 0		/

	DATA cv_category (to_qsen+2)/ cat_file_var	/
	DATA cv_variable (to_qsen+2)/ ptemp		/
	DATA cv_lo_d1  (to_qsen+2)/ 0		/
	DATA cv_hi_d1  (to_qsen+2)/ 0		/
	DATA cv_lo_d2  (to_qsen+2)/ 0		/
	DATA cv_hi_d2  (to_qsen+2)/ 0		/
	DATA cv_lo_d3  (to_qsen+2)/ 0		/
	DATA cv_hi_d3  (to_qsen+2)/ 0		/
	DATA cv_lo_d4  (to_qsen+2)/ 0		/
	DATA cv_hi_d4  (to_qsen+2)/ 0		/

	DATA cv_category (to_qsen+3)/ cat_calc_var	/
	DATA cv_variable (to_qsen+3)/ pair		/
	DATA cv_lo_d1  (to_qsen+3)/ 0		/
	DATA cv_hi_d1  (to_qsen+3)/ 0		/
	DATA cv_lo_d2  (to_qsen+3)/ 0		/
	DATA cv_hi_d2  (to_qsen+3)/ 0		/
	DATA cv_lo_d3  (to_qsen+3)/ 0		/
	DATA cv_hi_d3  (to_qsen+3)/ 0		/
	DATA cv_lo_d4  (to_qsen+3)/ 0		/
	DATA cv_hi_d4  (to_qsen+3)/ 0		/

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

	DATA cv_category (to_qrad+1)/ cat_file_var	/
	DATA cv_variable (to_qrad+1)/ ptemp_special	/
	DATA cv_lo_d1  (to_qrad+1)/ 0		/
	DATA cv_hi_d1  (to_qrad+1)/ 0		/
	DATA cv_lo_d2  (to_qrad+1)/ 0		/
	DATA cv_hi_d2  (to_qrad+1)/ 0		/
	DATA cv_lo_d3  (to_qrad+1)/ 0		/	! SST
	DATA cv_hi_d3  (to_qrad+1)/ 0		/	! SST
	DATA cv_lo_d4  (to_qrad+1)/ 0		/
	DATA cv_hi_d4  (to_qrad+1)/ 0		/

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	
	DATA cv_category (to_qflx+1)/ cat_calc_var	/
	DATA cv_variable (to_qflx+1)/ pqeva		/
	DATA cv_lo_d1  (to_qflx+1)/ 0		/
	DATA cv_hi_d1  (to_qflx+1)/ 0		/
	DATA cv_lo_d2  (to_qflx+1)/ 0		/
	DATA cv_hi_d2  (to_qflx+1)/ 0		/
	DATA cv_lo_d3  (to_qflx+1)/ 0		/
	DATA cv_hi_d3  (to_qflx+1)/ 0		/
	DATA cv_lo_d4  (to_qflx+1)/ 0		/
	DATA cv_hi_d4  (to_qflx+1)/ 0		/

	DATA cv_category (to_qflx+2)/ cat_calc_var	/
	DATA cv_variable (to_qflx+2)/ pqsen		/
	DATA cv_lo_d1  (to_qflx+2)/ 0		/
	DATA cv_hi_d1  (to_qflx+2)/ 0		/
	DATA cv_lo_d2  (to_qflx+2)/ 0		/
	DATA cv_hi_d2  (to_qflx+2)/ 0		/
	DATA cv_lo_d3  (to_qflx+2)/ 0		/
	DATA cv_hi_d3  (to_qflx+2)/ 0		/
	DATA cv_lo_d4  (to_qflx+2)/ 0		/
	DATA cv_hi_d4  (to_qflx+2)/ 0		/

	DATA cv_category (to_qflx+3)/ cat_calc_var	/
	DATA cv_variable (to_qflx+3)/ pqrad		/
	DATA cv_lo_d1  (to_qflx+3)/ 0		/
	DATA cv_hi_d1  (to_qflx+3)/ 0		/
	DATA cv_lo_d2  (to_qflx+3)/ 0		/
	DATA cv_hi_d2  (to_qflx+3)/ 0		/
	DATA cv_lo_d3  (to_qflx+3)/ 0		/
	DATA cv_hi_d3  (to_qflx+3)/ 0		/
	DATA cv_lo_d4  (to_qflx+3)/ 0		/
	DATA cv_hi_d4  (to_qflx+3)/ 0		/

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	
	DATA cv_category (to_rho +1)/ cat_file_var	/
	DATA cv_variable (to_rho +1)/ ptemp		/
	DATA cv_lo_d1  (to_rho +1)/ 0		/
	DATA cv_hi_d1  (to_rho +1)/ 0		/
	DATA cv_lo_d2  (to_rho +1)/ 0		/
	DATA cv_hi_d2  (to_rho +1)/ 0		/
	DATA cv_lo_d3  (to_rho +1)/ 0		/
	DATA cv_hi_d3  (to_rho +1)/ 0		/
	DATA cv_lo_d4  (to_rho +1)/ 0		/
	DATA cv_hi_d4  (to_rho +1)/ 0		/
	
	DATA cv_category (to_rho +2)/ cat_file_var	/
	DATA cv_variable (to_rho +2)/ psalt		/
	DATA cv_lo_d1  (to_rho +2)/ 0		/
	DATA cv_hi_d1  (to_rho +2)/ 0		/
	DATA cv_lo_d2  (to_rho +2)/ 0		/
	DATA cv_hi_d2  (to_rho +2)/ 0		/
	DATA cv_lo_d3  (to_rho +2)/ 0		/
	DATA cv_hi_d3  (to_rho +2)/ 0		/
	DATA cv_lo_d4  (to_rho +2)/ 0		/
	DATA cv_hi_d4  (to_rho +2)/ 0		/

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

	DATA cv_category (to_dpdx+1)/ cat_calc_var	/
	DATA cv_variable (to_dpdx+1)/ prho		/
	DATA cv_lo_d1  (to_dpdx+1)/ 0		/
	DATA cv_hi_d1  (to_dpdx+1)/ +1		/
	DATA cv_lo_d2  (to_dpdx+1)/ 0		/
	DATA cv_hi_d2  (to_dpdx+1)/ +1		/
	DATA cv_lo_d3  (to_dpdx+1)/ 0		/	! needs k=1 absolute
	DATA cv_hi_d3  (to_dpdx+1)/ 0		/
	DATA cv_lo_d4  (to_dpdx+1)/ 0		/
	DATA cv_hi_d4  (to_dpdx+1)/ 0		/

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	
	DATA cv_category (to_dpdy+1)/ cat_calc_var	/
	DATA cv_variable (to_dpdy+1)/ prho		/
	DATA cv_lo_d1  (to_dpdy+1)/ 0		/
	DATA cv_hi_d1  (to_dpdy+1)/ +1		/
	DATA cv_lo_d2  (to_dpdy+1)/ 0		/
	DATA cv_hi_d2  (to_dpdy+1)/ +1		/
	DATA cv_lo_d3  (to_dpdy+1)/ 0		/	! needs k=1 absolute
	DATA cv_hi_d3  (to_dpdy+1)/ 0		/
	DATA cv_lo_d4  (to_dpdy+1)/ 0		/
	DATA cv_hi_d4  (to_dpdy+1)/ 0		/

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

	DATA cv_category (to_w_u +1)/ cat_calc_var	/
	DATA cv_variable (to_w_u +1)/ pfuw		/
	DATA cv_lo_d1  (to_w_u +1)/ 0		/
	DATA cv_hi_d1  (to_w_u +1)/ +1		/
	DATA cv_lo_d2  (to_w_u +1)/ 0		/
	DATA cv_hi_d2  (to_w_u +1)/ 0		/
	DATA cv_lo_d3  (to_w_u +1)/ 0		/
	DATA cv_hi_d3  (to_w_u +1)/ 0		/
	DATA cv_lo_d4  (to_w_u +1)/ 0		/
	DATA cv_hi_d4  (to_w_u +1)/ 0		/
	
	DATA cv_category (to_w_u +2)/ cat_calc_var	/
	DATA cv_variable (to_w_u +2)/ pfvn		/
	DATA cv_lo_d1  (to_w_u +2)/ 0		/
	DATA cv_hi_d1  (to_w_u +2)/ 0		/
	DATA cv_lo_d2  (to_w_u +2)/ -1		/
	DATA cv_hi_d2  (to_w_u +2)/ 0		/
	DATA cv_lo_d3  (to_w_u +2)/ 0		/
	DATA cv_hi_d3  (to_w_u +2)/ 0		/
	DATA cv_lo_d4  (to_w_u +2)/ 0		/
	DATA cv_hi_d4  (to_w_u +2)/ 0		/

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	
	DATA cv_category (to_uavz+1)/ cat_file_var	/
	DATA cv_variable (to_uavz+1)/ ppsi		/
	DATA cv_lo_d1  (to_uavz+1)/ 0		/
	DATA cv_hi_d1  (to_uavz+1)/ +1		/
	DATA cv_lo_d2  (to_uavz+1)/ 0		/
	DATA cv_hi_d2  (to_uavz+1)/ +1		/
	DATA cv_lo_d3  (to_uavz+1)/ 0		/
	DATA cv_hi_d3  (to_uavz+1)/ 0		/
	DATA cv_lo_d4  (to_uavz+1)/ 0		/
	DATA cv_hi_d4  (to_uavz+1)/ 0		/

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	
	DATA cv_category (to_vavz+1)/ cat_file_var	/
	DATA cv_variable (to_vavz+1)/ ppsi		/
	DATA cv_lo_d1  (to_vavz+1)/ 0		/
	DATA cv_hi_d1  (to_vavz+1)/ +1		/
	DATA cv_lo_d2  (to_vavz+1)/ 0		/
	DATA cv_hi_d2  (to_vavz+1)/ +1		/
	DATA cv_lo_d3  (to_vavz+1)/ 0		/
	DATA cv_hi_d3  (to_vavz+1)/ 0		/
	DATA cv_lo_d4  (to_vavz+1)/ 0		/
	DATA cv_hi_d4  (to_vavz+1)/ 0		/

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	
	DATA cv_category (to_fuw +1)/ cat_file_var	/
	DATA cv_variable (to_fuw +1)/ ppsi		/
	DATA cv_lo_d1  (to_fuw +1)/ -1		/	! extra for UAVZ
	DATA cv_hi_d1  (to_fuw +1)/ +1		/	! extra for UAVZ
	DATA cv_lo_d2  (to_fuw +1)/ 0		/
	DATA cv_hi_d2  (to_fuw +1)/ +1		/
	DATA cv_lo_d3  (to_fuw +1)/ 0		/
	DATA cv_hi_d3  (to_fuw +1)/ 0		/
	DATA cv_lo_d4  (to_fuw +1)/ 0		/
	DATA cv_hi_d4  (to_fuw +1)/ 0		/
	
	DATA cv_category (to_fuw +2)/ cat_calc_var	/
	DATA cv_variable (to_fuw +2)/ puavz		/
	DATA cv_lo_d1  (to_fuw +2)/ -1		/
	DATA cv_hi_d1  (to_fuw +2)/ 0		/
	DATA cv_lo_d2  (to_fuw +2)/ 0		/
	DATA cv_hi_d2  (to_fuw +2)/ 0		/
	DATA cv_lo_d3  (to_fuw +2)/ 0		/
	DATA cv_hi_d3  (to_fuw +2)/ 0		/
	DATA cv_lo_d4  (to_fuw +2)/ 0		/
	DATA cv_hi_d4  (to_fuw +2)/ 0		/

	DATA cv_category (to_fuw +3)/ cat_file_var	/
	DATA cv_variable (to_fuw +3)/ pu		/
	DATA cv_lo_d1  (to_fuw +3)/ -1		/
	DATA cv_hi_d1  (to_fuw +3)/ 0		/
	DATA cv_lo_d2  (to_fuw +3)/ 0		/
	DATA cv_hi_d2  (to_fuw +3)/ 0		/
	DATA cv_lo_d3  (to_fuw +3)/ 0		/
	DATA cv_hi_d3  (to_fuw +3)/ 0		/
	DATA cv_lo_d4  (to_fuw +3)/ 0		/
	DATA cv_hi_d4  (to_fuw +3)/ 0		/

***************************************************************************
	
	DATA cv_category (to_fvn +1)/ cat_file_var	/
	DATA cv_variable (to_fvn +1)/ ppsi		/
	DATA cv_lo_d1  (to_fvn +1)/ 0		/
	DATA cv_hi_d1  (to_fvn +1)/ +1		/
	DATA cv_lo_d2  (to_fvn +1)/ 0		/	! extra for VAVZ
	DATA cv_hi_d2  (to_fvn +1)/ +2		/	! extra for VAVZ
	DATA cv_lo_d3  (to_fvn +1)/ 0		/
	DATA cv_hi_d3  (to_fvn +1)/ 0		/
	DATA cv_lo_d4  (to_fvn +1)/ 0		/
	DATA cv_hi_d4  (to_fvn +1)/ 0		/
	
	DATA cv_category (to_fvn +2)/ cat_calc_var	/
	DATA cv_variable (to_fvn +2)/ pvavz		/
	DATA cv_lo_d1  (to_fvn +2)/ 0		/
	DATA cv_hi_d1  (to_fvn +2)/ 0		/
	DATA cv_lo_d2  (to_fvn +2)/ 0		/
	DATA cv_hi_d2  (to_fvn +2)/ +1		/
	DATA cv_lo_d3  (to_fvn +2)/ 0		/
	DATA cv_hi_d3  (to_fvn +2)/ 0		/
	DATA cv_lo_d4  (to_fvn +2)/ 0		/
	DATA cv_hi_d4  (to_fvn +2)/ 0		/

	DATA cv_category (to_fvn +3)/ cat_file_var	/
	DATA cv_variable (to_fvn +3)/ pv		/
	DATA cv_lo_d1  (to_fvn +3)/ 0		/
	DATA cv_hi_d1  (to_fvn +3)/ 0		/
	DATA cv_lo_d2  (to_fvn +3)/ 0		/
	DATA cv_hi_d2  (to_fvn +3)/ +1		/
	DATA cv_lo_d3  (to_fvn +3)/ 0		/
	DATA cv_hi_d3  (to_fvn +3)/ 0		/
	DATA cv_lo_d4  (to_fvn +3)/ 0		/
	DATA cv_hi_d4  (to_fvn +3)/ 0		/

*****************************************************************************
	
	DATA cv_category (to_uadx+1)/ cat_calc_var	/
	DATA cv_variable (to_uadx+1)/ pfuw		/
	DATA cv_lo_d1  (to_uadx+1)/ 0		/
	DATA cv_hi_d1  (to_uadx+1)/ +1		/
	DATA cv_lo_d2  (to_uadx+1)/ 0		/
	DATA cv_hi_d2  (to_uadx+1)/ 0		/
	DATA cv_lo_d3  (to_uadx+1)/ 0		/
	DATA cv_hi_d3  (to_uadx+1)/ 0		/
	DATA cv_lo_d4  (to_uadx+1)/ 0		/
	DATA cv_hi_d4  (to_uadx+1)/ 0		/
	
	DATA cv_category (to_uadx+2)/ cat_file_var	/
	DATA cv_variable (to_uadx+2)/ pu		/
	DATA cv_lo_d1  (to_uadx+2)/ -1		/
	DATA cv_hi_d1  (to_uadx+2)/ +1		/
	DATA cv_lo_d2  (to_uadx+2)/ 0		/
	DATA cv_hi_d2  (to_uadx+2)/ 0		/
	DATA cv_lo_d3  (to_uadx+2)/ 0		/
	DATA cv_hi_d3  (to_uadx+2)/ 0		/
	DATA cv_lo_d4  (to_uadx+2)/ 0		/
	DATA cv_hi_d4  (to_uadx+2)/ 0		/

*****************************************************************************
	
	DATA cv_category (to_uady+1)/ cat_calc_var	/
	DATA cv_variable (to_uady+1)/ pfvn		/
	DATA cv_lo_d1  (to_uady+1)/ 0		/
	DATA cv_hi_d1  (to_uady+1)/ 0		/
	DATA cv_lo_d2  (to_uady+1)/ -1		/
	DATA cv_hi_d2  (to_uady+1)/ 0		/
	DATA cv_lo_d3  (to_uady+1)/ 0		/
	DATA cv_hi_d3  (to_uady+1)/ 0		/
	DATA cv_lo_d4  (to_uady+1)/ 0		/
	DATA cv_hi_d4  (to_uady+1)/ 0		/
	
	DATA cv_category (to_uady+2)/ cat_file_var	/
	DATA cv_variable (to_uady+2)/ pu		/
	DATA cv_lo_d1  (to_uady+2)/ 0		/
	DATA cv_hi_d1  (to_uady+2)/ 0		/
	DATA cv_lo_d2  (to_uady+2)/ -1		/
	DATA cv_hi_d2  (to_uady+2)/ +1		/
	DATA cv_lo_d3  (to_uady+2)/ 0		/
	DATA cv_hi_d3  (to_uady+2)/ 0		/
	DATA cv_lo_d4  (to_uady+2)/ 0		/
	DATA cv_hi_d4  (to_uady+2)/ 0		/

*****************************************************************************
	
	DATA cv_category (to_uadz+1)/ cat_file_var	/
	DATA cv_variable (to_uadz+1)/ pu		/
	DATA cv_lo_d1  (to_uadz+1)/ 0		/
	DATA cv_hi_d1  (to_uadz+1)/ 0		/
	DATA cv_lo_d2  (to_uadz+1)/ 0		/
	DATA cv_hi_d2  (to_uadz+1)/ 0		/
	DATA cv_lo_d3  (to_uadz+1)/ -1		/
	DATA cv_hi_d3  (to_uadz+1)/ +1		/
	DATA cv_lo_d4  (to_uadz+1)/ 0		/
	DATA cv_hi_d4  (to_uadz+1)/ 0		/
	
	DATA cv_category (to_uadz+2)/ cat_calc_var	/
	DATA cv_variable (to_uadz+2)/ pw_u		/
	DATA cv_lo_d1  (to_uadz+2)/ 0		/
	DATA cv_hi_d1  (to_uadz+2)/ 0		/
	DATA cv_lo_d2  (to_uadz+2)/ 0		/
	DATA cv_hi_d2  (to_uadz+2)/ 0		/
	DATA cv_lo_d3  (to_uadz+2)/ -1		/
	DATA cv_hi_d3  (to_uadz+2)/ 0		/
	DATA cv_lo_d4  (to_uadz+2)/ 0		/
	DATA cv_hi_d4  (to_uadz+2)/ 0		/

**************************************************************************
	
	DATA cv_category (to_vadx+1)/ cat_calc_var	/
	DATA cv_variable (to_vadx+1)/ pfuw		/
	DATA cv_lo_d1  (to_vadx+1)/ 0		/
	DATA cv_hi_d1  (to_vadx+1)/ +1		/
	DATA cv_lo_d2  (to_vadx+1)/ 0		/
	DATA cv_hi_d2  (to_vadx+1)/ 0		/
	DATA cv_lo_d3  (to_vadx+1)/ 0		/
	DATA cv_hi_d3  (to_vadx+1)/ 0		/
	DATA cv_lo_d4  (to_vadx+1)/ 0		/
	DATA cv_hi_d4  (to_vadx+1)/ 0		/
	
	DATA cv_category (to_vadx+2)/ cat_file_var	/
	DATA cv_variable (to_vadx+2)/ pv		/
	DATA cv_lo_d1  (to_vadx+2)/ -1		/
	DATA cv_hi_d1  (to_vadx+2)/ +1		/
	DATA cv_lo_d2  (to_vadx+2)/ 0		/
	DATA cv_hi_d2  (to_vadx+2)/ 0		/
	DATA cv_lo_d3  (to_vadx+2)/ 0		/
	DATA cv_hi_d3  (to_vadx+2)/ 0		/
	DATA cv_lo_d4  (to_vadx+2)/ 0		/
	DATA cv_hi_d4  (to_vadx+2)/ 0		/

*****************************************************************************
	
	DATA cv_category (to_vady+1)/ cat_calc_var	/
	DATA cv_variable (to_vady+1)/ pfvn		/
	DATA cv_lo_d1  (to_vady+1)/ 0		/
	DATA cv_hi_d1  (to_vady+1)/ 0		/
	DATA cv_lo_d2  (to_vady+1)/ -1		/
	DATA cv_hi_d2  (to_vady+1)/ 0		/
	DATA cv_lo_d3  (to_vady+1)/ 0		/
	DATA cv_hi_d3  (to_vady+1)/ 0		/
	DATA cv_lo_d4  (to_vady+1)/ 0		/
	DATA cv_hi_d4  (to_vady+1)/ 0		/
	
	DATA cv_category (to_vady+2)/ cat_file_var	/
	DATA cv_variable (to_vady+2)/ pv		/
	DATA cv_lo_d1  (to_vady+2)/ 0		/
	DATA cv_hi_d1  (to_vady+2)/ 0		/
	DATA cv_lo_d2  (to_vady+2)/ -1		/
	DATA cv_hi_d2  (to_vady+2)/ +1		/
	DATA cv_lo_d3  (to_vady+2)/ 0		/
	DATA cv_hi_d3  (to_vady+2)/ 0		/
	DATA cv_lo_d4  (to_vady+2)/ 0		/
	DATA cv_hi_d4  (to_vady+2)/ 0		/

*****************************************************************************
	
	DATA cv_category (to_vadz+1)/ cat_file_var	/
	DATA cv_variable (to_vadz+1)/ pv		/
	DATA cv_lo_d1  (to_vadz+1)/ 0		/
	DATA cv_hi_d1  (to_vadz+1)/ 0		/
	DATA cv_lo_d2  (to_vadz+1)/ 0		/
	DATA cv_hi_d2  (to_vadz+1)/ 0		/
	DATA cv_lo_d3  (to_vadz+1)/ -1		/
	DATA cv_hi_d3  (to_vadz+1)/ +1		/
	DATA cv_lo_d4  (to_vadz+1)/ 0		/
	DATA cv_hi_d4  (to_vadz+1)/ 0		/
	
	DATA cv_category (to_vadz+2)/ cat_calc_var	/
	DATA cv_variable (to_vadz+2)/ pw_u		/
	DATA cv_lo_d1  (to_vadz+2)/ 0		/
	DATA cv_hi_d1  (to_vadz+2)/ 0		/
	DATA cv_lo_d2  (to_vadz+2)/ 0		/
	DATA cv_hi_d2  (to_vadz+2)/ 0		/
	DATA cv_lo_d3  (to_vadz+2)/ -1		/
	DATA cv_hi_d3  (to_vadz+2)/ 0		/
	DATA cv_lo_d4  (to_vadz+2)/ 0		/
	DATA cv_hi_d4  (to_vadz+2)/ 0		/

**************************************************************************
	
	DATA cv_category (to_udfh+1)/ cat_file_var	/
	DATA cv_variable (to_udfh+1)/ pu		/
	DATA cv_lo_d1  (to_udfh+1)/ -1		/
	DATA cv_hi_d1  (to_udfh+1)/ +1		/
	DATA cv_lo_d2  (to_udfh+1)/ -1		/
	DATA cv_hi_d2  (to_udfh+1)/ +1		/
	DATA cv_lo_d3  (to_udfh+1)/ 0		/
	DATA cv_hi_d3  (to_udfh+1)/ 0		/
	DATA cv_lo_d4  (to_udfh+1)/ ptstep_1hr_b4	/
	DATA cv_hi_d4  (to_udfh+1)/ 0		/
	
	DATA cv_category (to_udfh+2)/ cat_file_var	/
	DATA cv_variable (to_udfh+2)/ pv		/
	DATA cv_lo_d1  (to_udfh+2)/ -1		/
	DATA cv_hi_d1  (to_udfh+2)/ +1		/
	DATA cv_lo_d2  (to_udfh+2)/ -1		/
	DATA cv_hi_d2  (to_udfh+2)/ +1		/
	DATA cv_lo_d3  (to_udfh+2)/ 0		/
	DATA cv_hi_d3  (to_udfh+2)/ 0		/
	DATA cv_lo_d4  (to_udfh+2)/ 0		/
	DATA cv_hi_d4  (to_udfh+2)/ 0		/
	
**************************************************************************
	
	DATA cv_category (to_udfz+1)/ cat_calc_var	/
	DATA cv_variable (to_udfz+1)/ ppcdz		/
	DATA cv_lo_d1  (to_udfz+1)/ 0		/
	DATA cv_hi_d1  (to_udfz+1)/ 0		/
	DATA cv_lo_d2  (to_udfz+1)/ 0		/
	DATA cv_hi_d2  (to_udfz+1)/ 0		/
	DATA cv_lo_d3  (to_udfz+1)/ -1		/
	DATA cv_hi_d3  (to_udfz+1)/ 0		/
	DATA cv_lo_d4  (to_udfz+1)/ ptstep_1hr_b4	/
	DATA cv_hi_d4  (to_udfz+1)/ ptstep_1hr_b4	/
	
	DATA cv_category (to_udfz+2)/ cat_file_var	/
	DATA cv_variable (to_udfz+2)/ pu		/
	DATA cv_lo_d1  (to_udfz+2)/ 0		/
	DATA cv_hi_d1  (to_udfz+2)/ 0		/
	DATA cv_lo_d2  (to_udfz+2)/ 0		/
	DATA cv_hi_d2  (to_udfz+2)/ 0		/
	DATA cv_lo_d3  (to_udfz+2)/ -1		/
	DATA cv_hi_d3  (to_udfz+2)/ +1		/
	DATA cv_lo_d4  (to_udfz+2)/ ptstep_1hr_b4	/
	DATA cv_hi_d4  (to_udfz+2)/ ptstep_1hr_b4	/

	DATA cv_category (to_udfz+3)/ cat_calc_var	/
	DATA cv_variable (to_udfz+3)/ pubwi		/
	DATA cv_lo_d1  (to_udfz+3)/ 0		/
	DATA cv_hi_d1  (to_udfz+3)/ 0		/
	DATA cv_lo_d2  (to_udfz+3)/ 0		/
	DATA cv_hi_d2  (to_udfz+3)/ 0		/
	DATA cv_lo_d3  (to_udfz+3)/ 0		/
	DATA cv_hi_d3  (to_udfz+3)/ 0		/
	DATA cv_lo_d4  (to_udfz+3)/ ptstep_1hr_b4	/
	DATA cv_hi_d4  (to_udfz+3)/ ptstep_1hr_b4	/

**************************************************************************
	
	DATA cv_category (to_vdfh+1)/ cat_file_var	/
	DATA cv_variable (to_vdfh+1)/ pv		/
	DATA cv_lo_d1  (to_vdfh+1)/ -1		/
	DATA cv_hi_d1  (to_vdfh+1)/ +1		/
	DATA cv_lo_d2  (to_vdfh+1)/ -1		/
	DATA cv_hi_d2  (to_vdfh+1)/ +1		/
	DATA cv_lo_d3  (to_vdfh+1)/ 0		/
	DATA cv_hi_d3  (to_vdfh+1)/ 0		/
	DATA cv_lo_d4  (to_vdfh+1)/ ptstep_1hr_b4	/
	DATA cv_hi_d4  (to_vdfh+1)/ 0		/

	DATA cv_category (to_vdfh+2)/ cat_file_var	/
	DATA cv_variable (to_vdfh+2)/ pu		/
	DATA cv_lo_d1  (to_vdfh+2)/ -1		/
	DATA cv_hi_d1  (to_vdfh+2)/ +1		/
	DATA cv_lo_d2  (to_vdfh+2)/ -1		/
	DATA cv_hi_d2  (to_vdfh+2)/ +1		/
	DATA cv_lo_d3  (to_vdfh+2)/ 0		/
	DATA cv_hi_d3  (to_vdfh+2)/ 0		/
	DATA cv_lo_d4  (to_vdfh+2)/ 0		/
	DATA cv_hi_d4  (to_vdfh+2)/ 0		/
	
**************************************************************************
	
	DATA cv_category (to_vdfz+1)/ cat_calc_var	/
	DATA cv_variable (to_vdfz+1)/ ppcdz		/
	DATA cv_lo_d1  (to_vdfz+1)/ 0		/
	DATA cv_hi_d1  (to_vdfz+1)/ 0		/
	DATA cv_lo_d2  (to_vdfz+1)/ 0		/
	DATA cv_hi_d2  (to_vdfz+1)/ 0		/
	DATA cv_lo_d3  (to_vdfz+1)/ -1		/
	DATA cv_hi_d3  (to_vdfz+1)/ 0		/
	DATA cv_lo_d4  (to_vdfz+1)/ ptstep_1hr_b4	/
	DATA cv_hi_d4  (to_vdfz+1)/ ptstep_1hr_b4	/
	
	DATA cv_category (to_vdfz+2)/ cat_file_var	/
	DATA cv_variable (to_vdfz+2)/ pv		/
	DATA cv_lo_d1  (to_vdfz+2)/ 0		/
	DATA cv_hi_d1  (to_vdfz+2)/ 0		/
	DATA cv_lo_d2  (to_vdfz+2)/ 0		/
	DATA cv_hi_d2  (to_vdfz+2)/ 0		/
	DATA cv_lo_d3  (to_vdfz+2)/ -1		/
	DATA cv_hi_d3  (to_vdfz+2)/ +1		/
	DATA cv_lo_d4  (to_vdfz+2)/ ptstep_1hr_b4	/
	DATA cv_hi_d4  (to_vdfz+2)/ ptstep_1hr_b4	/

	DATA cv_category (to_vdfz+3)/ cat_calc_var	/
	DATA cv_variable (to_vdfz+3)/ pvbwi		/
	DATA cv_lo_d1  (to_vdfz+3)/ 0		/
	DATA cv_hi_d1  (to_vdfz+3)/ 0		/
	DATA cv_lo_d2  (to_vdfz+3)/ 0		/
	DATA cv_hi_d2  (to_vdfz+3)/ 0		/
	DATA cv_lo_d3  (to_vdfz+3)/ 0		/
	DATA cv_hi_d3  (to_vdfz+3)/ 0		/
	DATA cv_lo_d4  (to_vdfz+3)/ ptstep_1hr_b4	/
	DATA cv_hi_d4  (to_vdfz+3)/ ptstep_1hr_b4	/

**************************************************************************
	
	DATA cv_category (to_ucoi+1)/ cat_file_var	/
	DATA cv_variable (to_ucoi+1)/ ptemp		/
	DATA cv_lo_d1  (to_ucoi+1)/ 0		/
	DATA cv_hi_d1  (to_ucoi+1)/ 0		/
	DATA cv_lo_d2  (to_ucoi+1)/ 0		/
	DATA cv_hi_d2  (to_ucoi+1)/ 0		/
	DATA cv_lo_d3  (to_ucoi+1)/ 0		/
	DATA cv_hi_d3  (to_ucoi+1)/ 0		/
	DATA cv_lo_d4  (to_ucoi+1)/ 0		/
	DATA cv_hi_d4  (to_ucoi+1)/ 0		/
	
**************************************************************************
	
	DATA cv_category (to_vcoi+1)/ cat_file_var	/
	DATA cv_variable (to_vcoi+1)/ ptemp		/
	DATA cv_lo_d1  (to_vcoi+1)/ 0		/
	DATA cv_hi_d1  (to_vcoi+1)/ 0		/
	DATA cv_lo_d2  (to_vcoi+1)/ 0		/
	DATA cv_hi_d2  (to_vcoi+1)/ 0		/
	DATA cv_lo_d3  (to_vcoi+1)/ 0		/
	DATA cv_hi_d3  (to_vcoi+1)/ 0		/
	DATA cv_lo_d4  (to_vcoi+1)/ 0		/
	DATA cv_hi_d4  (to_vcoi+1)/ 0		/

**************************************************************************
	
	DATA cv_category (to_ubpi+1)/ cat_calc_var	/
	DATA cv_variable (to_ubpi+1)/ pdpdx		/
	DATA cv_lo_d1  (to_ubpi+1)/ 0		/
	DATA cv_hi_d1  (to_ubpi+1)/ 0		/
	DATA cv_lo_d2  (to_ubpi+1)/ 0		/
	DATA cv_hi_d2  (to_ubpi+1)/ 0		/
	DATA cv_lo_d3  (to_ubpi+1)/ 0		/
	DATA cv_hi_d3  (to_ubpi+1)/ 0		/
	DATA cv_lo_d4  (to_ubpi+1)/ 0		/
	DATA cv_hi_d4  (to_ubpi+1)/ 0		/

**************************************************************************
	
	DATA cv_category (to_vbpi+1)/ cat_calc_var	/
	DATA cv_variable (to_vbpi+1)/ pdpdy		/
	DATA cv_lo_d1  (to_vbpi+1)/ 0		/
	DATA cv_hi_d1  (to_vbpi+1)/ 0		/
	DATA cv_lo_d2  (to_vbpi+1)/ 0		/
	DATA cv_hi_d2  (to_vbpi+1)/ 0		/
	DATA cv_lo_d3  (to_vbpi+1)/ 0		/
	DATA cv_hi_d3  (to_vbpi+1)/ 0		/
	DATA cv_lo_d4  (to_vbpi+1)/ 0		/
	DATA cv_hi_d4  (to_vbpi+1)/ 0		/

**************************************************************************
	
	DATA cv_category (to_ubwi+1)/ cat_file_var	/
	DATA cv_variable (to_ubwi+1)/ ptaux		/
	DATA cv_lo_d1  (to_ubwi+1)/ 0		/
	DATA cv_hi_d1  (to_ubwi+1)/ 0		/
	DATA cv_lo_d2  (to_ubwi+1)/ 0		/
	DATA cv_hi_d2  (to_ubwi+1)/ 0		/
	DATA cv_lo_d3  (to_ubwi+1)/ 0		/
	DATA cv_hi_d3  (to_ubwi+1)/ 0		/
	DATA cv_lo_d4  (to_ubwi+1)/ 0		/
	DATA cv_hi_d4  (to_ubwi+1)/ 0		/

**************************************************************************
	
	DATA cv_category (to_vbwi+1)/ cat_file_var	/
	DATA cv_variable (to_vbwi+1)/ ptauy		/
	DATA cv_lo_d1  (to_vbwi+1)/ 0		/
	DATA cv_hi_d1  (to_vbwi+1)/ 0		/
	DATA cv_lo_d2  (to_vbwi+1)/ 0		/
	DATA cv_hi_d2  (to_vbwi+1)/ 0		/
	DATA cv_lo_d3  (to_vbwi+1)/ 0		/
	DATA cv_hi_d3  (to_vbwi+1)/ 0		/
	DATA cv_lo_d4  (to_vbwi+1)/ 0		/
	DATA cv_hi_d4  (to_vbwi+1)/ 0		/

**************************************************************************
	
	DATA cv_category (to_ucoe+1)/ cat_file_var	/
	DATA cv_variable (to_ucoe+1)/ pu		/
	DATA cv_lo_d1  (to_ucoe+1)/ 0		/
	DATA cv_hi_d1  (to_ucoe+1)/ 0		/
	DATA cv_lo_d2  (to_ucoe+1)/ 0		/
	DATA cv_hi_d2  (to_ucoe+1)/ 0		/
	DATA cv_lo_d3  (to_ucoe+1)/ 0		/
	DATA cv_hi_d3  (to_ucoe+1)/ 0		/
	DATA cv_lo_d4  (to_ucoe+1)/ 0		/
	DATA cv_hi_d4  (to_ucoe+1)/ 0		/
	
	DATA cv_category (to_ucoe+2)/ cat_file_var	/
	DATA cv_variable (to_ucoe+2)/ pv		/
	DATA cv_lo_d1  (to_ucoe+2)/ 0		/
	DATA cv_hi_d1  (to_ucoe+2)/ 0		/
	DATA cv_lo_d2  (to_ucoe+2)/ 0		/
	DATA cv_hi_d2  (to_ucoe+2)/ 0		/
	DATA cv_lo_d3  (to_ucoe+2)/ 0		/
	DATA cv_hi_d3  (to_ucoe+2)/ 0		/
	DATA cv_lo_d4  (to_ucoe+2)/ 0		/
	DATA cv_hi_d4  (to_ucoe+2)/ 0		/

**************************************************************************
	
	DATA cv_category (to_vcoe+1)/ cat_file_var	/
	DATA cv_variable (to_vcoe+1)/ pu		/
	DATA cv_lo_d1  (to_vcoe+1)/ 0		/
	DATA cv_hi_d1  (to_vcoe+1)/ 0		/
	DATA cv_lo_d2  (to_vcoe+1)/ 0		/
	DATA cv_hi_d2  (to_vcoe+1)/ 0		/
	DATA cv_lo_d3  (to_vcoe+1)/ 0		/
	DATA cv_hi_d3  (to_vcoe+1)/ 0		/
	DATA cv_lo_d4  (to_vcoe+1)/ 0		/
	DATA cv_hi_d4  (to_vcoe+1)/ 0		/
	
	DATA cv_category (to_vcoe+2)/ cat_file_var	/
	DATA cv_variable (to_vcoe+2)/ pv		/
	DATA cv_lo_d1  (to_vcoe+2)/ 0		/
	DATA cv_hi_d1  (to_vcoe+2)/ 0		/
	DATA cv_lo_d2  (to_vcoe+2)/ 0		/
	DATA cv_hi_d2  (to_vcoe+2)/ 0		/
	DATA cv_lo_d3  (to_vcoe+2)/ 0		/
	DATA cv_hi_d3  (to_vcoe+2)/ 0		/
	DATA cv_lo_d4  (to_vcoe+2)/ 0		/
	DATA cv_hi_d4  (to_vcoe+2)/ 0		/

**************************************************************************
	
	DATA cv_category (to_uad +1)/ cat_calc_var	/
	DATA cv_variable (to_uad +1)/ puadx		/
	DATA cv_lo_d1  (to_uad +1)/ 0		/
	DATA cv_hi_d1  (to_uad +1)/ 0		/
	DATA cv_lo_d2  (to_uad +1)/ 0		/
	DATA cv_hi_d2  (to_uad +1)/ 0		/
	DATA cv_lo_d3  (to_uad +1)/ 0		/
	DATA cv_hi_d3  (to_uad +1)/ 0		/
	DATA cv_lo_d4  (to_uad +1)/ 0		/
	DATA cv_hi_d4  (to_uad +1)/ 0		/
	
	DATA cv_category (to_uad +2)/ cat_calc_var	/
	DATA cv_variable (to_uad +2)/ puady		/
	DATA cv_lo_d1  (to_uad +2)/ 0		/
	DATA cv_hi_d1  (to_uad +2)/ 0		/
	DATA cv_lo_d2  (to_uad +2)/ 0		/
	DATA cv_hi_d2  (to_uad +2)/ 0		/
	DATA cv_lo_d3  (to_uad +2)/ 0		/
	DATA cv_hi_d3  (to_uad +2)/ 0		/
	DATA cv_lo_d4  (to_uad +2)/ 0		/
	DATA cv_hi_d4  (to_uad +2)/ 0		/

	DATA cv_category (to_uad +3)/ cat_calc_var	/
	DATA cv_variable (to_uad +3)/ puadz		/
	DATA cv_lo_d1  (to_uad +3)/ 0		/
	DATA cv_hi_d1  (to_uad +3)/ 0		/
	DATA cv_lo_d2  (to_uad +3)/ 0		/
	DATA cv_hi_d2  (to_uad +3)/ 0		/
	DATA cv_lo_d3  (to_uad +3)/ 0		/
	DATA cv_hi_d3  (to_uad +3)/ 0		/
	DATA cv_lo_d4  (to_uad +3)/ 0		/
	DATA cv_hi_d4  (to_uad +3)/ 0		/

**************************************************************************
	
	DATA cv_category (to_vad +1)/ cat_calc_var	/
	DATA cv_variable (to_vad +1)/ pvadx		/
	DATA cv_lo_d1  (to_vad +1)/ 0		/
	DATA cv_hi_d1  (to_vad +1)/ 0		/
	DATA cv_lo_d2  (to_vad +1)/ 0		/
	DATA cv_hi_d2  (to_vad +1)/ 0		/
	DATA cv_lo_d3  (to_vad +1)/ 0		/
	DATA cv_hi_d3  (to_vad +1)/ 0		/
	DATA cv_lo_d4  (to_vad +1)/ 0		/
	DATA cv_hi_d4  (to_vad +1)/ 0		/
	
	DATA cv_category (to_vad +2)/ cat_calc_var	/
	DATA cv_variable (to_vad +2)/ pvady		/
	DATA cv_lo_d1  (to_vad +2)/ 0		/
	DATA cv_hi_d1  (to_vad +2)/ 0		/
	DATA cv_lo_d2  (to_vad +2)/ 0		/
	DATA cv_hi_d2  (to_vad +2)/ 0		/
	DATA cv_lo_d3  (to_vad +2)/ 0		/
	DATA cv_hi_d3  (to_vad +2)/ 0		/
	DATA cv_lo_d4  (to_vad +2)/ 0		/
	DATA cv_hi_d4  (to_vad +2)/ 0		/

	DATA cv_category (to_vad +3)/ cat_calc_var	/
	DATA cv_variable (to_vad +3)/ pvadz		/
	DATA cv_lo_d1  (to_vad +3)/ 0		/
	DATA cv_hi_d1  (to_vad +3)/ 0		/
	DATA cv_lo_d2  (to_vad +3)/ 0		/
	DATA cv_hi_d2  (to_vad +3)/ 0		/
	DATA cv_lo_d3  (to_vad +3)/ 0		/
	DATA cv_hi_d3  (to_vad +3)/ 0		/
	DATA cv_lo_d4  (to_vad +3)/ 0		/
	DATA cv_hi_d4  (to_vad +3)/ 0		/

**************************************************************************
	
	DATA cv_category (to_udf +1)/ cat_calc_var	/
	DATA cv_variable (to_udf +1)/ pudfh		/
	DATA cv_lo_d1  (to_udf +1)/ 0		/
	DATA cv_hi_d1  (to_udf +1)/ 0		/
	DATA cv_lo_d2  (to_udf +1)/ 0		/
	DATA cv_hi_d2  (to_udf +1)/ 0		/
	DATA cv_lo_d3  (to_udf +1)/ 0		/
	DATA cv_hi_d3  (to_udf +1)/ 0		/
	DATA cv_lo_d4  (to_udf +1)/ 0		/
	DATA cv_hi_d4  (to_udf +1)/ 0		/
	
	DATA cv_category (to_udf +2)/ cat_calc_var	/
	DATA cv_variable (to_udf +2)/ pudfz		/
	DATA cv_lo_d1  (to_udf +2)/ 0		/
	DATA cv_hi_d1  (to_udf +2)/ 0		/
	DATA cv_lo_d2  (to_udf +2)/ 0		/
	DATA cv_hi_d2  (to_udf +2)/ 0		/
	DATA cv_lo_d3  (to_udf +2)/ 0		/
	DATA cv_hi_d3  (to_udf +2)/ 0		/
	DATA cv_lo_d4  (to_udf +2)/ 0		/
	DATA cv_hi_d4  (to_udf +2)/ 0		/

**************************************************************************
	
	DATA cv_category (to_vdf +1)/ cat_calc_var	/
	DATA cv_variable (to_vdf +1)/ pvdfh		/
	DATA cv_lo_d1  (to_vdf +1)/ 0		/
	DATA cv_hi_d1  (to_vdf +1)/ 0		/
	DATA cv_lo_d2  (to_vdf +1)/ 0		/
	DATA cv_hi_d2  (to_vdf +1)/ 0		/
	DATA cv_lo_d3  (to_vdf +1)/ 0		/
	DATA cv_hi_d3  (to_vdf +1)/ 0		/
	DATA cv_lo_d4  (to_vdf +1)/ 0		/
	DATA cv_hi_d4  (to_vdf +1)/ 0		/
	
	DATA cv_category (to_vdf +2)/ cat_calc_var	/
	DATA cv_variable (to_vdf +2)/ pvdfz		/
	DATA cv_lo_d1  (to_vdf +2)/ 0		/
	DATA cv_hi_d1  (to_vdf +2)/ 0		/
	DATA cv_lo_d2  (to_vdf +2)/ 0		/
	DATA cv_hi_d2  (to_vdf +2)/ 0		/
	DATA cv_lo_d3  (to_vdf +2)/ 0		/
	DATA cv_hi_d3  (to_vdf +2)/ 0		/
	DATA cv_lo_d4  (to_vdf +2)/ 0		/
	DATA cv_hi_d4  (to_vdf +2)/ 0		/

**************************************************************************
	
	DATA cv_category (to_dudt+1)/ cat_file_var	/
	DATA cv_variable (to_dudt+1)/ pu   		/
	DATA cv_lo_d1  (to_dudt+1)/ 0		/
	DATA cv_hi_d1  (to_dudt+1)/ 0		/
	DATA cv_lo_d2  (to_dudt+1)/ 0		/
	DATA cv_hi_d2  (to_dudt+1)/ 0		/
	DATA cv_lo_d3  (to_dudt+1)/ 0		/
	DATA cv_hi_d3  (to_dudt+1)/ 0		/
	DATA cv_lo_d4  (to_dudt+1)/ -1		/
	DATA cv_hi_d4  (to_dudt+1)/ +1		/
	
**************************************************************************
	
	DATA cv_category (to_dvdt+1)/ cat_file_var	/
	DATA cv_variable (to_dvdt+1)/ pv   		/
	DATA cv_lo_d1  (to_dvdt+1)/ 0		/
	DATA cv_hi_d1  (to_dvdt+1)/ 0		/
	DATA cv_lo_d2  (to_dvdt+1)/ 0		/
	DATA cv_hi_d2  (to_dvdt+1)/ 0		/
	DATA cv_lo_d3  (to_dvdt+1)/ 0		/
	DATA cv_hi_d3  (to_dvdt+1)/ 0		/
	DATA cv_lo_d4  (to_dvdt+1)/ -1		/
	DATA cv_hi_d4  (to_dvdt+1)/ +1		/
	
**************************************************************************
	
	DATA cv_category (to_dwdt+1)/ cat_file_var	/
	DATA cv_variable (to_dwdt+1)/ pw   		/
	DATA cv_lo_d1  (to_dwdt+1)/ 0		/
	DATA cv_hi_d1  (to_dwdt+1)/ 0		/
	DATA cv_lo_d2  (to_dwdt+1)/ 0		/
	DATA cv_hi_d2  (to_dwdt+1)/ 0		/
	DATA cv_lo_d3  (to_dwdt+1)/ 0		/
	DATA cv_hi_d3  (to_dwdt+1)/ 0		/
	DATA cv_lo_d4  (to_dwdt+1)/ -1		/
	DATA cv_hi_d4  (to_dwdt+1)/ +1		/
	
**************************************************************************
	
	DATA cv_category (to_dtdt+1)/ cat_file_var	/
	DATA cv_variable (to_dtdt+1)/ ptemp		/
	DATA cv_lo_d1  (to_dtdt+1)/ 0		/
	DATA cv_hi_d1  (to_dtdt+1)/ 0		/
	DATA cv_lo_d2  (to_dtdt+1)/ 0		/
	DATA cv_hi_d2  (to_dtdt+1)/ 0		/
	DATA cv_lo_d3  (to_dtdt+1)/ 0		/
	DATA cv_hi_d3  (to_dtdt+1)/ 0		/
	DATA cv_lo_d4  (to_dtdt+1)/ -1		/
	DATA cv_hi_d4  (to_dtdt+1)/ +1		/
	
**************************************************************************
	
	DATA cv_category (to_ubps+1)/ cat_calc_var	/
	DATA cv_variable (to_ubps+1)/ pudf		/
	DATA cv_lo_d1  (to_ubps+1)/ 0		/
	DATA cv_hi_d1  (to_ubps+1)/ 0		/
	DATA cv_lo_d2  (to_ubps+1)/ 0		/
	DATA cv_hi_d2  (to_ubps+1)/ 0		/
	DATA cv_lo_d3  (to_ubps+1)/ 0		/
	DATA cv_hi_d3  (to_ubps+1)/ 0		/
	DATA cv_lo_d4  (to_ubps+1)/ 0		/
	DATA cv_hi_d4  (to_ubps+1)/ 0		/
	
	DATA cv_category (to_ubps+2)/ cat_calc_var	/
	DATA cv_variable (to_ubps+2)/ puad		/
	DATA cv_lo_d1  (to_ubps+2)/ 0		/
	DATA cv_hi_d1  (to_ubps+2)/ 0		/
	DATA cv_lo_d2  (to_ubps+2)/ 0		/
	DATA cv_hi_d2  (to_ubps+2)/ 0		/
	DATA cv_lo_d3  (to_ubps+2)/ 0		/
	DATA cv_hi_d3  (to_ubps+2)/ 0		/
	DATA cv_lo_d4  (to_ubps+2)/ 0		/
	DATA cv_hi_d4  (to_ubps+2)/ 0		/

	DATA cv_category (to_ubps+3)/ cat_calc_var	/
	DATA cv_variable (to_ubps+3)/ pucoe		/
	DATA cv_lo_d1  (to_ubps+3)/ 0		/
	DATA cv_hi_d1  (to_ubps+3)/ 0		/
	DATA cv_lo_d2  (to_ubps+3)/ 0		/
	DATA cv_hi_d2  (to_ubps+3)/ 0		/
	DATA cv_lo_d3  (to_ubps+3)/ 0		/
	DATA cv_hi_d3  (to_ubps+3)/ 0		/
	DATA cv_lo_d4  (to_ubps+3)/ 0		/
	DATA cv_hi_d4  (to_ubps+3)/ 0		/

	DATA cv_category (to_ubps+4)/ cat_calc_var	/
	DATA cv_variable (to_ubps+4)/ pubpi		/
	DATA cv_lo_d1  (to_ubps+4)/ 0		/
	DATA cv_hi_d1  (to_ubps+4)/ 0		/
	DATA cv_lo_d2  (to_ubps+4)/ 0		/
	DATA cv_hi_d2  (to_ubps+4)/ 0		/
	DATA cv_lo_d3  (to_ubps+4)/ 0		/
	DATA cv_hi_d3  (to_ubps+4)/ 0		/
	DATA cv_lo_d4  (to_ubps+4)/ 0		/
	DATA cv_hi_d4  (to_ubps+4)/ 0		/

	DATA cv_category (to_ubps+5)/ cat_calc_var	/
	DATA cv_variable (to_ubps+5)/ pdudt		/
	DATA cv_lo_d1  (to_ubps+5)/ 0		/
	DATA cv_hi_d1  (to_ubps+5)/ 0		/
	DATA cv_lo_d2  (to_ubps+5)/ 0		/
	DATA cv_hi_d2  (to_ubps+5)/ 0		/
	DATA cv_lo_d3  (to_ubps+5)/ 0		/
	DATA cv_hi_d3  (to_ubps+5)/ 0		/
	DATA cv_lo_d4  (to_ubps+5)/ 0		/
	DATA cv_hi_d4  (to_ubps+5)/ 0		/

**************************************************************************
	
	DATA cv_category (to_vbps+1)/ cat_calc_var	/
	DATA cv_variable (to_vbps+1)/ pvdf		/
	DATA cv_lo_d1  (to_vbps+1)/ 0		/
	DATA cv_hi_d1  (to_vbps+1)/ 0		/
	DATA cv_lo_d2  (to_vbps+1)/ 0		/
	DATA cv_hi_d2  (to_vbps+1)/ 0		/
	DATA cv_lo_d3  (to_vbps+1)/ 0		/
	DATA cv_hi_d3  (to_vbps+1)/ 0		/
	DATA cv_lo_d4  (to_vbps+1)/ 0		/
	DATA cv_hi_d4  (to_vbps+1)/ 0		/
	
	DATA cv_category (to_vbps+2)/ cat_calc_var	/
	DATA cv_variable (to_vbps+2)/ pvad		/
	DATA cv_lo_d1  (to_vbps+2)/ 0		/
	DATA cv_hi_d1  (to_vbps+2)/ 0		/
	DATA cv_lo_d2  (to_vbps+2)/ 0		/
	DATA cv_hi_d2  (to_vbps+2)/ 0		/
	DATA cv_lo_d3  (to_vbps+2)/ 0		/
	DATA cv_hi_d3  (to_vbps+2)/ 0		/
	DATA cv_lo_d4  (to_vbps+2)/ 0		/
	DATA cv_hi_d4  (to_vbps+2)/ 0		/

	DATA cv_category (to_vbps+3)/ cat_calc_var	/
	DATA cv_variable (to_vbps+3)/ pvcoe		/
	DATA cv_lo_d1  (to_vbps+3)/ 0		/
	DATA cv_hi_d1  (to_vbps+3)/ 0		/
	DATA cv_lo_d2  (to_vbps+3)/ 0		/
	DATA cv_hi_d2  (to_vbps+3)/ 0		/
	DATA cv_lo_d3  (to_vbps+3)/ 0		/
	DATA cv_hi_d3  (to_vbps+3)/ 0		/
	DATA cv_lo_d4  (to_vbps+3)/ 0		/
	DATA cv_hi_d4  (to_vbps+3)/ 0		/

	DATA cv_category (to_vbps+4)/ cat_calc_var	/
	DATA cv_variable (to_vbps+4)/ pvbpi		/
	DATA cv_lo_d1  (to_vbps+4)/ 0		/
	DATA cv_hi_d1  (to_vbps+4)/ 0		/
	DATA cv_lo_d2  (to_vbps+4)/ 0		/
	DATA cv_hi_d2  (to_vbps+4)/ 0		/
	DATA cv_lo_d3  (to_vbps+4)/ 0		/
	DATA cv_hi_d3  (to_vbps+4)/ 0		/
	DATA cv_lo_d4  (to_vbps+4)/ 0		/
	DATA cv_hi_d4  (to_vbps+4)/ 0		/

	DATA cv_category (to_vbps+5)/ cat_calc_var	/
	DATA cv_variable (to_vbps+5)/ pdvdt		/
	DATA cv_lo_d1  (to_vbps+5)/ 0		/
	DATA cv_hi_d1  (to_vbps+5)/ 0		/
	DATA cv_lo_d2  (to_vbps+5)/ 0		/
	DATA cv_hi_d2  (to_vbps+5)/ 0		/
	DATA cv_lo_d3  (to_vbps+5)/ 0		/
	DATA cv_hi_d3  (to_vbps+5)/ 0		/
	DATA cv_lo_d4  (to_vbps+5)/ 0		/
	DATA cv_hi_d4  (to_vbps+5)/ 0		/

**************************************************************************
	
	DATA cv_category (to_sali+1)/ cat_file_var	/
	DATA cv_variable (to_sali+1)/ psalt		/
	DATA cv_lo_d1  (to_sali+1)/ 0		/
	DATA cv_hi_d1  (to_sali+1)/ 0		/
	DATA cv_lo_d2  (to_sali+1)/ 0		/
	DATA cv_hi_d2  (to_sali+1)/ 0		/
	DATA cv_lo_d3  (to_sali+1)/ 0		/
	DATA cv_hi_d3  (to_sali+1)/ 0		/
	DATA cv_lo_d4  (to_sali+1)/ 0		/
	DATA cv_hi_d4  (to_sali+1)/ 0		/

**************************************************************************
	
	DATA cv_category (to_dens+1)/ cat_file_var	/
	DATA cv_variable (to_dens+1)/ ptemp		/
	DATA cv_lo_d1  (to_dens+1)/ 0		/
	DATA cv_hi_d1  (to_dens+1)/ 0		/
	DATA cv_lo_d2  (to_dens+1)/ 0		/
	DATA cv_hi_d2  (to_dens+1)/ 0		/
	DATA cv_lo_d3  (to_dens+1)/ 0		/
	DATA cv_hi_d3  (to_dens+1)/ 0		/
	DATA cv_lo_d4  (to_dens+1)/ 0		/
	DATA cv_hi_d4  (to_dens+1)/ 0		/

	DATA cv_category (to_dens+2)/ cat_file_var	/
	DATA cv_variable (to_dens+2)/ psalt		/
	DATA cv_lo_d1  (to_dens+2)/ 0		/
	DATA cv_hi_d1  (to_dens+2)/ 0		/
	DATA cv_lo_d2  (to_dens+2)/ 0		/
	DATA cv_hi_d2  (to_dens+2)/ 0		/
	DATA cv_lo_d3  (to_dens+2)/ 0		/
	DATA cv_hi_d3  (to_dens+2)/ 0		/
	DATA cv_lo_d4  (to_dens+2)/ 0		/
	DATA cv_hi_d4  (to_dens+2)/ 0		/

**************************************************************************
	
	DATA cv_category (to_ubp +1)/ cat_calc_var	/
	DATA cv_variable (to_ubp +1)/ pubpi		/
	DATA cv_lo_d1  (to_ubp +1)/ 0		/
	DATA cv_hi_d1  (to_ubp +1)/ 0		/
	DATA cv_lo_d2  (to_ubp +1)/ 0		/
	DATA cv_hi_d2  (to_ubp +1)/ 0		/
	DATA cv_lo_d3  (to_ubp +1)/ 0	/
	DATA cv_hi_d3  (to_ubp +1)/ 0	/
	DATA cv_lo_d4  (to_ubp +1)/ 0		/
	DATA cv_hi_d4  (to_ubp +1)/ 0		/
	
	DATA cv_category (to_ubp +2)/ cat_calc_var	/
	DATA cv_variable (to_ubp +2)/ pubps		/
	DATA cv_lo_d1  (to_ubp +2)/ 0		/
	DATA cv_hi_d1  (to_ubp +2)/ 0		/
	DATA cv_lo_d2  (to_ubp +2)/ 0		/
	DATA cv_hi_d2  (to_ubp +2)/ 0		/
	DATA cv_lo_d3  (to_ubp +2)/ 0		/
	DATA cv_hi_d3  (to_ubp +2)/ 0		/
	DATA cv_lo_d4  (to_ubp +2)/ 0		/
	DATA cv_hi_d4  (to_ubp +2)/ 0		/

**************************************************************************
	
	DATA cv_category (to_vbp +1)/ cat_calc_var	/
	DATA cv_variable (to_vbp +1)/ pvbpi		/
	DATA cv_lo_d1  (to_vbp +1)/ 0		/
	DATA cv_hi_d1  (to_vbp +1)/ 0		/
	DATA cv_lo_d2  (to_vbp +1)/ 0		/
	DATA cv_hi_d2  (to_vbp +1)/ 0		/
	DATA cv_lo_d3  (to_vbp +1)/ 0	/
	DATA cv_hi_d3  (to_vbp +1)/ 0	/
	DATA cv_lo_d4  (to_vbp +1)/ 0		/
	DATA cv_hi_d4  (to_vbp +1)/ 0		/
	
	DATA cv_category (to_vbp +2)/ cat_calc_var	/
	DATA cv_variable (to_vbp +2)/ pvbps		/
	DATA cv_lo_d1  (to_vbp +2)/ 0		/
	DATA cv_hi_d1  (to_vbp +2)/ 0		/
	DATA cv_lo_d2  (to_vbp +2)/ 0		/
	DATA cv_hi_d2  (to_vbp +2)/ 0		/
	DATA cv_lo_d3  (to_vbp +2)/ 0		/
	DATA cv_hi_d3  (to_vbp +2)/ 0		/
	DATA cv_lo_d4  (to_vbp +2)/ 0		/
	DATA cv_hi_d4  (to_vbp +2)/ 0		/

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	
	DATA cv_category (to_cmsk+1)/ cat_file_var	/
	DATA cv_variable (to_cmsk+1)/ ptemp		/
	DATA cv_lo_d1  (to_cmsk+1)/ 0		/
	DATA cv_hi_d1  (to_cmsk+1)/ 0		/
	DATA cv_lo_d2  (to_cmsk+1)/ 0		/
	DATA cv_hi_d2  (to_cmsk+1)/ 0		/
	DATA cv_lo_d3  (to_cmsk+1)/ -1		/
	DATA cv_hi_d3  (to_cmsk+1)/ +1		/
	DATA cv_lo_d4  (to_cmsk+1)/ 0		/
	DATA cv_hi_d4  (to_cmsk+1)/ 0		/

	DATA cv_category (to_cmsk+2)/ cat_file_var	/
	DATA cv_variable (to_cmsk+2)/ psalt		/
	DATA cv_lo_d1  (to_cmsk+2)/ 0		/
	DATA cv_hi_d1  (to_cmsk+2)/ 0		/
	DATA cv_lo_d2  (to_cmsk+2)/ 0		/
	DATA cv_hi_d2  (to_cmsk+2)/ 0		/
	DATA cv_lo_d3  (to_cmsk+2)/ -1		/
	DATA cv_hi_d3  (to_cmsk+2)/ +1		/
	DATA cv_lo_d4  (to_cmsk+2)/ 0 		/
	DATA cv_hi_d4  (to_cmsk+2)/ 0 		/

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	
	DATA cv_category (to_w_t +1)/ cat_file_var	/
	DATA cv_variable (to_w_t +1)/ pu		/
	DATA cv_lo_d1  (to_w_t +1)/ -1		/
	DATA cv_hi_d1  (to_w_t +1)/ 0		/
	DATA cv_lo_d2  (to_w_t +1)/ -1		/
	DATA cv_hi_d2  (to_w_t +1)/ 0		/
	DATA cv_lo_d3  (to_w_t +1)/ 0		/
	DATA cv_hi_d3  (to_w_t +1)/ 0		/
	DATA cv_lo_d4  (to_w_t +1)/ 0		/
	DATA cv_hi_d4  (to_w_t +1)/ 0		/

	DATA cv_category (to_w_t +2)/ cat_file_var	/
	DATA cv_variable (to_w_t +2)/ pv		/
	DATA cv_lo_d1  (to_w_t +2)/ -1		/
	DATA cv_hi_d1  (to_w_t +2)/ 0		/
	DATA cv_lo_d2  (to_w_t +2)/ -1		/
	DATA cv_hi_d2  (to_w_t +2)/ 0		/
	DATA cv_lo_d3  (to_w_t +2)/ 0		/
	DATA cv_hi_d3  (to_w_t +2)/ 0		/
	DATA cv_lo_d4  (to_w_t +2)/ 0		/
	DATA cv_hi_d4  (to_w_t +2)/ 0		/

**************************************************************************
	
	DATA cv_category (to_cair+1)/ cat_file_var	/
	DATA cv_variable (to_cair+1)/ pairt_special	/
	DATA cv_lo_d1  (to_cair+1)/ 0		/
	DATA cv_hi_d1  (to_cair+1)/ 0		/
	DATA cv_lo_d2  (to_cair+1)/ 0		/
	DATA cv_hi_d2  (to_cair+1)/ 0		/
	DATA cv_lo_d3  (to_cair+1)/ 0		/
	DATA cv_hi_d3  (to_cair+1)/ 0		/
	DATA cv_lo_d4  (to_cair+1)/ 0		/
	DATA cv_hi_d4  (to_cair+1)/ 0		/

**************************************************************************
	
	DATA cv_category (to_air +1)/ cat_file_var	/
	DATA cv_variable (to_air +1)/ ptemp		/
	DATA cv_lo_d1  (to_air +1)/ 0		/
	DATA cv_hi_d1  (to_air +1)/ 0		/
	DATA cv_lo_d2  (to_air +1)/ 0		/
	DATA cv_hi_d2  (to_air +1)/ 0		/
	DATA cv_lo_d3  (to_air +1)/ 0		/
	DATA cv_hi_d3  (to_air +1)/ 0		/
	DATA cv_lo_d4  (to_air +1)/ 0		/
	DATA cv_hi_d4  (to_air +1)/ 0		/

	DATA cv_category (to_air +2)/ cat_calc_var	/
	DATA cv_variable (to_air +2)/ pcair_special	/
	DATA cv_lo_d1  (to_air +2)/ 0		/
	DATA cv_hi_d1  (to_air +2)/ 0		/
	DATA cv_lo_d2  (to_air +2)/ 0		/
	DATA cv_hi_d2  (to_air +2)/ 0		/
	DATA cv_lo_d3  (to_air +2)/ 0		/
	DATA cv_hi_d3  (to_air +2)/ 0		/
	DATA cv_lo_d4  (to_air +2)/ 0		/
	DATA cv_hi_d4  (to_air +2)/ 0		/

	END
