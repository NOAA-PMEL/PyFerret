	BLOCK DATA XPLOT_STATE_DATA

* initialization for plotted output state variables

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 6/4/86
* revision 1.0 - 6/28/88 - major re-write for multiple windows
* revision 1.1 - 8/26/88 - added multiple viewports and consolidated window
*			   sizing
* revision 1.2 - 1/24/89 - added ppl_interrupted
* V200: 10/29/89 - "FULL" viewport is just like others.  Default is NONE
* V230:  11/4/92 - 4 corner viewports (ll,lr,ul,ur) with clipping ON
* V301:  22/2/94 - changed wn_scale(1) from 1.0 to 0.7

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/xplot_state.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:XPLOT_STATE.CMN'
#endif

	INTEGER i

* local parameter declarations:
	INTEGER		mwn_maxm1,
     .			mvp_maxm9
	PARAMETER     ( mwn_maxm1  = max_windows - 1,
     .			mvp_maxm9 = max_viewport - 9 )

	DATA	pplus_started	/ .FALSE. /
	DATA	ppl_interrupted / .FALSE. /

	DATA curr_seg_name / 0 /	! to be incremented for each new sgmt

	DATA	wn_open	  / max_windows * .FALSE. /
	DATA	wn_active / max_windows * .FALSE. /
	DATA	wn_scale  /wn_dflt_scale, mwn_maxm1 * wn_dflt_scale /

* lower left default window positions
	DATA xlowlef / .20, .004, .20,  .008, .06 , 0.09, .012, 0.10  /
	DATA ylowlef / .20, .004, .008, .20,  .012, 0.06, .10 , 0.05  /

* various pre-programmed viewport choices
	DATA	vp_num / mvp_dflt /

	DATA	vp_name ( mvp_dflt ) / 'NONE' /,	! whole window
     .		vp_size ( mvp_dflt ) / 1.0 /,
     .		vp_xorg ( mvp_dflt ) / 0.0 /,
     .		vp_yorg ( mvp_dflt ) / 0.0 /,
     .		vp_xclip( mvp_dflt ) / unspecified_val4 /,
     .		vp_yclip( mvp_dflt ) / unspecified_val4 /

	DATA	vp_name ( 1 ) / 'UPPER' /,	! upper half
     .		vp_size ( 1 ) / 1.0 /,
     .		vp_xorg ( 1 ) / 0.0 /,
     .		vp_yorg ( 1 ) / 0.5 /,
     .		vp_xclip( 1 ) / 1.0 /,
     .		vp_yclip( 1 ) / 1.0 /

	DATA	vp_name ( 2 ) / 'LOWER' /,	! lower half
     .		vp_size ( 2 ) / 1.0 /,
     .		vp_xorg ( 2 ) / 0.0 /,
     .		vp_yorg ( 2 ) / 0.0 /,
     .		vp_xclip( 2 ) / 1.0 /,
     .		vp_yclip( 2 ) / 0.5 /

	DATA	vp_name ( 3 ) / 'RIGHT' /,	! right half
     .		vp_size ( 3 ) / 1.0 /,
     .		vp_xorg ( 3 ) / 0.5 /,
     .		vp_yorg ( 3 ) / 0.0 /,
     .		vp_xclip( 3 ) / 1.0 /,
     .		vp_yclip( 3 ) / 1.0 /

	DATA	vp_name ( 4 ) / 'LEFT' /,	! left half
     .		vp_size ( 4 ) / 1.0 /,
     .		vp_xorg ( 4 ) / 0.0 /,
     .		vp_yorg ( 4 ) / 0.0 /,
     .		vp_xclip( 4 ) / 0.5 /,
     .		vp_yclip( 4 ) / 1.0 /

	DATA	vp_name ( 5 ) / 'LL' /,	! Lower Left
     .		vp_size ( 5 ) / 0.5 /,
     .		vp_xorg ( 5 ) / 0.0 /,
     .		vp_yorg ( 5 ) / 0.0 /,
     .		vp_xclip( 5 ) / 0.5 /,
     .		vp_yclip( 5 ) / 0.5 /

	DATA	vp_name ( 6 ) / 'LR' /,	! Lower Right
     .		vp_size ( 6 ) / 0.5 /,
     .		vp_xorg ( 6 ) / 0.5 /,
     .		vp_yorg ( 6 ) / 0.0 /,
     .		vp_xclip( 6 ) / 1.0 /,
     .		vp_yclip( 6 ) / 0.5 /

	DATA	vp_name ( 7 ) / 'UL' /,	! Upper Left
     .		vp_size ( 7 ) / 0.5 /,
     .		vp_xorg ( 7 ) / 0.0 /,
     .		vp_yorg ( 7 ) / 0.5 /,
     .		vp_xclip( 7 ) / 0.5 /,
     .		vp_yclip( 7 ) / 1.0 /

	DATA	vp_name ( 8 ) / 'UR' /,	! Upper Right
     .		vp_size ( 8 ) / 0.5 /,
     .		vp_xorg ( 8 ) / 0.5 /,
     .		vp_yorg ( 8 ) / 0.5 /,
     .		vp_xclip( 8 ) / 1.0 /,
     .		vp_yclip( 8 ) / 1.0 /

	DATA	vp_name ( 9 ) / 'FULL' /,	! FULL window
     .		vp_size ( 9 ) / 1.0 /,
     .		vp_xorg ( 9 ) / 0.0 /,
     .		vp_yorg ( 9 ) / 0.0 /,
     .		vp_xclip( 9 ) / 1.0 /,
     .		vp_yclip( 9 ) / 1.0 /

	DATA	(vp_name(i),i=10,max_viewport) /mvp_maxm9*unspecified_name4/

	END
