        SUBROUTINE SETUP_DEFAULTS
        
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
*       Sets ppl+ defaults & the options for SHADE for initial calls 
*       J Davison JISAO/PMEL/TMAP 8.24.88/7.27.89
*	*jd* 2.11.92 Initialize item_* variables here
*	*jd* 5.28.92 Inititalize sqfill to on for AREA command
*	*jd* Use area bundles for DEC , but not for ATC -- too few available
* 	*jd  Set default values for CENTERED & SET_DELTA in CONT.INC 8.15.95

#ifdef unix
        include         'tmap_pplv11inc/shade_vars.cmn'
#else
        INCLUDE         'tmap_pplv11inc:tmap_shade_vars.inc'
#endif

#ifdef unix
        include         'tmap_pplv11inc/parampl5_dat.decl'
        include         'pplv11inc/PARAMPL5.DAT'
        include         'tmap_pplv11inc/axis_inc.decl'
        include         'pplv11inc/AXIS.INC'
        include         'tmap_pplv11inc/plt_inc.decl'
        include         'pplv11inc/PLT.INC'
        include         'tmap_pplv11inc/gkscm2.cmn'
        include         'pplv11inc/ARFILL.INC'
        include         'tmap_pplv11inc/cont_inc.decl'
        INCLUDE         'pplv11inc/CONT.INC'
#else
        INCLUDE         'tmap_pplv11inc:tmap_parampl5.dat'
        INCLUDE         'tmap_pplv11inc:tmap_axis.inc'
        INCLUDE         'tmap_pplv11inc:tmap_plt.inc'
        INCLUDE         'tmap_pplv11inc:tmap_gkscm2.inc'
        include         'pplinc:ARFILL.INC'
        INCLUDE 	'pplinc:CONT.INC'
#endif

	real		rem
	external	ws_types_init,pplp_ver_data

******************************************************************************

*	INITIALIZE WS/METAFILE THINGS
	ws_lun  = 111
	ws_file = 'ppl_port'

	meta_wsid = 101
*       MOD *JD* 3.25.91 Ultrix doesn't like LUN >= 19 for GOPWK
*	meta_lun  = 112
	meta_lun  = 18
*	meta_file = 'metafile.plt' -- initialize in block data

	tmp_wsid = 102
*	tmp_lun  = 113
	tmp_lun  = 17
	tmp_file = 'meta.tmp'

	meta_open = .false.
	meta_actv = .false.
	meta_clsplt = .false.

	vpt_was_set = .false.
	first_meta  = .true.

	item_gclrwk    = 101
	item_ws_xform  = 102
	item_ndx_range = 103
	item_spectrum  = 104
	item_protect   = 105
	item_fill_ndx  = 106
	item_pplpver   = 107

	bad_val = 1

*	SHADE DEFAULT COLORS
	ncl = 6
	custom_colors = .true.
*#ifdef atc_gks
	area_bundles  = .false.
*#else
*	area_bundles  = .true.
*#endif
	mncl = 0

	slvl(1) = 0.0
	sr(1) = 0.8
	sg(1) = 0.0
	sb(1) = 1.0

	slvl(2) = 10.0
	sr(2) = 0.3
	sg(2) = 0.2
	sb(2) = 1.0

	slvl(3) = 33.0
	sr(3) = 0.0
	sg(3) = 0.6
	sb(3) = 0.3

	slvl(4) = 66.0
	sr(4) = 1.0
	sg(4) = 1.0
	sb(4) = 0.0

	slvl(5) = 90.0
	sr(5) = 1.0
	sg(5) = 0.0
	sb(5) = 0.0

	slvl(6) = 100.0
	sr(6) = 0.6
	sg(6) = 0.0
	sb(6) = 0.0

*       DO THE KEY
	do_key	     	= .true.
        vertical_key 	= .false.
	kuser_loc 	= .false.

	klab_size = 0.0
	klab_dig  = 3
	klab_len  = 9
	klab_incr = 0

        if (vertical_key) then
           rem = width - (xlen + xorg)
           kx_lo = 1000.0 * (xorg + xlen + 0.25 * rem)
           kx_hi = 1000.0 * (width - 0.1 * rem)
           ky_lo = 1000.0 *  yorg
           ky_hi = 1000.0 * (yorg + ylen)
        else
           rem = height - (ylen + yorg)
           kx_lo = 1000.0 *  xorg
           kx_hi = 1000.0 * (xorg + xlen)
           ky_lo = 1000.0 * (yorg + ylen + 0.25 * rem)
           ky_hi = 1000.0 * (height - 0.1 * rem)
        end if

*	INITIALIZE SQFFLG TO TRUE
	sqfflg = .true.

*       INITIALIZE DRAFTSMAN AND SPLINE_TENSION
        draftsman=.false.
        spline_tension=0.

*	INITIALIZE CONT.INC ADD ONS
	centered  = .false.
	set_delta = .false.

	return
	end




