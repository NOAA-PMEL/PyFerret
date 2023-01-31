        SUBROUTINE SETUP_DEFAULTS

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

*       Sets ppl+ defaults & the options for SHADE for initial calls 
*       J Davison JISAO/PMEL/TMAP 8.24.88/7.27.89
*	*jd* 2.11.92 Initialize item_* variables here
*	*jd* 5.28.92 Inititalize sqfill to on for AREA command
*	*jd* Use area bundles for DEC , but not for ATC -- too few available
* 	*jd* Set default values for CENTERED & SET_DELTA in CONT.INC 8.15.95
*       *jd* Set default values for patterns 10.98
*       *jd* Initialize rgb_mapping 10.98
* 3/99 *kob* 	move var init off of decl list  - cleaning up after jerry.
* V552 5/03 *acm* Move shade key setup to routine SHADE_KEY
* V553 6/03 *acm* Keep the initialization of do_key here.
* V65 *acm   2/10 If Open-ended levels were used, symbol LEV_OPNLEVS saves 
* the /LEV command needed to duplicate the levels
* PyFr *kms* 5/13 Add alpha channel (opaque) variables SA

        IMPLICIT NONE
        include         'shade_vars.cmn'
        include         'parampl5_dat.decl'
        include         'PARAMPL5.DAT'
        include         'axis_inc.decl'
        include         'AXIS.INC'
        include         'plt_inc.decl'
        include         'PLT.INC'
        include         'gkscm2.cmn'
        include         'ARFILL.INC'
        include         'cont_inc.decl'
        INCLUDE         'CONT.INC'

	real		rem
	external	ws_types_init,pplp_ver_data

        integer         i,solid
        DATA            solid/21/
        integer         set_default_patterns(20)
        DATA            set_default_patterns
     .          /1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20/

        character*25    set_pattern_list(21)
        DATA            set_pattern_list
     . /'TINY_TRIANGLES', 
     .  'TINY_GRID',
     .  'TINY_OPEN_SQUARES',
     .	'TINY_SQUARES',
     .	'LITE_HORIZONTAL',
     .	'LITE_VERTICAL',
     .	'DARK_HORIZONTAL',
     .	'DARK_VERTICAL',
     .	'DARK_UP_LEFT_TO_RIGHT',
     .	'LITE_UP_LEFT_TO_RIGHT',
     .	'DARK_UP_RIGHT_TO_LEFT',
     .	'LITE_UP_RIGHT_TO_LEFT',
     .	'LARGE_GRID',
     .	'ANGLED_GRID',
     .	'DOUBLE_VERTICAL',
     .	'DOUBLE_HORIZONTAL',
     .	'BALL_BEARINGS',
     .	'WEAVE',
     .	'FISH_SCALE',
     .	'BRICK',
     .  'SOLID'/

******************************************************************************

*	INITIALIZE WS/METAFILE THINGS
	ws_lun  = 111
	ws_file = 'ppl_port'

	tmp_wsid = 102
	tmp_lun  = 17
	tmp_file = 'meta.tmp'

	vpt_was_set = .false.

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
	area_bundles  = .false.
	mncl = 0

*       RGB MAPPING PARAMETERS
        percent  = 0
        by_value = 1
        by_level = 2

        rgb_mapping = percent

	slvl(1) = 0.0
	sr(1) = 0.8
	sg(1) = 0.0
	sb(1) = 1.0
	sa(1) = 1.0

	slvl(2) = 10.0
	sr(2) = 0.3
	sg(2) = 0.2
	sb(2) = 1.0
	sa(2) = 1.0

	slvl(3) = 33.0
	sr(3) = 0.0
	sg(3) = 0.6
	sb(3) = 0.3
	sa(3) = 1.0

	slvl(4) = 66.0
	sr(4) = 1.0
	sg(4) = 1.0
	sb(4) = 0.0
	sa(4) = 1.0

	slvl(5) = 90.0
	sr(5) = 1.0
	sg(5) = 0.0
	sb(5) = 0.0
	sa(5) = 1.0

	slvl(6) = 100.0
	sr(6) = 0.6
	sg(6) = 0.0
	sb(6) = 0.0
	sa(6) = 1.0

* negative overriding_alpha means not activated
        overriding_alpha = -1.0;

*       DO THE KEY --- move most of this setup to routine SHADE_KEY; values
*       need to be initialized with each call.

	do_key	     	= .true.

*	when open-ended levels; create the /LEV command needed to 
*	duplicate them.

	lev_str = 'none'

*	INITIALIZE SQFFLG TO TRUE
	sqfflg = .true.

*       INITIALIZE DRAFTSMAN AND SPLINE_TENSION
        draftsman=.false.
        spline_tension=0.

*	INITIALIZE CONT.INC ADD ONS
	centered  = .false.
	set_delta = .false.

*       DEFAULTS FOR PATTERN SUPPORT
        do 10 i = 1, 20
           default_patterns(i) = set_default_patterns(i)
           pattern_list(i) = set_pattern_list(i)
 10     end do

*       DEFAULT PATTERN IS SOLID ("pattern" 21)
        npat = 1
        patterns(1) = solid
        pattern_list(solid) = set_pattern_list(solid)

	return
	end




