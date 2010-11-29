	BLOCK DATA WS_TYPES_INIT

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
*	Initialize wstype constants for either DEC or ATC GKS
*	J Davison 2.13.92
*	Mod to support XGKS *jd* 8.10.93

#ifdef unix
#   ifndef atc_gks
#      ifndef xgks
	include		'/usr/include/GKS3D/GKS3Dforbnd.f'
#      else
	include		'gkspar.inc'
#      endif
#   endif
	include		'ws_types.cmn'
        include         'gkscm2.cmn'
#else
	include		'sys$library:gksdefs.bnd'
	include		'tmap_pplv11inc:ws_types.inc'
        INCLUDE         'tmap_pplv11inc:tmap_gkscm2.inc'
#endif

*******************************************************************************

#ifdef atc_gks
*	ATC GKS values for workstation types

        data         	ws_xwindow/5350/,
     .			ws_uis/-1/,
     .                  ws_meta_in/1001/,
     .                  ws_meta_out/1002/,
     .                  ws_ps_mono/1901/,
     .                  ws_ps_color/1901/,
     .                  ws_tek4014/2500/,
     .                  ws_tek4107/3100/
#elif xgks
*	 XGKS values for workstation types

      data              ws_xwindow/4/,
     .			ws_uis/-1/,
     .                  ws_meta_in/-1/,
     .                  ws_meta_out/-1/,
     .                  ws_ps_mono/-1/,
     .                  ws_ps_color/-1/,
     .                  ws_tek4107/-1/
#else
*	DEC GKS values for workstation types

      data              ws_xwindow/0/,
     .			ws_uis/gvsii/,
     .                  ws_meta_in/gminpt/,
     .                  ws_meta_out/gmoutp/,
     .                  ws_ps_mono/gptsc/,
     .                  ws_ps_color/gcptsc/,
     .                  ws_tek4014/gteko/,
     .                  ws_tek4107/gt41o/
#endif

*	INITIALIZE METAFILE NAME

	data		meta_file/'metafile.plt'/


	end






