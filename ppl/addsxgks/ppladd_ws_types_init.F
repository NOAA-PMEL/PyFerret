	BLOCK DATA WS_TYPES_INIT

*	Initialize wstype constants for either DEC or ATC GKS
*	J Davison 2.13.92
*	Mod to support XGKS *jd* 8.10.93

#ifdef unix
#   ifndef atc_gks
#      ifndef xgks
	include		'/usr/include/GKS3D/GKS3Dforbnd.f'
#      else
	include		'tmap_pplv11inc/gkspar.inc'
#      endif
#   endif
	include		'tmap_pplv11inc/ws_types.cmn'
        include         'tmap_pplv11inc/gkscm2.cmn'
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






