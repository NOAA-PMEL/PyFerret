	subroutine query_xgks_wkstn(zwstype,error,max_pline,max_pmark,
     1	max_text,max_fill_area,max_pattern,max_color)

c
c
c
c       Support queries on metafile workstations
c
c v491 *kob* 2/98 - moved this out of ppladd_open_metafile

	implicit none
#ifdef unix
	include 'tmap_pplv11inc/gkscm2.cmn'! with declarations    
	include 'tmap_pplv11inc/ws_types.cmn'
#else
	INCLUDE 'tmap_pplv11inc:tmap_gkscm2.inc'! WITH DECLARATIONS    
	INCLUDE 'TMAP_PPLV11INC:WS_TYPES.CMN'
#endif
	integer zwstype,error,max_pline,max_pmark
	integer max_text, max_fill_area, max_pattern,max_color
	LOGICAL		ITS_BATCH_GRAPHICS, batch

	if (.not.its_batch_graphics() .and. zwstype .eq. ws_xwindow) then
	   call gqlwk(zwstype,error,max_pline,max_pmark,
     1	   max_text,max_fill_area,max_pattern,max_color)
	else
	   error = 0
	   max_pline = 20
	   max_pmark = 20
	   max_text = 20
	   max_fill_area = 20
	   max_pattern = 20
	   max_color = 256
	endif
	return
	end
