	LOGICAL FUNCTION ITSA_MODULO_CURVI_AXIS( xfield, yfield, mr_xpos, 
     .                                           mr_ypos, idim )

*       determine whether a particular curvilinear context grid axis is a modulo one

*       programmer - jonathan callahan
*       NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

*       V420: 04/97

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xcontext.cmn'
	include 'ferret_cmn/xvariables.cmn'
	include 'tmap_format/tmap_dims.parm'
	include 'tmap_format/tmap_errors.parm'
	include 'tmap_format/xtm_grid.cmn_text'
	include 'ferret_cmn/xcurvilinear.cmn'
#else   
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN_TEXT'
#endif  

*       calling argument declarations:
	LOGICAL ITSA_MODULO_CURVI_AXIS_SUB
	REAL    xfield(*), yfield(*)
	INTEGER	mr_xpos, mr_ypos, idim

c *kob* 8/97 had to move the continuation marker to col 5 - it was in col 8
c       this bombed aix compiler, though it seems to work on others
	ITSA_MODULO_CURVI_AXIS = ITSA_MODULO_CURVI_AXIS_SUB( xfield, yfield,
     1    mr_lo_s1(mr_xpos), mr_hi_s1(mr_xpos),
     2    mr_lo_s2(mr_xpos), mr_hi_s2(mr_xpos),
     3    mr_lo_s1(mr_ypos), mr_hi_s1(mr_ypos),
     4    mr_lo_s2(mr_ypos), mr_hi_s2(mr_ypos),
     5    idim )

	RETURN
	END
	
