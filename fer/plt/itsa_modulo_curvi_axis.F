	LOGICAL FUNCTION ITSA_MODULO_CURVI_AXIS( xfield, yfield, mr_xpos, 
     .                                           mr_ypos, idim )

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
	
