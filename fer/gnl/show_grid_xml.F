	SUBROUTINE SHOW_GRID_XML( lun, grid )

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
*
*  list the axes for a grid in xml-style format

* programmer - Ansley Manke, based on show_grid.F
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* 6-Aug-2003

c	include	'ferret.parm'
c	include	'slash.parm'
c	include 'errmsg.parm'
c	include 'xprog_state.cmn'
c	include 'xrisc.cmn'



	include 'tmap_dims.parm'
#	include "tmap_dset.parm"
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include	'ferret.parm'
	include	'xvariables.cmn'
	include	'xtext_info.cmn'
	include	'xprog_state.cmn'
	include	'xrisc.cmn'


* calling argument declarations
	INTEGER lun, grid

* local variable declarations:
	INTEGER TM_LENSTR1, TM_LENSTR, idim, slen, line
        CHARACTER*1 axdir(4)
        DATA axdir/ 'x', 'y', 'z', 't'/


 2020   FORMAT ('<grid name="',A,'">' )
 2030   FORMAT( '<', A1, 'axis>' ,A , '</', A1, 'axis>' )
 2040   FORMAT ('</grid>')

* list the 4 axes of the grid

        slen = TM_LENSTR1(grid_name(grid))
	WRITE ( risc_buff, 2020 ) grid_name(grid)(1:slen)
	CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

	DO 110 idim = 1, 4
           line = grid_line(idim, grid)
           slen = TM_LENSTR(line_name(line))
           IF (slen .GT. 0) THEN
              slen = TM_LENSTR1(line_name(line))
              WRITE (risc_buff, 2030 ) axdir(idim),
     .           line_name(line)(1:slen), axdir(idim)
	      CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
           ENDIF
 110	CONTINUE

        WRITE ( risc_buff, 2040 )
        CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

        RETURN

	END

