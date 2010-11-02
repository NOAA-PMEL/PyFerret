	SUBROUTINE SHOW_GRID_XML( lun, grid, dset, cx_expr )

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
* V580 *acm* 11/4/04 add <axes> tags
* v6.5 *acm* 3/10 Cleanup - remove commented-out includes
* V6.65*acm* 11/10 If an untransformed axis is of length 1, write out
*                 its information (LAS ticket 969)
* V665 11/10 *acm* Fix bug 1177: preserve case of varnames and axis names in xml output
*                  Also fixing LAS ticket 969. If index_hi=index_lo, check whether its a
*                  one-point axis or an axis that has been transformed away.

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
	include	'xcontext.cmn'


* calling argument declarations
	INTEGER lun, grid, dset, cx_expr

* local variable declarations:
	INTEGER TM_LENSTR1, TM_LENSTR, idim, slen, line, nlen
        CHARACTER*1 axdir(4)
	CHARACTER*2048 outstring
	CHARACTER*130  CD_AXIS_NAME, out_name
        DATA axdir/ 'x', 'y', 'z', 't'/


 2020   FORMAT ('<grid name="',A,'">' )
 2030   FORMAT ('<axes>' )
 2040   FORMAT( '<', A1, 'axis>' ,A , '</', A1, 'axis>' )
 2050   FORMAT ('</axes>')
 2060   FORMAT ('</grid>')

* list the 4 axes of the grid

        CALL ESCAPE_FOR_XML (grid_name(grid), outstring, slen)
	WRITE ( risc_buff, 2020 ) outstring(1:slen)
	CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
	WRITE ( risc_buff, 2030 ) 
	CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

	DO 110 idim = 1, 4
           line = grid_line(idim, grid)
           slen = TM_LENSTR(line_name(line))

           IF (line .GT. 0  .AND.  slen .GT. 0 .AND.
     .         cx_expr .EQ. unspecified_int4) THEN
	      out_name = CD_AXIS_NAME ( dset, grid, idim, nlen )
              CALL ESCAPE_FOR_XML (out_name, outstring, slen)
              WRITE (risc_buff, 2040 ) axdir(idim),
     .           outstring(1:slen), axdir(idim)
	      CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
           ENDIF

           IF (line .GT. 0  .AND.  slen .GT. 0 .AND.
     .         cx_expr .NE. unspecified_int4) THEN

              IF ( cx_hi_ss(cx_expr,idim).NE.unspecified_int4 .AND. 
     .	           cx_lo_ss(cx_expr,idim).NE.unspecified_int4 ) THEN
                 out_name = CD_AXIS_NAME ( dset, grid, idim, nlen )
                 slen = TM_LENSTR1(out_name)
                 WRITE (risc_buff, 2040 ) axdir(idim),
     .              out_name(1:slen), axdir(idim)
	         CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
              ENDIF
           ENDIF

 110	CONTINUE

        WRITE ( risc_buff, 2050 )
        CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
        WRITE ( risc_buff, 2060 )
        CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

        RETURN

	END

