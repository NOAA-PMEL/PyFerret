      SUBROUTINE SHOW_DIM_XML( lun, dname, npts)

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
* display basic facts about a dimension that is not a coordinate variable

* programmer - Ansley Manke, based on line_facts_xml.F
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* 27-Jan-2011
* 
	include 'tmap_dims.parm'
#	include "tmap_dset.parm"
	include	'ferret.parm'
	include 'xprog_state.cmn'
	include 'xtext_info.cmn'
	include 'xrisc.cmn'
	include 'xdset_info.cmn_text'

* calling argument declarations
	INTEGER 	lun, npts
	CHARACTER*(*)   dname

* local variable declarations:
	INTEGER		slen, status
	CHARACTER	TM_FMT*16, show_str*16, outstring*128

 2010	FORMAT ( '<axis name="', A, '">' )
 2020	FORMAT ( '<attribute name="length" type="short">' )
 2030   FORMAT ('   <value>', A, '</value>')
 2040   FORMAT ('</attribute>')
 2050	FORMAT ( '</axis>' )

	CALL ESCAPE_FOR_XML (dname, outstring, slen)
	WRITE ( risc_buff, 2010 ) outstring(1:slen)
	CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

* dimension length

        show_str = TM_FMT(FLOAT(npts), 16,16,slen)
        CALL ESCAPE_FOR_XML (show_str, outstring, slen)
        WRITE ( risc_buff, 2020 ) outstring(1:slen)
        CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

* close tag

        WRITE ( risc_buff, 2050 ) 
	CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

	RETURN
	END



