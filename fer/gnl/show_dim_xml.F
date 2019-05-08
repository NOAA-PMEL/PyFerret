	SUBROUTINE SHOW_DIM_XML( dim_name, dim_len, lun )

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
* display basic facts about the given dimension (which is not a coordiante axis) 
* in xml-style format 

* programmer - Ansley Manke, based on show_line_xml.F
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* Dec-2013

        IMPLICIT NONE
	include 'tmap_dims.parm'
#	include "tmap_dset.parm"
	include	'ferret.parm'
	include 'xrisc.cmn'

* calling argument declarations
	CHARACTER*(*)   dim_name
	INTEGER 	dim_len, lun

* local variable declarations:
	INTEGER		slen
	CHARACTER	TM_FMT*64, outbuff*20
        CHARACTER       attname*128, attbuff*2048, 
     .                  outstring*2048

 2010	FORMAT ( '<dimension name="', A, '">' )
 2030	FORMAT ('<attribute name="', A, '" type="short">' )
 2050   FORMAT ('   <value>', A, '</value>')
 2060   FORMAT ('</attribute>')
 2080	FORMAT ( '</dimension>' )

	CALL ESCAPE_FOR_XML (dim_name, outstring, slen)
	WRITE ( risc_buff, 2010 ) outstring(1:slen)
	CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
	
* Length of dimension

	attname = 'length'
	CALL ESCAPE_FOR_XML(attname, outstring, slen)
	WRITE ( risc_buff, 2030 ) outstring(:slen)
	CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

	attbuff = TM_FMT(FLOAT(dim_len), 7, 14, slen)
	CALL ESCAPE_FOR_XML (attbuff, outstring, slen)
	WRITE ( risc_buff, 2050 ) outstring(1:slen)
	CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

	WRITE ( risc_buff, 2060 )
	CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

* closing tag

	WRITE ( risc_buff, 2080 ) 
	CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)

	RETURN
	END
