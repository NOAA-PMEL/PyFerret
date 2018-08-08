	SUBROUTINE TM_WARN_SUBSPAN (iaxis)

*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*.
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
* 5/2016 Ansley Manke
* See ticket 1432.
* Write a NOTE if an axis definition is a subspan modulo axis
* but its length is within a grid cell of the full modulo length.

        include 'tmap_dims.parm'
#include "tmap_dset.parm"
#include "gt_lib.parm"
        include 'xio.cmn_text'
        external xio_data
        include 'xtm_grid.cmn_text'
        external xgt_grid_data

* argument declarations
	INTEGER iaxis


* local variable declarations:
	LOGICAL	TM_ITS_SUBSPAN_MODULO
	INTEGER TM_LENSTR1, alen, blen, vlen
	REAL	TM_WW_AXLEN, GET_LINE_COORD, axwwlen, delta
	CHARACTER*32 TM_FMT, buff1, buff2

	INTEGER tt  ! nice short name
	EQUIVALENCE (tt,lunit_errors)

	vlen = TM_LENSTR1(line_name(iaxis))
        axwwlen = TM_WW_AXLEN(iaxis)

* get the nominal delta grid cell length for comparison

	IF (line_delta(iaxis) .NE. unspecified_val8) THEN
	   delta = line_delta(iaxis)
	ELSE
	   delta = GET_LINE_COORD(linemem(iaxis)%ptr, 2) - GET_LINE_COORD(linemem(iaxis)%ptr, 1)
	ENDIF

        IF (line_modulo(iaxis) .AND. TM_ITS_SUBSPAN_MODULO(iaxis) .AND.
     .	    (line_modulo_len(iaxis)-axwwlen) .LE. delta )  THEN
           buff1 = TM_FMT( SNGL(axwwlen),7,32,alen )
           buff2 = TM_FMT( SNGL(line_modulo_len(iaxis)),7,32,blen )
           CALL TM_NOTE('Subspan modulo axis '//line_name(iaxis)(:vlen)//
     .		'. Axis span = '//buff1(:alen)//
     .		', modulo length = ' //buff2(:blen)//
     .		'. Is this intended to be a full-span axis?', tt)
        ENDIF
	
	RETURN
	END
