	SUBROUTINE POTENT_AX_LIMS ( cx, axis, lo_ss, hi_ss )

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
* determine the "potential" limits of axis in context cx - that is,
* the subscript limits that would exist if we ignored compressing transforms  
* note: uses cx_buff

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* V702 *sh* 11/16 - using technique employed in POTENT_SIZE
 
        include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'xcontext.cmn'

* calling argument declarations:
	INTEGER cx, axis, lo_ss, hi_ss

* internal variable declarations:
	INTEGER	status


	IF (cx_trans(axis, cx) .GT. trans_compress_code) THEN
* make a copy of the context axis in a scratch buffer
	   cx_variable( cx_buff ) = cx_variable( cx )
	   cx_category( cx_buff ) = cx_category( cx )
	   cx_grid    ( cx_buff ) = cx_grid    ( cx )
	   CALL TRANSFER_AXIS( axis, cx, cx_buff )
	   cx_trans(axis,cx_buff) = trans_no_transform
	   cx_by_ss(axis,cx_buff) = .FALSE.
	   CALL FLESH_OUT_AXIS( axis, cx_buff, status )	! ignore status
	   lo_ss = cx_lo_ss(cx_buff,axis)
	   hi_ss = cx_hi_ss(cx_buff,axis)
	ELSE
	   lo_ss = cx_lo_ss(cx,axis)
	   hi_ss = cx_hi_ss(cx,axis)
	ENDIF

	RETURN
	END
