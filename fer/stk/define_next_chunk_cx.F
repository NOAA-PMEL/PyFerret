	SUBROUTINE DEFINE_NEXT_CHUNK_CX(idim, com_cx, dflt_cx, frag, status)

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
* increment com_cx so that it becomes the context describing the next chunk
* needed in a gather operation

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* V702: *sh* 11/16 - code pulled from IS_GATHER

	include	'tmap_dims.parm'
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'xcontext.cmn'

* calling argument declarations:
	INTEGER idim, com_cx, dflt_cx, frag, status

* calling argument definitions:
*	idim - the axis of the gather operation
*	com_cx - the context to be updated (the next component of the gather)
*	dflt_cx - the target context, with transforms, pre-split/gather
*	frag - the fragment (a.k.a increment) size

* internal variable declarations:
	INTEGER top_ss, lo_ss

* increment the subscript range by fragment size
	top_ss = cx_hi_ss(com_cx,idim)
	CALL TRANSFER_CONTEXT( dflt_cx, com_cx )
	cx_lo_ss(com_cx,idim) = top_ss + 1
	cx_hi_ss(com_cx,idim) = MIN( top_ss+frag, cx_hi_ss(dflt_cx,idim))
	cx_by_ss(idim,com_cx) = .TRUE.
	CALL FLESH_OUT_AXIS( idim, com_cx, status )       ! ignore status

* same exit whether success or failure
	RETURN
	END	

