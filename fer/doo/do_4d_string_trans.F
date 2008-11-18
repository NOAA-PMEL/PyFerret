	INTEGER FUNCTION DO_4D_STRING_TRANS (	trans,
     .					com, com_mr, com_cx,
     .					res, res_mr, res_cx, arg)

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
* call DO_4D_STRING_GOODBAD with the appropriate transformation

* programmer - Ansley Manke
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* From DO_4D_NBD

* calling argument declarations:
	INTEGER	trans, com_mr, com_cx, res_mr, res_cx
	REAL*8  com
	REAL	res, arg

* internal variable declarations:
        LOGICAL at_ok
	INTEGER	CX_DIM_LEN

	include	'ferret.parm'
	include	'errmsg.parm'
	include	'interp_stack.parm'
	include	'xprog_state.cmn'


* diagnostic mode output: " doing --> VAR_NAME[x=lo:hi@TRN:n,D=#]"
	IF ( mode_diagnostic ) 	CALL DIAG_OP
     .				( 'doing',isact_class_trans, res_cx, idim )

* determine the type of transform requested
	
	at_ok = (trans .EQ. trans_4d_bad_pt) .OR. 
     .          (trans .EQ. trans_4d_good_pt)
	IF (.NOT. at_ok) GOTO 5010

* pass the transform to be performed

	CALL DO_4D_STRING_GOODBAD(   trans,
     .			    com, com_mr, com_cx,
     .			    res, res_mr, res_cx, arg )

	DO_4D_STRING_TRANS = ferr_ok

 5000	RETURN

 5010	DO_4D_STRING_TRANS = ferr_data_type
        CALL ERRMSG( ferr_data_type, status, 
     .  'Only NGD or NBD transforms allowed for string data', *5000 )
	END	
