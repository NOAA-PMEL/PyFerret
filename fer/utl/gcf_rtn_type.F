	INTEGER FUNCTION GCF_RTN_TYPE( gcfcn, arg1_cx )

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

* return the data type of the result of this function call

* V533: *sh* 6/01

	include 'ferret.parm'
	include 'grid_chg_fcns.parm'   ! for place-holder behavior
        include 'EF_Util.parm'
	include 'xcontext.cmn'
	include 'xgrid_chg_fcns.cmn'

* Upon entry to this routine the context stack should be organized as
*         default context (var=uvar, region=(1:1,1:1,1:1,1:1) for base var)
* big_cx: the merged grid context
*         argument N context
*         argument N-1 context
*         ...
*   -->   argument 1 context

* calling argument declarations
	INTEGER		gcfcn, arg1_cx

* internal variable declarations
!	INTEGER	GCF_ARG_EVAL_ORDER
	INTEGER	EFCN_GET_NUM_REQD_ARGS,
     .          iarg, cx, i
        INTEGER EF_cx_list(EF_MAX_ARGS), narg
!	INTEGER efcn_get_rtn_type


	IF ( gcfcn .LE. gfcn_num_internal ) THEN

* internal grid-changing function
	   GCF_RTN_TYPE = gfcn_rtn_type(gcfcn)

* ... inherit the return type from the indicated argument
	   IF (GCF_RTN_TYPE .LT. 0) THEN
	     iarg = ABS(GCF_RTN_TYPE)
!	     cx = arg1_cx-(GCF_ARG_EVAL_ORDER(gcfcn,iarg)-1)
	     cx = arg1_cx-(iarg-1)
	     IF (cx_type(cx) .EQ. ptype_string) THEN
	       GCF_RTN_TYPE = pfcn_rtn_string
	     ELSE
	       GCF_RTN_TYPE = pfcn_rtn_float
	     ENDIF
	   ENDIF

	ELSE

* external function
*   ... create a context list array
           narg = EFCN_GET_NUM_REQD_ARGS(gcfcn)
           DO 300 i = 1, narg
              EF_cx_list(i) = cx_stack_ptr - i + 1
 300       CONTINUE

	   GCF_RTN_TYPE = pfcn_rtn_float  ! TO BE REPLACED IN FUTURE !!
!           GCF_RTN_TYPE = EFCN_GET_RTN_TYPE(gcfcn, EF_cx_list, status)
!           IF (status .NE. ferr_ok) CALL ERRMSG
!     .                          ( status, status, ' ', *5000) 

	ENDIF

	RETURN

	END
