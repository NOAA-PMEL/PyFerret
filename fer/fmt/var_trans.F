	CHARACTER*(*) FUNCTION VAR_TRANS( idim, cx, out_len )

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
* create a string of the form NAME[Z
* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  7/21/89 - extracted from DIAG_OP
* V230:  11/4/92 - expanded user variable name length
* V500 *kob* 3/99- up VAR_CODE and local variable var to 64 chars
* V552 *acm* 4/03- up VAR_CODE decl from 64 to 128 chars

* calling argument declarations:
	INTEGER	idim, cx, out_len

* internal variable declarations:
	INTEGER	  TM_LENSTR1, slen, vlen
	CHARACTER VAR_CODE*128, CX_DIM_STR*48,
     .            var*128, out_buff*80, dim_str*48

#ifdef unix
	include	'ferret.parm'
	include	'xcontext.cmn'
	include	'xtext_info.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XTEXT_INFO.CMN'
#endif

* internal parameter declarations:
	LOGICAL     full_precision
	PARAMETER ( full_precision = .TRUE. )

* initialize
	var = VAR_CODE( cx_category(cx), cx_variable(cx) )
        vlen = TM_LENSTR1( var )

* TRANSFORMATION
* "VAR_NAME[x=lo:hi@TRN:n,D=#]"
	dim_str = CX_DIM_STR( idim, cx, ':',  full_precision, slen )
	WRITE (out_buff,3000) var(:vlen), dim_str(:slen), cx_data_set(cx)
	CALL TM_PACK_STRING( out_buff, out_buff, 1, 80, out_len )

 3000	FORMAT (A,'[',A,',D=',I3,']')

	VAR_TRANS = out_buff

	RETURN
	END
