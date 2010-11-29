	SUBROUTINE RESTRICT_FILE_LIMITS ( cx, modified, status )

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
* restrict the context limits of a file variable to the range that is
* indicated by ds_grid_start/end in COMMON

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V500: 3/99 - *sh*
*       5/99 *sh* - when strides are used on a netCDF variable, we may be
*		looking at an implicit grid. In this case VAR_SS_LIMS returns
*		unspecified_int4.
*		NOTE: possible bug introduced with strides since the region
*			checks here are bypassed altogether
* V530: 11/00 *sh* - return "modified" flag
* V552 *acm* 4/03- up VAR_CODE decl from 64 to 128 chars

	include	'ferret.parm'
	include	'errmsg.parm'
	include	'xcontext.cmn'
	include	'xtext_info.cmn'

* calling argument declarations:
	LOGICAL	modified
	INTEGER	cx, status

* internal variable declarations:
	INTEGER 	TM_LENSTR1, 
     .			idim, variable, category,
     .			slen, slen2, hilim, lolim
	CHARACTER*128	VAR_CODE, variable_code
	CHARACTER	LEFINT*5, CX_DIM_STR*48, buff1*48, buff2*5

* local parameter declarations
	LOGICAL     full_precision
	PARAMETER ( full_precision = .TRUE. )

* --- end of introductory code ---

* initialize
	modified = .FALSE.
	category = cx_category( cx )
	variable = cx_variable( cx )

	DO 10 idim = 1, 4
	   IF ( cx_lo_ss(cx,idim) .EQ. unspecified_int4 ) GOTO 10
	   CALL VAR_SS_LIMS( idim, cx, lolim, hilim )
	   IF ( lolim .EQ. unspecified_int4 ) GOTO 10
	   IF ( cx_lo_ss(cx,idim) .GT. hilim .OR.
     .          cx_hi_ss(cx,idim) .LT. lolim )
     .		GOTO 5100
	   IF ( (cx_lo_ss(cx,idim) .LT. lolim)
     .     .OR. (cx_hi_ss(cx,idim) .GT. hilim) ) THEN
	      IF (cx_lo_ss(cx,idim).LT.lolim) cx_lo_ss(cx,idim) = lolim
	      IF (cx_hi_ss(cx,idim).GT.hilim) cx_hi_ss(cx,idim) = hilim
	      cx_by_ss(idim, cx) = .TRUE.
	      CALL FLESH_OUT_AXIS( idim, cx, status )
	      IF ( status.NE.ferr_ok ) GOTO 5900
	      modified = .TRUE.
	   ENDIF
 10	CONTINUE

* successful completion
	status = ferr_ok
	RETURN

* error exit
 5100	buff1 = CX_DIM_STR(idim, cx, ':',full_precision,slen)
	WRITE(buff2,'(I5)') lolim
	variable_code = VAR_CODE( category, variable )
	CALL ERRMSG( ferr_limits, status,
     .		variable_code(:TM_LENSTR1(variable_code))//
     .		' does not contain '//buff1(:slen)//
     .          pCR//'Data are available in '//ss_dim_name(idim)//
     .		' ='//buff2//':'//LEFINT(hilim, slen2)
     .		,*5900 )   ! 3/99
 5900	RETURN
	END	
