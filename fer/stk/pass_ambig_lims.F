	SUBROUTINE PASS_AMBIG_LIMS( cx, mr, status )

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
* for axes in which limits are fully contained in the definition the
* requesting context limits are not necessarily complete or correct
* when the variable is computed (via IS_ALGEBRA) the limits are
* corrected and stored with the memory resident variable.
* This routine functions to pass those corrected limits back to the
* requesting context when that memory resident data is re-accessed.

* In one case the limits are not restrictive (e.g. DEFINE VAR NEW=U[K=1:5]
* which may be used via LOAD NEW[K=2:3] or LOAD NEW )
* and the requesting context limits should be used if they are given explicitly
* otherwise the full limits of the variable will be passed back.
* This is the case of uvlim_gvn_gnl (a non-restrictive range given in def'n)

* In a second case limits are restrictive (e.g. DEFINE VAR NEW=U[K=1:3@AVE])
* and the requesting context must be an exact hit if given explicitly.
* This is the case of uvlim_gvn_xact (an exact range given in the def'n)

* (uvlim_need_xact): Also, if a compressing transformation is specified
* within user-defined variables but limits are not given.  For example,
*	yes? DEFINE VAR KAVE = U[K=@AVE]
*	yes? LIST/K=1:3 KAVE
* The computed and stored context will have the correct Z world coord limits
* but will have K=-9999:-9999 (since it has been reduced to a point)
* The context used in searching will have K=1:3, incorrectly.
* Thus, unpecified_int4 subscript values must be passed back.

* Also, since data may be grabbed with limits completely unspecified
* (determined automatically from the file limits) the context limits may
* be unspecified although the mr variable limits are known.
* Pass back the mr limits.

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  6/22/89
*	10/25/89 - allow unspecified context limits
*	 5/16/90 - allow file data to come back with smaller limits than
*		   requested --> context limits .GT. memory var. size
* Unix/RISC port - 8/12/91 *sh*
*     bug fix that needs to be applied to VMS version, too
*     see "8/12/91" below
*     8/28/91 - fix the 5/90 "kludge" so it cant result in lo>hi
* V230:  6/10/92 - bug fix for err220_unspec_limits.jnl
*                  CANCEL REGION;SET REG/K=1/L=1;STAT TEMP-TEMP[I=@AVE]
*                  results in unspecified world coord limits on I axis
*        11/4/92 - expanded variable name length
* V420:  11/95 - error conditions loosened to match the documentation
* V450:   1/97   - fix to never-observed bug: the check
*		        "uvar_given(idim,var) .LE. uvlim_gvn_xact"
*		   ignores the possibility of uvlim_irrelevant



#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/errmsg.parm'
	include	'ferret_cmn/xcontext.cmn'
	include	'ferret_cmn/xvariables.cmn'
	include	'ferret_cmn/xtext_info.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XTEXT_INFO.CMN'
#endif

* calling argument declarations
	INTEGER	cx, mr, status

* local variable declarations
	INTEGER		idim, uvar
	CHARACTER	VAR_CODE*24

* initialize
	uvar = mr_variable( mr )

* data set:
* uvar data set is not necessarily the same as the searching context
	IF ( .NOT.uvar_need_dset(uvar) ) cx_data_set(cx)=mr_data_set(mr)

* take bad data flag from the memory resident result
	cx_bad_data( cx ) = mr_bad_data( mr )
	
* axis limits and subscripts:
	DO 100 idim = 1,4

* error checks:
	   IF ( cx_given(idim,cx) ) THEN
	      IF ( uvar_given(idim,uvar) .EQ. uvlim_gvn_gnl ) THEN
* ... suppose DEFINE VARIABLE NEW=TEMP[K=1:3]
* ... then    LIST NEW[K=5] doesn't make sense
	         IF ( mr_lo_ww(idim,mr).GT.cx_hi_ww(idim,cx)
     .           .OR. mr_hi_ww(idim,mr).LT.cx_lo_ww(idim,cx) ) GOTO 5100
!pre 11/95            IF ( mr_lo_ww(idim,mr).GT.cx_lo_ww(idim,cx)
!pre 11/95 .           .OR. mr_hi_ww(idim,mr).LT.cx_hi_ww(idim,cx) ) GOTO 5100
	      ELSEIF ( uvar_given(idim,uvar) .EQ. uvlim_gvn_xact ) THEN
* ... suppose DEFINE VARIABLE NEW=TEMP[K=1:3@AVE]
* ... then    LIST NEW[K=1:2] doesn't make sense
* 11/95 - not an error - just ignore the K=1:2 limits in the example
	         IF ( mr_lo_ww(idim,mr).NE.cx_lo_ww(idim,cx)
     .           .OR. mr_hi_ww(idim,mr).NE.cx_hi_ww(idim,cx) ) THEN
	            cx_lo_ss(cx,idim) = mr_lo_ss(mr,idim)
	            cx_hi_ss(cx,idim) = mr_hi_ss(mr,idim)
	            cx_lo_ww(idim,cx) = mr_lo_ww(idim,mr)
	            cx_hi_ww(idim,cx) = mr_hi_ww(idim,mr)
	         ENDIF
!pre 11/95      IF ( mr_lo_ww(idim,mr).NE.cx_lo_ww(idim,cx)
!pre 11/95.    .OR. mr_hi_ww(idim,mr).NE.cx_hi_ww(idim,cx) ) GOTO 5100
	      ENDIF
	   ENDIF

* pass limits back
	   IF (   uvar_given(idim,uvar) .EQ. uvlim_gvn_xact
     .	   .OR. ( uvar_given(idim,uvar) .EQ. uvlim_gvn_gnl
     .		  .AND. .NOT.cx_given(idim,cx) )	    ) THEN

	         cx_lo_ss(cx,idim) = mr_lo_ss(mr,idim)
	         cx_hi_ss(cx,idim) = mr_hi_ss(mr,idim)
	         cx_lo_ww(idim,cx) = mr_lo_ww(idim,mr)
	         cx_hi_ww(idim,cx) = mr_hi_ww(idim,mr)

	   ELSEIF ( uvar_given(idim,uvar) .EQ. uvlim_need_xact ) THEN
	      cx_lo_ss(cx,idim) = mr_lo_ss(mr,idim)
	      cx_hi_ss(cx,idim) = mr_hi_ss(mr,idim)
! 6/92 bug fix: (possibly the IF test isn't needed - it's here to minimize
!                impacts of this fix)
              IF ( cx_lo_ww(idim,cx) .EQ. unspecified_val8
     .       .AND. mr_lo_ww(idim,mr) .NE. unspecified_val8 ) THEN
	         cx_lo_ww(idim,cx) = mr_lo_ww(idim,mr)
	         cx_hi_ww(idim,cx) = mr_hi_ww(idim,mr)
              ENDIF

* all abstract variables live on grid ABSTRACT, however,
* an expression like X+Y is normal to Z and T so they are flagged as irrelevant
	   ELSEIF ( uvar_given(idim,uvar) .EQ. uvlim_irrelevant ) THEN
	      cx_lo_ss(cx,idim) = unspecified_int4
	      cx_hi_ss(cx,idim) = unspecified_int4
	      cx_lo_ww(idim,cx) = unspecified_val8
	      cx_hi_ww(idim,cx) = unspecified_val8
* 9/97 *kob* cx_calendar must reflect these changes as well	  
	      IF (idim .eq. t_dim) 
     .                cx_calendar(cx) = .FALSE.

* context limits unspecified but untransformed data limits available anyway
	   ELSEIF ( cx_lo_ss(cx,idim) .EQ. unspecified_int4 ) THEN
	      cx_lo_ss(cx,idim) = mr_lo_ss(mr,idim)
	      cx_hi_ss(cx,idim) = mr_hi_ss(mr,idim)
	      cx_lo_ww(idim,cx) = mr_lo_ww(idim,mr)
	      cx_hi_ww(idim,cx) = mr_hi_ww(idim,mr)

	   ENDIF

* 8/12/91 bug fix during Unix port
* this is not known to be the perfect fix (kludge)...
* User-defined variables that have explicit limits given in their definitions
* need to pass that information to the context that is requesting them
* in, for example:  LET MASK=TEMP[L=1]
*                   STAT/L=2 MASK*TEMP
* the expression needs to be evaluated as if it were MASK[L=1]*TEMP
! 1/97 - check against uvlim_irrelevant just to be thorough
           IF ( uvar_given(idim,uvar) .LE. uvlim_gvn_xact
     .    .OR.  uvar_given(idim,uvar) .EQ. uvlim_irrelevant  )
     .                               cx_given(idim,cx) = .TRUE.

* a bit of a kludge ... for some expressions with @AVE, etc. the cx subscript
* limits don't get set to unspecified.  Too lazy to find the "right" sol'n.
	   IF ( mr_lo_ss(mr,idim) .EQ. unspecified_int4 ) THEN
	      cx_lo_ss(cx,idim) = unspecified_int4
	      cx_hi_ss(cx,idim) = unspecified_int4
	   ENDIF

* another kludge (5/90) file data may come back smaller than
* requested --> calculated results from it will be smaller, too
	   IF ( cx_lo_ss(cx,idim) .LT. mr_lo_ss(mr,idim) ) THEN
	      cx_lo_ss(cx,idim) = mr_lo_ss(mr,idim)
	      cx_lo_ww(idim,cx) = mr_lo_ww(idim,mr)
*     8/28/91 - don't allow limits to be reversed: kludge patch
              IF ( cx_lo_ss(cx,idim) .GT.cx_hi_ss(cx,idim) ) THEN
                 cx_hi_ss(cx,idim) = mr_hi_ss(mr,idim)
                 cx_hi_ww(idim,cx) = mr_hi_ww(idim,mr)
              ENDIF
	   ENDIF
	   IF ( cx_hi_ss(cx,idim) .GT. mr_hi_ss(mr,idim) ) THEN
	      cx_hi_ss(cx,idim) = mr_hi_ss(mr,idim)
	      cx_hi_ww(idim,cx) = mr_hi_ww(idim,mr)
*     8/28/91 - don't allow limits to be reversed: kludge patch
              IF ( cx_lo_ss(cx,idim) .GT.cx_hi_ss(cx,idim) ) THEN
                 cx_lo_ss(cx,idim) = mr_lo_ss(mr,idim)
                 cx_lo_ww(idim,cx) = mr_lo_ww(idim,mr)
              ENDIF
	   ENDIF

 100	CONTINUE

* success
	status = ferr_ok
	RETURN

* error
 5000	RETURN
 5100	CALL ERRMSG( ferr_limits, status,
     .		ww_dim_name(idim)//
     .		' axis limits don''t match variable def''n: '//
     .		VAR_CODE( cat_user_var, uvar )//pCR//
     .	'Are you giving explicit limits when SET REGION would suffice ?',
     .	*5000 )
	END
