	SUBROUTINE DFLT_TRANS_ARG( idim, cx, status )

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
* if no argument was specified for the given transformation then insert
* a default

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:   6/7/89 - removed this function from GET_CONTEXT_MODS
*	 1/11/90 - added @LOC
*	  2/7/90 - corrected error detect bug for transforms without args
* V400:  5/30/95 - increased arbitrary argument limit of 1000 to 32766

* calling arguments
	INTEGER	idim, cx, status

* internal variable declarations
	INTEGER	i, ktrans
	REAL	arg

#ifdef unix
	include 'ferret_cmn/ferret.parm'	
	include 'ferret_cmn/errmsg.parm'	
	include 'ferret_cmn/xcontext.cmn'
	include 'ferret_cmn/xalgebra.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'	
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'	
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE 'FERRET_CMN:XALGEBRA.CMN'
#endif

* initialize
	ktrans = cx_trans( idim, cx )
	arg   = cx_trans_arg( idim, cx )

* identify the transformation in terms of numbering in COMMON (XALG*)
	DO 100 i = 1, num_transforms
	   IF ( ktrans .EQ. alg_trans_num(i) ) GOTO 200
 100	CONTINUE

* not found !
	status = ferr_internal
	GOTO 5100

* does it take an argument ?
 200	IF ( alg_trans_has_arg( i ) ) THEN
* yes !
	   IF ( arg .EQ. bad_val4 ) THEN
* ... insert default value
	      cx_trans_arg( idim, cx ) = alg_trans_dflt_arg(i)

	   ELSEIF ( alg_trans_its_int(i) ) THEN
* ... convert argument to integer if possible
	      IF ( ABS(arg) .GT. 32766. ) THEN	! was 1000 5/95
	         GOTO 5000
	      ELSE
	         cx_trans_arg( idim, cx ) = INT( arg )
	      ENDIF
	   ELSE
* ... leave it as it is
	      CONTINUE
	   ENDIF

	ELSE
* no! if the user gave an argument he/she shouldn't have
	   IF ( arg .NE. bad_val4 ) THEN
	      status = ferr_unknown_arg
	      GOTO 5100
	   ENDIF
	ENDIF

* argument of zero is illegal for some things
	IF( arg .EQ. 0.0
     .	 .AND. (   ktrans.EQ.trans_smth_box
     .	      .OR. ktrans.EQ.trans_smth_binml
     .	      .OR. ktrans.EQ.trans_fill_ave )  ) GOTO 5000

* successful completion
 1000	status = ferr_ok
	RETURN

* error exit
 5000	status = ferr_out_of_range
 5100	RETURN
	END
