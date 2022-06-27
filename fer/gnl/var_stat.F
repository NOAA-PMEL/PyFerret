	SUBROUTINE VAR_STAT( dat, mv, cx, lun, full, stat_digits, status )

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
* report basic statistics about the data passed

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* revision 0.0 - 3/25/88
* V200:	10/11/89 - modified array declarations using XMEM_SUBSC.CMN
*	12/19/89 - eliminated the possibility of multiple data set variables
*		 - corrected serious error in standard deviation calculation
* Unix/RISC port - 3/12/91 - "formatted internal IO" bugs with
*                            FUNCTIONS in WRITE list
* V230: 7/29/92 - fixed precision problems when the variance is very small
*                 compared to the mean - sumsq_dev could be negative.
*                 Switched to a 2-pass computation
*       8/28/92 - document shape of regions, too  (e.g. 160*100*27*12)
*               - avoid overflows with "**too big**"
* V300: 10/20/93 - changed **0.5 to SQRT to work-around Macintosh Lang. Sys.
*		   FORTRAN bug
* V320: 10/26/94 - changed routine name "STAT" to VAR_STAT for IBM/AIX
*			compatibility
*       12/29/94 - use SPLIT_LIST to redirect tty output for GUI
* V420	 2/7/96  - added STAT/BRIEF
* 12/96 *kob* - Linux Port - had to move variable assignments off of
*               declaration line.
* V530: 9/00 *sh* - increase rbuff size to allow longer "(isiz*jsiz*ksiz*lsiz)"
*		  - added support for string variables
* V532: 5/01 *sh* - minor bug fix: "small" and "large" init'ed to wrong dtype
* V606: 7/07 *acm*- define symbols with the latest results of a call to STAT
*                   STAT_MIN, STAT_MAX, STAT_MEAN, STAT_STD
* V608:10/07 *acm*- If all missing then set these symbols to 'bad'. Also set
*                   symbols STAT_NGOOD, STAT_NBAD
* V664:  8/10 - implement robust method for computing variances
*       *acm* 3/12 Add E and F dimensions (use nferdims in tmap_dims.parm)
*                  Rename the file to match the function name var_stat.F
* V6.8  acm  6/12 Implement mode_6d_lab, set by default. If canceled, the 
*                 various SHOW commnands will not list info for E,F dimensions.
* V6.68  acm 3/14 ticket 2148: fix truncated ntot or nbad in stat listing
* v6.931 acm 11/14 Delete ALL the stat* symbols that may be left over from 
*                  a previous evaluation. STAT/BRIEF does not reset them all.
* V720 *acm*  3/17 Ticket 2512: Add STAT/PRECICISION=
* v767 6/22 *acm* See issue 112: The total size could need INTEGER*8

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include 'errmsg.parm'
	include 'ferret.parm'
	include 'xvariables.cmn'
	include	'xmem_subsc.cmn'
	include 'xcontext.cmn'
        include 'xrisc.cmn'           ! for Unix/RISC bugs (&12/94)
	include 'xprog_state.cmn'

* special equivalence needed only with risc compiler: bug fix
        CHARACTER rbuff(4)*24
        EQUIVALENCE ( rbuff, risc_buff )

* calling argument declarations:
	LOGICAL	full
	INTEGER	mv, cx, lun, stat_digits, status
	REAL    dat( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit,m1loe:m1hie,m1lof:m1hif )

* internal variable declarations:
	INTEGER*8 CGRID_SIZE8
	INTEGER	TM_LENSTR1, GET_MAX_STRING_LEN,
     .		i, j, k, l, m, n, nbad, ngood, ntot, idim,
     .		len_line, len_set_name, grid, slen, i1, i2, nok, 
     .		listdims, nfig, max_dig
	REAL	datum, bad, std_dev, large, small, mean, sumsq_dev, dev
	REAL*8	sum, sum2, xmean, zmean, x, xdelta, variance_c, diff

	CHARACTER*255	VAR_TITLE, VAR_TITLE_MOD, vname_buff
	CHARACTER	TM_FMT*48, LEFINT*20, LEFINT8*20, AX_TITLE*32, CX_SIZE_STR*24,
     .			line_buff*130, title*32, bl13*13, symbuff*30
	PARAMETER (bl13 = '             ' )

* local parameter declarations:
	INTEGER		sig_fig
	PARAMETER     ( sig_fig = 5 )

* initialize
	grid  = mr_grid( mv )
	bad   = mr_bad_data( mv )
	nbad  = 0
	ngood = 0
	small = arbitrary_large_val4
	large = arbitrary_small_val4
	sum   = 0.0D0
        sumsq_dev = 0.0

	max_dig = 0
	IF (stat_digits .NE. 0) THEN
	   max_dig = stat_digits
	ELSE
	   stat_digits = sig_fig
	ENDIF

	listdims = nferdims
	IF ( .NOT. mode_6d_lab) listdims = 4

	IF ( full ) THEN
* give background on the data
* ... first the variable name
	   vname_buff   = VAR_TITLE( cx )
	   CALL SPLIT_LIST(pttmode_explct, lun, ' ', 1)
	   CALL SPLIT_LIST(pttmode_explct, lun,
     .		bl13//vname_buff( 1:TM_LENSTR1(vname_buff)), 0)
 3010	   FORMAT (13X,A)

* ... qualify title with modification line ?
	   vname_buff = VAR_TITLE_MOD( cx )
	   IF ( vname_buff .NE. ' ' ) THEN
	      len_line   = TM_LENSTR1( vname_buff )
	      CALL SPLIT_LIST(pttmode_explct, lun, 
     .			bl13//vname_buff(1:len_line), len_line+13)
	   ENDIF

* ... then location information normal to grid
	   DO 60 idim = 1, listdims
	      CALL ASCII_LIMITS( cx, idim, line_buff, len_line )
	      title = AX_TITLE( idim, grid, slen )
	      CALL SPLIT_LIST(pttmode_explct, lun, 
     .		bl13//title( :slen )//': '//line_buff(:len_line), 0)
 60	   CONTINUE

* ... then data set name or names
	   CALL GET_DSET_NAME( cx_data_set(cx), line_buff, len_set_name )
	   CALL SPLIT_LIST(pttmode_explct, lun,
     .		bl13//'DATA SET: '//line_buff(:len_set_name), 0)

	ENDIF

* string variables
	IF ( mr_type(mv) .EQ. ptype_string ) THEN
	   rbuff(1) = LEFINT(GET_MAX_STRING_LEN(cx,mv,dat), slen) 
	   rbuff(2) = LEFINT8(CGRID_SIZE8(cx), slen)
	   rbuff(3) = CX_SIZE_STR(cx,listdims,len_line)
	   CALL SPLIT_LIST(pttmode_explct, lun,
     .			' Total # of strings: '//
     .			 rbuff(2)(:slen)//' ('//rbuff(3)(:len_line)//')',0)
	   CALL SPLIT_LIST(pttmode_explct, lun,
     .			' Maximum string length: '//rbuff(1),0)
	   RETURN
	ENDIF

* pass 1: compute mean 
        IF (mode_6d_lab) THEN
	   DO 200 n = cx_lo_s6(cx), cx_hi_s6(cx)
	   DO 200 m = cx_lo_s5(cx), cx_hi_s5(cx)
	   DO 200 l = cx_lo_s4(cx), cx_hi_s4(cx)
	   DO 200 k = cx_lo_s3(cx), cx_hi_s3(cx)
	   DO 200 j = cx_lo_s2(cx), cx_hi_s2(cx)
	   DO 200 i = cx_lo_s1(cx), cx_hi_s1(cx)

	      datum = dat(i,j,k,l,m,n)

	      IF ( datum .EQ. bad ) THEN
	         nbad = nbad + 1

	      ELSE
	         ngood = ngood + 1
	         small = MIN( small, datum )
	         large = MAX( large, datum )
                 sum   = sum + datum
	      ENDIF
 200	   CONTINUE
	ELSE 
* If the data is 4D, and the user has canceled mode_6d_lab, just 
* compute and list the 4D result.

	   IF (cx_lo_s6(cx) .NE. unspecified_int4 .OR.
     .	       cx_lo_s5(cx) .NE. unspecified_int4) GOTO 5100
	   m = unspecified_int4
	   n = unspecified_int4
	   DO 210 m = cx_lo_s5(cx), cx_hi_s5(cx)
	   DO 210 l = cx_lo_s4(cx), cx_hi_s4(cx)
	   DO 210 k = cx_lo_s3(cx), cx_hi_s3(cx)
	   DO 210 j = cx_lo_s2(cx), cx_hi_s2(cx)
	   DO 210 i = cx_lo_s1(cx), cx_hi_s1(cx)

	      datum = dat(i,j,k,l,m,n)

	      IF ( datum .EQ. bad ) THEN
	         nbad = nbad + 1

	      ELSE
	         ngood = ngood + 1
	         small = MIN( small, datum )
	         large = MAX( large, datum )
                 sum   = sum + datum
	      ENDIF
 210	   CONTINUE
	ENDIF

        IF ( ngood .GT. 0 ) mean = sum / FLOAT(ngood)
        IF ( ABS(sum) .GE. 3.D35 ) sumsq_dev = 3.E35  ! flag for next pass

	IF ( full .AND. (ngood .GT. 1)) THEN
* pass 2: compute variance 

* Compute variance. See http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

* On-line algorithm with mean subtracted first. Noted as the most robust.
* Since we always compute the mean first anyway, use it.

	   nok = 0
	   xmean = 0.D0
	   sum2 = 0.D0
           zmean = mean

           IF (mode_6d_lab) THEN
	      DO 300 n = cx_lo_s6(cx), cx_hi_s6(cx)
	      DO 300 m = cx_lo_s5(cx), cx_hi_s5(cx)
	      DO 300 l = cx_lo_s4(cx), cx_hi_s4(cx)
	      DO 300 k = cx_lo_s3(cx), cx_hi_s3(cx)
	      DO 300 j = cx_lo_s2(cx), cx_hi_s2(cx)
	      DO 300 i = cx_lo_s1(cx), cx_hi_s1(cx)

	         x = dat(i,j,k,l,m,n)
	         IF ( x .NE. bad ) THEN
                    nok = nok + 1
		    x = x - zmean
                    xdelta = x - xmean
                    xmean = xmean + xdelta/FLOAT(nok)
                    sum2 = sum2 + xdelta*(x - xmean)  ! This expression uses the new value of mean
 	         ENDIF

 300	      CONTINUE
           ELSE  ! 4D result
	      DO 310 n = cx_lo_s6(cx), cx_hi_s6(cx)
	      DO 310 m = cx_lo_s5(cx), cx_hi_s5(cx)
	      DO 310 l = cx_lo_s4(cx), cx_hi_s4(cx)
	      DO 310 k = cx_lo_s3(cx), cx_hi_s3(cx)
	      DO 310 j = cx_lo_s2(cx), cx_hi_s2(cx)
	      DO 310 i = cx_lo_s1(cx), cx_hi_s1(cx)

	         x = dat(i,j,k,l,m,n)
	         IF ( x .NE. bad ) THEN
                    nok = nok + 1
		    x = x - zmean
                    xdelta = x - xmean
                    xmean = xmean + xdelta/FLOAT(nok)
                    sum2 = sum2 + xdelta*(x - xmean)  ! This expression uses the new value of mean
 	         ENDIF

 310	      CONTINUE 
           ENDIF

           variance_c = sum2/FLOAT(nok - 1)
	   std_dev = SQRT(variance_c)
 
	ENDIF

* compute and display results
	ntot = nbad + ngood

        rbuff(1) = LEFINT(nbad, slen)
        rbuff(2) = LEFINT(ntot, slen)
        rbuff(3) = CX_SIZE_STR(cx,listdims,len_line)
	CALL SPLIT_LIST(pttmode_explct, lun,' ',1)
	CALL SPLIT_LIST(pttmode_explct, lun,
     .			' Total # of data points: '//
     .			rbuff(2)(:slen)//' ('//rbuff(3)(:len_line)//')',0)
	CALL SPLIT_LIST(pttmode_explct, lun,
     .			' # flagged as bad  data: '//rbuff(1),0)
	IF ( ngood .GT. 0 ) THEN
	   nfig = 0
           rbuff(1) = TM_FMT(small, stat_digits, 16, slen)
           rbuff(2) = TM_FMT(large, stat_digits, 16, slen)
	   diff = (large-small)/MAX( ABS(large), ABS(small) )
	   IF (diff.LT.0.01 .AND. ngood.GT.1) THEN
	      IF (max_dig .EQ. 0) nfig = 4
              rbuff(1) = TM_FMT(small, stat_digits+nfig, 20, slen)
              rbuff(2) = TM_FMT(large, stat_digits+nfig, 20, slen)
	      diff = (large-small)/MAX( ABS(large), ABS(small) )
	      IF (diff .LT. 1.e-7) THEN
	       IF (max_dig .EQ. 0) nfig = 16-stat_digits
	        rbuff(1) = TM_FMT(small, stat_digits+nfig, 20, slen)
	        rbuff(2) = TM_FMT(large, stat_digits+nfig, 20, slen)
	      ENDIF
	   ENDIF

           IF ( ABS(sum) .GE. 3.D35 ) THEN
              rbuff(3) = '**too big**'
	      slen = 11
           ELSE
              rbuff(3) = TM_FMT(mean , stat_digits+nfig, 20, slen)
           ENDIF
	   CALL SPLIT_LIST(pttmode_explct, lun,
     .			' Minimum value: '//rbuff(1), 0)
	   CALL SPLIT_LIST(pttmode_explct, lun,
     .			' Maximum value: '//rbuff(2), 0)
	   CALL SPLIT_LIST(pttmode_explct, lun,
     .			' Mean    value: '//rbuff(3)(:slen)//
     .					' (unweighted average)', 0)
	ENDIF

! Define Ferret symbols with min, max, mean
        IF (ngood .EQ. 0) THEN 
           rbuff(1) = 'bad'
           rbuff(2) = 'bad'
           rbuff(3) = 'bad'
        ENDIF

! First delete all the STAT* symbols that may be set
! Leave none left over from a previous evaluation

	symbuff = 'STAT_MIN'
	CALL GETSYM( symbuff, title, i2, i1 )
	IF ( i1 .EQ. 0 ) CALL DELSYM(symbuff,i1)

	symbuff = 'STAT_MAX'
	CALL GETSYM( symbuff, title, i2, i1 )
	IF ( i1 .EQ. 0 ) CALL DELSYM(symbuff,i1)

	symbuff = 'STAT_MEAN'
	CALL GETSYM( symbuff, title, i2, i1 )
	IF ( i1 .EQ. 0 ) CALL DELSYM(symbuff,i1)

	symbuff = 'STAT_STD'
	CALL GETSYM( symbuff, title, i2, i1 )
	IF ( i1 .EQ. 0 ) CALL DELSYM(symbuff,i1)

	symbuff = 'STAT_NGOOD'
	CALL GETSYM( symbuff, title, i2, i1 )
	IF ( i1 .EQ. 0 ) CALL DELSYM(symbuff,i1)

	symbuff = 'STAT_NBAD'
	CALL GETSYM( symbuff, title, i2, i1 )
	IF ( i1 .EQ. 0 ) CALL DELSYM(symbuff,i1)


	IF ( mr_type(mv) .NE. ptype_string ) THEN
           symbuff = 'STAT_MIN'
           slen = TM_LENSTR1(rbuff(1))
      	   CALL PUTSYM( symbuff, rbuff(1), slen, i2)
           symbuff = 'STAT_MAX'
           slen = TM_LENSTR1(rbuff(2))
      	   CALL PUTSYM( symbuff, rbuff(2), slen, i2)
           symbuff = 'STAT_MEAN'
           slen = TM_LENSTR1(rbuff(3))
      	   CALL PUTSYM( symbuff, rbuff(3), slen, i2)
        ENDIF


	IF ( full .AND. ngood.GT.1 ) THEN
           IF ( sumsq_dev .GE. 3.E35 ) THEN
              rbuff(1) = ' **too big**' 
           ELSE
              rbuff(1) = TM_FMT(std_dev, stat_digits, 16, slen)
           ENDIF
	   CALL SPLIT_LIST(pttmode_explct, lun,
     .			' Standard deviation: '//rbuff(1), 0)

! Define Ferret symbols with std 
	   IF ( mr_type(mv) .NE. ptype_string ) THEN
              symbuff = 'STAT_STD'
              slen = TM_LENSTR1(rbuff(1))
      	      CALL PUTSYM( symbuff, rbuff(1), slen, i2)

           ENDIF
	ENDIF

	IF ( full .AND. ngood.LE.1 .AND.
     .        mr_type(mv) .NE. ptype_string) THEN
           symbuff = 'STAT_STD'
           slen = 3
           CALL PUTSYM( symbuff, 'bad', slen, i2)
	ENDIF

! Define Ferret symbols with ngood and nbad
	IF ( full .AND. mr_type(mv) .NE. ptype_string ) THEN
           symbuff = 'STAT_NGOOD'
           rbuff(1) = LEFINT(ngood, slen)
           slen = TM_LENSTR1(rbuff(1))
           CALL PUTSYM( symbuff, rbuff(1), slen, i2)

           symbuff = 'STAT_NBAD'
           rbuff(1) = LEFINT(nbad, slen)
           slen = TM_LENSTR1(rbuff(1))
           CALL PUTSYM( symbuff, rbuff(1), slen, i2)
        ENDIF

5000    RETURN
5100	CALL ERRMSG( ferr_invalid_command, status,
     .		     'variable has more than 4 dimensions. Set mode 6D_LAB'
     .		     //pCR//cmnd_buff(:len_cmnd),
     .			*5000 )
	END
