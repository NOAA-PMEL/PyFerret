	SUBROUTINE VAR_STAT( dat, mv, cx, lun, full )

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

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xvariables.cmn'
	include	'ferret_cmn/xmem_subsc.cmn'
	include 'ferret_cmn/xcontext.cmn'
        include 'ferret_cmn/xrisc.cmn'           ! for Unix/RISC bugs (&12/94)
* special equivalence needed only with risc compiler: bug fix
        CHARACTER rbuff(4)*12
        EQUIVALENCE ( rbuff, risc_buff )
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XMEM_SUBSC.CMN'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
#endif

* calling argument declarations:
	LOGICAL	full
	INTEGER	mv, cx, lun
	REAL    dat( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit )

* internal variable declarations:
	INTEGER	TM_LENSTR1,
     .		i, j, k, l, nbad, ngood, ntot, idim,
     .		len_line, len_set_name, grid, slen
	REAL	datum, bad, std_dev, large, small, mean, sumsq_dev, dev
	REAL*8	sum
	CHARACTER*255	VAR_TITLE, VAR_TITLE_MOD, vname_buff
	CHARACTER	TM_FMT*12, LEFINT*8, AX_TITLE*32, CX_SIZE_STR*12,
     .			line_buff*130, title*32, bl13*13
	PARAMETER (bl13 = '             ' )

* local parameter declarations:
	INTEGER		sig_fig
	PARAMETER     ( sig_fig = 5 )

* initialize
	grid  = mr_grid( mv )
	bad   = mr_bad_data( mv )
	nbad  = 0
	ngood = 0
	small = arbitrary_large_val8
	large = arbitrary_small_val8
	sum   = 0.0D0
        sumsq_dev = 0.0

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
	   DO 60 idim = 1,4
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

* pass 1: compute mean 
	DO 200 l = cx_lo_s4(cx), cx_hi_s4(cx)
	DO 200 k = cx_lo_s3(cx), cx_hi_s3(cx)
	DO 200 j = cx_lo_s2(cx), cx_hi_s2(cx)
	DO 200 i = cx_lo_s1(cx), cx_hi_s1(cx)

	   datum = dat(i,j,k,l)

	   IF ( datum .EQ. bad ) THEN
	      nbad = nbad + 1

	   ELSE
	      ngood = ngood + 1
	      small = MIN( small, datum )
	      large = MAX( large, datum )
              IF ( ABS(sum) .LT. 3.D35 ) sum   = sum + datum
	   ENDIF
 200	CONTINUE
        IF ( ngood .GT. 0 ) mean = sum / ngood
        IF ( ABS(sum) .GE. 3.D35 ) sumsq_dev = 3.E35  ! flag for next pass

	IF ( full ) THEN
* pass 2: compute variance 
           IF ( ngood .GT. 1 ) THEN
	      DO 300 l = cx_lo_s4(cx), cx_hi_s4(cx)
	      DO 300 k = cx_lo_s3(cx), cx_hi_s3(cx)
	      DO 300 j = cx_lo_s2(cx), cx_hi_s2(cx)
	      DO 300 i = cx_lo_s1(cx), cx_hi_s1(cx)

	         datum = dat(i,j,k,l)
	         IF ( datum .NE. bad ) THEN
                    dev = datum - mean
	            IF ( sumsq_dev .LT. 3.E35 )
     .                   sumsq_dev = sumsq_dev + dev*dev
	         ENDIF
 300	      CONTINUE
	      std_dev   = SQRT( sumsq_dev / (ngood-1) )     ! was **0.5  10/93
           ENDIF
	ENDIF

* compute and display results
	ntot = nbad + ngood

        rbuff(1) = LEFINT(nbad, slen)
        rbuff(2) = LEFINT(ntot, slen)
        rbuff(3) = CX_SIZE_STR(cx,len_line)
	CALL SPLIT_LIST(pttmode_explct, lun,' ',1)
	CALL SPLIT_LIST(pttmode_explct, lun,
     .			' Total # of data points: '//
     .			rbuff(2)(:slen)//' ('//rbuff(3)(:len_line)//')',0)
	CALL SPLIT_LIST(pttmode_explct, lun,
     .			' # flagged as bad  data: '//rbuff(1),0)
	IF ( ngood .GT. 0 ) THEN
           rbuff(1) = TM_FMT(small, sig_fig, 12, slen)
           rbuff(2) = TM_FMT(large, sig_fig, 12, slen)
           IF ( ABS(sum) .GE. 3.D35 ) THEN
              rbuff(3) = '**too big**' 
           ELSE
              rbuff(3) = TM_FMT(mean , sig_fig, 12, slen)
           ENDIF
	   CALL SPLIT_LIST(pttmode_explct, lun,
     .			' Minimum value: '//rbuff(1), 0)
	   CALL SPLIT_LIST(pttmode_explct, lun,
     .			' Maximum value: '//rbuff(2), 0)
	   CALL SPLIT_LIST(pttmode_explct, lun,
     .			' Mean    value: '//rbuff(3)//
     .					' (unweighted average)', 0)
	ENDIF
	IF ( full .AND. ngood.GT.1 ) THEN
           IF ( sumsq_dev .GE. 3.E35 ) THEN
              rbuff(1) = ' **too big**' 
           ELSE
              rbuff(1) = TM_FMT(std_dev, sig_fig, 12, slen)
           ENDIF
	   CALL SPLIT_LIST(pttmode_explct, lun,
     .			' Standard deviation: '//rbuff(1), 0)
	ENDIF

	RETURN
	END

