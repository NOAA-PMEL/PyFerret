      SUBROUTINE setlab_syms (zlevels, nlevels)
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
*  updates, revisions, new versions or 'bug fixes'.
*
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP 'AS IS' AND ANY EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE. 
*
*  v5.5  *acm* 11/2002 New symbols to save levels
* v552 *acm* 3/03 Longer strings: changes for multi-line labels.
* V581 *acm* 3/05 fix bug 1205: If delta was set by user (with SHADE/LEV=.4d), then 
*                 use the given delta rather than trying to compute it for LEV_DEL
* V62 *acm*  5/08 Fix bug 1581. NLEV used is acutally one less
* V65 *acm   2/10 If Open-ended levels were used, symbol LEV_OPNLEVS saves 
* the /LEV command needed to duplicate the levels
* V68  *acm* 1/12  changes for double-precision ferret, single-precision pplus
* V686 *acm* 11/13 Allow symbol names to be up to 120 characters long
* V698 *acm*  2/16 Fix the LEV_DEL computation which was wrong for irregular levels.
*                  Clear out-of-date symbols, for variance-based or histogram
*                  levels on previous plots
* V720 *acm*  7/17 Ticket 2250: precision errors when data has very small values
* V74+  *acm* 3/18 Issue 1607.  /KEY=centerlab

*    LEV_MIN  - lowest level used on last CONTOUR, SHADE, or FILL call.
*    LEV_MAX -  highest level used last
*    LEV_DEL - either the value or 'irregular'
*    LEV_NUM - Number of levels used
*    LEV_ALL - ?? list of all the levels used

*    (LEV_TEXT - full text string of last levels used, set in routine LEV.F)

        IMPLICIT NONE
	include 	'parampl5_dat.decl'
	include 	'PARAMPL5.DAT'
	include		'cont_inc.decl'
	include		'CONT.INC'
        include         'shade_vars.cmn'

* Calling argument declarations

      REAL zlevels(*)
      INTEGER nlevels

* Internal arguments

      REAL zmax, zmin, dz, diff, eps
      REAL*8 zmaxD, zminD

      CHARACTER ppl_buff*2048, TM_FMT_SNGL*10, TM_FMT*10, 
     .          buff*10, symbuff*30,buffmin*10, buffmax*10

     
	CHARACTER symbuff30*30, title*32

      INTEGER TM_LENSTR1, STR_SAME, i, ier, ilen, slen, ist
      LOGICAL TM_FPEQ_EPS_SNGL, reglevs

* Lowest and highest Z level

      zmin = zlevels(1)
      buffmin = TM_FMT_SNGL(zmin, -2, 10, slen)
      IF (neginf) buffmin = 'open-ended'

      zmax = zlevels(nlevels)
      buffmax = TM_FMT_SNGL(zmax, -2, 10, slen)
      IF (posinf) buffmax = 'open-ended'

* An epsilon for equal-sized deltas, need not be too tight a test
      eps = 0.01* (zmax - zmin)/FLOAT(nlevels)

      IF ( STR_SAME(buffmin, buffmax).EQ.0 .AND. .NOT.neginf .AND.
     .    .NOT.TM_FPEQ_EPS_SNGL(eps, zmin, zmax) ) THEN
	 zminD = zmin
         zmaxD = zmax 
	 buffmin = TM_FMT(zminD, 2, 10, slen)
	 buffmax = TM_FMT(zmaxD, 2, 10, slen)
      ENDIF

      ppl_buff = 'LEV_MIN '//buffmin
      ilen = TM_LENSTR1(ppl_buff)
      ier = 0
      ist = 0
      CALL SETSYM (ppl_buff, ilen, ier, ist)


      ppl_buff = 'LEV_MAX '//buffmax
      ilen = TM_LENSTR1(ppl_buff)
      ier = 0
      ist = 0
      CALL SETSYM (ppl_buff, ilen, ier, ist)

* Number of levels used 

      zmax = MAX(nlevels-1,1)
      buff = TM_FMT_SNGL(zmax, -2, 10, slen)

      ppl_buff = 'LEV_NUM '//buff
      ilen = TM_LENSTR1(ppl_buff)
      ier = 0
      ist = 0
      CALL SETSYM (ppl_buff, ilen, ier, ist)
      
* If set_delta is .true. the delta value has been set
* elsewhere. use the delta given.

      IF (set_delta) THEN
	 dz = delta
         buff = TM_FMT_SNGL(dz, -2, 10, slen)

* Otherwise, check if levels are regularly spaced 

      ELSE


         dz = zlevels(2) - zlevels(1)
      
         IF (nlevels .EQ. 1) THEN
            buff = 'none'
         ELSE IF (nlevels .EQ. 2) THEN
            buff = TM_FMT_SNGL(dz, -2, 10, slen)
         ELSE
            reglevs = .TRUE.
            DO 100 i=3,nlevels
               diff = ABS( zlevels(i)-zlevels(i-1) )
               reglevs = TM_FPEQ_EPS_SNGL(eps, diff, dz)
	       IF (.NOT.reglevs) GOTO 110
  100       CONTINUE
  110       IF (reglevs) THEN
               buff = TM_FMT_SNGL(dz, -2, 10, slen)
            ELSE
               buff = 'irregular'
            ENDIF
         ENDIF

      ENDIF

      IF (STR_SAME(buff, '0') .EQ. 0 .AND. reglevs) THEN
	 zminD = zlevels(1)
         zmaxD = zlevels(2)
	 zminD = zmaxD - zminD
	 buff = TM_FMT(zminD, 2, 10, slen)
      ENDIF

      ppl_buff = 'LEV_DEL '//buff
      ilen = TM_LENSTR1(ppl_buff)
      ier = 0
      ist = 0
      CALL SETSYM (ppl_buff, ilen, ier, ist)
 
* If Open-ended levels were used, symbol LEV_OPNLEVS saves the /LEV 
* command needed to duplicate the levels

      symbuff = 'LEV_OPNLEVS'
      CALL GETSYM( symbuff, ppl_buff, ist, ier )
      IF ( ier .EQ. 0 ) THEN
         IF (STR_SAME(lev_str, 'none').EQ.0 .OR. 
     .       STR_SAME(lev_str, 'histogram').EQ.0) THEN 
            CALL DELSYM(symbuff,ier)
	 ENDIF
      ENDIF

      IF (STR_SAME(lev_str, 'none').NE.0 .AND. 
     .    STR_SAME(lev_str, 'histogram').NE.0) THEN
         ilen = TM_LENSTR1(lev_str)
         ppl_buff = 'LEV_OPNLEVS '//lev_str
         ilen = TM_LENSTR1(ppl_buff)
         ier = 0
         ist = 0
         CALL SETSYM (ppl_buff, ilen, ier, ist)
      ENDIF

* Were the levels specified as /KEY=CENTERLAB?

      IF (center_key_labels) THEN 
         ppl_buff = 'LEV_CENTERED_LABELS on'
      ELSE
         ppl_buff = 'LEV_CENTERED_LABELS off'
      ENDIF 
      ilen = TM_LENSTR1(ppl_buff)
      ier = 0
      ist = 0
      CALL SETSYM (ppl_buff, ilen, ier, ist)

* Delete any LEV_HISTO_LEVELS symbol that may be left over from a previous plot

      IF (.NOT. USE_HIST) THEN
         symbuff = 'LEV_HISTO_LEVELS'
         CALL GETSYM( symbuff, ppl_buff, ist, ier )
         IF (ier .EQ. 0) CALL DELSYM(symbuff,ier)
      ENDIF

      RETURN
      END
