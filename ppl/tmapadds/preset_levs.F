       SUBROUTINE PRESET_LEVS
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
* v602 *acm* 2/13/2007
* Called when the data is loaded in to pplus memory (pplldc or pplldx)
* to set the levels symbols at plot setup time. This means that the symbols
* are available after a [plot_command]/SET .
* v602 *acm*  2/07  Changes so that /SET on a 2D plot command sets up the LEVELS symbols.
*                   (bug 1491)

	include		'parampl5_dat.decl'
	include		'PARAMPL5.DAT'
	include		'hd_inc.decl'
	include		'HD.INC'
	include		'cont_inc.decl'
	include		'CONT.INC'
	include         'errmsg.parm'
        
        REAL dz, znp, zmp, fm, epsilon
        INTEGER m, i, slen1, slen2
        LOGICAL         TM_FPEQ
        CHARACTER       TM_FMT*20, buff1*20, buff2*20, err_msg*40
        PARAMETER       (epsilon = 1e-6)

        EXTERNAL range

        WARNED_NLEV = .FALSE.

	IF (IAUTOC.EQ.1) THEN

	   IF (ZMIN.EQ.ZMAX) THEN

	      NLEV2=1
              zlev(1) = zmin

	   ELSE
C       
C       If set_delta is .true. the delta value has been set
C       elsewhere and we should calculate the max and min values
C       of the range based on this delta.
C       (This portion stolen from RANGE)

	      if (set_delta) then 

		 dz = delta
C       
C       CALCULATE THE MINIMUM VALUE OF THE RANGE
C       
		 FM = zmin / DZ
		 M = IFIX(FM)
		 IF (FM .LT. 0.0) M = M - 1
		 znp = DZ * FLOAT(M)
C       
C       CALCULATE THE MAXIMUM VALUE OF THE RANGE.
C       
		 FM = zmax / DZ
		 M = IFIX(FM)
		 IF(FM .GT. 0.0)M = M + 1
		 zmp = DZ * FLOAT(M)
C
C       Set levels centered about zero if centered is .true.
C
	         IF (centered) THEN
		    IF (abs(znp) .GT. abs(zmp)) THEN
		       zmp = -znp
		    ELSE
		       znp = -zmp
		    ENDIF
	         ENDIF

C       Else we should allow RANGE to determine the max and min range
C       values and also the appropriate delta value. (ZNP, ZMP, DZ)

	      ELSE

	         znp = zmin
	         zmp = zmax
	         IF (centered) THEN
		    if (abs(znp) .GT. abs(zmp)) THEN
		       zmp = -znp
		    ELSE
		       znp = -zmp
		    ENDIF
	         ENDIF
		 CALL RANGE(znp,zmp,NLEV,ZNP,ZMP,DZ)

	      ENDIF

	      NLEV2=(ZMP-ZNP)/DZ+1 

C       Handle rounding errors *jd* 6.27.95/12.30.97

c *acm* 3/04 under g77 on linux this incorrectly uses
c            low-order bits to reset NLEV2 example:
c             yes?shade/lev=c {-1.1, 1.1}* ysequence({1,1})

c       4/04  But, dont just skip this, use TM_FPEQ to see if
c             values are equal... 
c       4/04  Also fix the negative-range case.

	      IF ( (ZNP + DZ*(NLEV2-1) .LT. ZMP) .AND. 
     .              .NOT. TM_FPEQ(ZNP + DZ*(NLEV2-1), ZMP) ) THEN
		 NLEV2 = NLEV2 + 1
	      ENDIF

              IF (NLEV2 .GT. ICNLEV) GO TO 5000

	      DO 100 I=1,NLEV2 
                 ZLEV(I)=ZNP+DZ*FLOAT(I-1)
 100	      CONTINUE 

	   ENDIF
C       
C       Else IAUTOC != 1 means: use supplied levels
C       
	ELSE 

C       Mod 4.18.90 to allow just on level to be SHADEd
C       CONTOUR requires more than one level to work but SHADE doesn't

	   NLEV2=NLEV
              IF (neginf) zlev(1) = MIN( zlev(1), zmin-1. )
              IF (posinf) zlev(nlev) = MAX (zlev(nlev), zmax+1. )
	END IF  

* Now that the levels are set, create the LEVELS symbols

        CALL setlab_syms (ZLEV, NLEV2)

	RETURN

C  Error exit when too many levels specified.
C  Array ZLEV is length ICNLEV=500

 5000   CONTINUE

* Set the symbols so the values can be checked
        CALL setlab_syms (ZLEV, NLEV2)

* Issue error message
        ii = ICNLEV
        buff1 = TM_FMT (REAL(ICNLEV), 20, 20, slen1)
        buff2 = TM_FMT (REAL(NLEV2), 20, 20, slen2)
        err_msg = '# of shade levels > '// buff1(1:slen1) //
     .          '  (' // buff2(1:slen2) // ')'
        
        WARNED_NLEV = .TRUE.

        CALL errmsg (ferr_out_of_range, status, err_msg, *5002)
 5002   RETURN
	END