    	SUBROUTINE SHOW_REGRID_TRANSFORMS

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
* show a nicely formatted list of the available regridding transforms

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V420:  11/95 - based on SHOW_TRANSFORMS
*	  1/96 - bug fix: crash on last (single transform) line
*			"num_transforms" ==> "max_regrid_trans"
*	  4/96 - branching bug detected by Sun f77 compiler

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/xprog_state.cmn'
	include	'ferret_cmn/xalgebra.cmn'
	include 'ferret_cmn/xrisc.cmn'          ! 12/94 SPLIT_LIST buffer
#else
	INCLUDE	'FERRET_CMN/FERRET.PARM'
	INCLUDE	'FERRET_CMN/XPROG_STATE.CMN'
	INCLUDE	'FERRET_CMN/XALGEBRA.CMN'
	INCLUDE 'FERRET_CMN/XRISC.CMN'          ! 12/94 SPLIT_LIST BUFFER
#endif

* calling argument declarations:
*...

* local variable declarations:
	INTEGER n, i, i1, i2

* count number of transforms
	n=0
	DO 110 i = prgrd_linear_interp, max_regrid_trans
	   IF ( alg_regrid_code(i) .EQ. unspecified_name4 ) GOTO 110
	   IF ( i .EQ. prgrd_xy_ave ) GOTO 110  ! special cases
	   n = n+1
 110	CONTINUE
* locate half-way point in the list
	i = 0
	DO 112 i2 =  prgrd_linear_interp, max_regrid_trans
	   IF ( alg_regrid_code(i2) .EQ. unspecified_name4 ) GOTO 112
	   IF ( i2 .EQ. prgrd_xy_ave ) GOTO 112  ! special cases
           i = i+1
	   IF ( i .GT. (n+1)/2 ) GOTO 115
 112	CONTINUE
* heading lines
 115	WRITE (risc_buff, 3050) ('code','description',i=1,2)
	CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
	risc_buff = ' '
        WRITE (risc_buff, 3050) ('----','-----------',i=1,2)
	CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
	risc_buff = ' '
 3050   FORMAT (T4,A,T16,A,T45,A,T57,A)
* list the transforms 2 per line - use 2 separate pointers
	i1 = prgrd_linear_interp - 1
	i2 = i2 - 1
	DO 150 i = 1, (n+1)/2
* ... locate first pointer
           DO 120 i1 = i1+1, max_regrid_trans
	      IF ( alg_regrid_code(i1) .EQ. unspecified_name4 ) GOTO 120
	      IF ( i1 .EQ. prgrd_xy_ave ) GOTO 120  ! special cases
	      GOTO 121
 120	   CONTINUE
* ... locate second pointer
 121	   DO 130 i2 = i2+1,  max_regrid_trans
	      IF ( alg_regrid_code(i2) .EQ. unspecified_name4 ) GOTO 130
	      IF ( i2 .EQ. prgrd_xy_ave ) GOTO 130  ! special cases
	      GOTO 131
 130	   CONTINUE
 131	   IF ( i .LE. n/2 ) THEN
* ... write two transforms per line
              WRITE (risc_buff, 3054)
     .                 alg_regrid_code(i1), alg_regrid_title(i1),
     .                 alg_regrid_code(i2), alg_regrid_title(i2)
	   ELSE
* ... write the last transformation on a line alone
              WRITE (risc_buff, 3055)
     .                 alg_regrid_code(i1), alg_regrid_title(i1)
	   ENDIF
	   CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
	   risc_buff = ' '
 150   CONTINUE
 3054   FORMAT (T4,'@',A6,T16,A12,T45,'@',A6,T57,A12)
 3055   FORMAT (T4,'@',A6,T16,A12)
	RETURN

	END
