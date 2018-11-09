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
* V650  Write to show_lun which may be set to a file unit number rather than std out
* V671  6/11 *acm*  New 2D linear xy regridding, fixing ticket 1862
*       *acm* 3/12 6D Ferret (common uses nferdims in tmap_dims.parm)
* V690  12/13 *sh* show auxiliary variable regridding transforms
*    Note:  this routine and SHOW_TRANSFORMS are both messy and could be
*           readily cleaned up and consolidated into 1 piece of code
*           by creating a single output routine and using logical arrays
*           to indicate which transforms to hide during SHOW TRANSFORMS


        IMPLICIT NONE
	include 'tmap_dims.parm'
	include	'ferret.parm'
	include	'xprog_state.cmn'
	include	'xalgebra.cmn'
	include 'xrisc.cmn'          ! 12/94 SPLIT_LIST buffer

* calling argument declarations:
*...

* local variable declarations:
	INTEGER n, i, i1, i2

* introductory line
	CALL SPLIT_LIST(pttmode_explct, show_lun,
     .		'regridding transforms e.g.SST[GX=x5deg@AVE]', 43)

* count number of transforms
	n=0
	DO 110 i = 1, max_regrid_trans
           IF (.NOT.alg_regrid_vizb(i)) GOTO 110
	   n = n+1
 110	CONTINUE
* locate half-way point in the list
	i = 0
	DO 112 i2 =  1, max_regrid_trans
           IF (.NOT.alg_regrid_vizb(i2)) GOTO 112
           i = i+1
	   IF ( i .GT. (n+1)/2 ) GOTO 115
 112	CONTINUE
* heading lines
 115	WRITE (risc_buff, 3050) ('code','description',i=1,2)
	CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)
	risc_buff = ' '
        WRITE (risc_buff, 3050) ('----','-----------',i=1,2)
	CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)
	risc_buff = ' '
 3050   FORMAT (T4,A,T16,A,T45,A,T57,A)
* list the transforms 2 per line - use 2 separate pointers
	i1 = 0
	i2 = i2 - 1
	DO 150 i = 1, (n+1)/2
* ... locate first pointer
           DO 120 i1 = i1+1, max_regrid_trans
              IF (.NOT.alg_regrid_vizb(i1)) GOTO 120
	      GOTO 121
 120	   CONTINUE
* ... locate second pointer
 121	   DO 130 i2 = i2+1,  max_regrid_trans
              IF (.NOT.alg_regrid_vizb(i2)) GOTO 130
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
	   CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)
	   risc_buff = ' '
 150   CONTINUE

***********
* now the auxiliary regridding transforms

* introductory line
	CALL SPLIT_LIST(pttmode_explct, show_lun, ' ', 1)
	CALL SPLIT_LIST(pttmode_explct, show_lun,
     .		'auxiliary varible regridding transforms'//
     .          ' e.g.TEMP[GZ(depth)=zax@AVE]', 0)

* count number of transforms
	n=0
	DO 210 i = max_regrid_trans+1, max_auxregrid_trans
           IF (.NOT.alg_regrid_vizb(i)) GOTO 210
	   n = n+1
 210	CONTINUE
* locate half-way point in the list
	i = 0
	DO 212 i2 =  max_regrid_trans+1, max_auxregrid_trans
           IF (.NOT.alg_regrid_vizb(i2)) GOTO 212
           i = i+1
	   IF ( i .GT. (n+1)/2 ) GOTO 215
 212	CONTINUE
* heading lines
 215	WRITE (risc_buff, 3050) ('code','description',i=1,2)
	CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)
	risc_buff = ' '
        WRITE (risc_buff, 3050) ('----','-----------',i=1,2)
	CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)
	risc_buff = ' '
* list the transforms 2 per line - use 2 separate pointers
	i1 = max_regrid_trans
	i2 = i2 - 1
	DO 250 i = 1, (n+1)/2
* ... locate first pointer
           DO 220 i1 = i1+1, max_auxregrid_trans
              IF (.NOT.alg_regrid_vizb(i1)) GOTO 220
	      GOTO 221
 220	   CONTINUE
* ... locate second pointer
 221	   DO 230 i2 = i2+1,  max_regrid_trans
              IF (.NOT.alg_regrid_vizb(i2)) GOTO 230
	      GOTO 231
 230	   CONTINUE
 231	   IF ( i .LE. n/2 ) THEN
* ... write two transforms per line
              WRITE (risc_buff, 3054)
     .                 alg_regrid_code(i1), alg_regrid_title(i1),
     .                 alg_regrid_code(i2), alg_regrid_title(i2)
	   ELSE
* ... write the last transformation on a line alone
              WRITE (risc_buff, 3055)
     .                 alg_regrid_code(i1), alg_regrid_title(i1)
	   ENDIF
	   CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)
	   risc_buff = ' '
 250   CONTINUE


***********
 3054   FORMAT (T4,'@',A6,T16,A20,T45,'@',A6,T57,A20)
 3055   FORMAT (T4,'@',A6,T16,A20)
	RETURN

	END
