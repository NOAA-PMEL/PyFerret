    	SUBROUTINE SHOW_TRANSFORMS

* show a nicely formatted list of the available transforms

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V420:  11/95 - lifted from XEQ_SHOW
*	       - and SHOW LIST (tab and comma delimited formats)
*	 12/95 - show explicit ITP as well (start at index 3 instead of 4)
* V430:   7/96 - incorporate 4D transformations: AVE, VAR, DIN, NDG, NBD

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
	INTEGER n, i, i1, i2, i4d

* count number of transforms
	n=0
	DO 110 i = 1, num_transforms
	   IF ( alg_trans_num( i ) .EQ. unspecified_int4  ) GOTO 110
	   IF ( alg_trans_num( i ) .EQ. trans_no_transform) GOTO 110
* ... skip hidden transforms (e.g. 4D internal implementations)
	   DO 109 i4d = 1, num_4d_trans
	      IF (alg_trans_num( i ) .EQ. alg_4d_trans(i4d)) GOTO 110
 109	   CONTINUE
	   n = n+1
 110	CONTINUE
* locate half-way point in the list
	i = 0
	DO 112 i2 = 1, num_transforms
	   IF ( alg_trans_num( i2 ) .EQ. unspecified_int4  ) GOTO 112
	   IF ( alg_trans_num( i2 ) .EQ. trans_no_transform) GOTO 112
* ... skip hidden transforms (e.g. 4D internal implementations)
	   DO 111 i4d = 1, num_4d_trans
	      IF (alg_trans_num( i2 ) .EQ. alg_4d_trans(i4d)) GOTO 112
 111	   CONTINUE
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
	i1 = 3 - 1
	i2 = i2 - 1
	DO 150 i = 1, (n+1)/2
* ... locate first pointer
           DO 120 i1 = i1+1, num_transforms
              IF ( alg_trans_num( i1 ) .EQ. unspecified_int4  ) GOTO 120
	      IF ( alg_trans_num( i1 ) .EQ. trans_no_transform) GOTO 120
* ... skip hidden transforms (e.g. 4D internal implementations)
	      DO 119 i4d = 1, num_4d_trans
	         IF (alg_trans_num( i1 ) .EQ. alg_4d_trans(i4d)) GOTO 120
 119	      CONTINUE
	      GOTO 121
 120	   CONTINUE
* ... locate second pointer
 121	   DO 130 i2 = i2+1, num_transforms
              IF ( alg_trans_num( i2 ) .EQ. unspecified_int4  ) GOTO 130
	      IF ( alg_trans_num( i2 ) .EQ. trans_no_transform) GOTO 130
* ... skip hidden transforms (e.g. 4D internal implementations)
	      DO 129 i4d = 1, num_4d_trans
	         IF (alg_trans_num( i1 ) .EQ. alg_4d_trans(i4d)) GOTO 130
 129	      CONTINUE
	      GOTO 131
 130	   CONTINUE
 131	   IF ( i .LE. n/2 ) THEN
* ... write two transforms per line
              WRITE (risc_buff, 3054)
     .                 alg_trans_text(i1), alg_trans_title(i1),
     .                 alg_trans_text(i2), alg_trans_title(i2)
	   ELSE
* ... write the last transformation on a line alone
              WRITE (risc_buff, 3055)
     .                 alg_trans_text(i1), alg_trans_title(i1)
	   ENDIF
	   CALL SPLIT_LIST(pttmode_explct, ttout_lun, risc_buff, 0)
	   risc_buff = ' '
 150   CONTINUE
 3054   FORMAT (T4,'@',A3,T16,A20,T45,'@',A3,T57,A20)
 3055   FORMAT (T4,'@',A3,T16,A20)
	RETURN

	END
