	SUBROUTINE PERMUTE_SUB( src_cx, perm,
     .		src, slox, shix, sloy, shiy, sloz, shiz, slot, shit,
     .		dst, dlox, dhix, dloy, dhiy, dloz, dhiz, dlot, dhit )

* permute the 4D data from the source grid into the destination grid
* - 24 possible permutations  (!4)

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200: 10/16/89 - extracted from PERMUTE
*	 3/16/90 - fixed bug: "src(i,j,l,k)" --> "src(i,j,k,l)"

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include	'ferret_cmn/xcontext.cmn'
	include	'ferret_cmn/xvariables.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
#endif

* calling argument declarations:
	INTEGER src_cx, perm(4),
     .		slox, shix, sloy, shiy, sloz, shiz, slot, shit,
     .		dlox, dhix, dloy, dhiy, dloz, dhiz, dlot, dhit
	REAL    src( slox:shix,sloy:shiy,sloz:shiz,slot:shit ),
     .		dst( dlox:dhix,dloy:dhiy,dloz:dhiz,dlot:dhit )

* internal variable declarations:
	INTEGER	loi, loj, lok, lol, hii, hij, hik, hil, i, j, k, l, idim
	INTEGER	lo(4), hi(4)
	EQUIVALENCE
     .		(lo(1), loi), (hi(1), hii),
     .		(lo(2), loj), (hi(2), hij),
     .		(lo(3), lok), (hi(3), hik),
     .		(lo(4), lol), (hi(4), hil)

* local parameter declarations
	INTEGER	i_ax, j_ax, k_ax, l_ax
	PARAMETER  (i_ax=1, j_ax=2, k_ax=3, l_ax=4)

* --- end of introductory code ---
* initialize - save loop indices as unsubscripted variables
	DO 100 idim = 1, 4
	   lo(idim) = cx_lo_ss(src_cx,idim)
	   hi(idim) = cx_hi_ss(src_cx,idim)
 100	CONTINUE

**************
	IF ( perm(1) .EQ. i_ax ) THEN
	   IF ( perm(2) .EQ. j_ax ) THEN
	      IF ( perm(3) .EQ. k_ax ) THEN
* IJKL
	        DO 1  l = lol, hil
	        DO 1  k = lok, hik
	        DO 1  j = loj, hij
	        DO 1  i = loi, hii
 1	        dst(i,j,k,l) = src(i,j,k,l)
	      ELSE
* IJLK
	        DO 2  l = lol, hil
	        DO 2  k = lok, hik
	        DO 2  j = loj, hij
	        DO 2  i = loi, hii
 2	        dst(i,j,l,k) = src(i,j,k,l)
	      ENDIF

	   ELSEIF ( perm(2) .EQ. k_ax ) THEN
	      IF ( perm(3) .EQ. j_ax ) THEN
* IKJL
	        DO 3  l = lol, hil
	        DO 3  k = lok, hik
	        DO 3  j = loj, hij
	        DO 3  i = loi, hii
 3	        dst(i,k,j,l) = src(i,j,k,l)
	      ELSE
* IKLJ
	        DO 4  l = lol, hil
	        DO 4  k = lok, hik
	        DO 4  j = loj, hij
	        DO 4  i = loi, hii
 4	        dst(i,k,l,j) = src(i,j,k,l)
	      ENDIF

	   ELSEIF ( perm(2) .EQ. l_ax ) THEN
	      IF ( perm(3) .EQ. j_ax ) THEN
* ILJK
	        DO 5  l = lol, hil
	        DO 5  k = lok, hik
	        DO 5  j = loj, hij
	        DO 5  i = loi, hii
 5	        dst(i,l,j,k) = src(i,j,k,l)
	      ELSE
* ILKJ
	        DO 6  l = lol, hil
	        DO 6  k = lok, hik
	        DO 6  j = loj, hij
	        DO 6  i = loi, hii
 6	        dst(i,l,k,j) = src(i,j,k,l)
	      ENDIF
	   ENDIF

**************
	ELSEIF ( perm(1) .EQ. j_ax ) THEN
	   IF ( perm(2) .EQ. i_ax ) THEN
	      IF ( perm(3) .EQ. k_ax ) THEN
* JIKL
	        DO 7  l = lol, hil
	        DO 7  k = lok, hik
	        DO 7  j = loj, hij
	        DO 7  i = loi, hii
 7	        dst(j,i,k,l) = src(i,j,k,l)
	      ELSE
* JILK
	        DO 8  l = lol, hil
	        DO 8  k = lok, hik
	        DO 8  j = loj, hij
	        DO 8  i = loi, hii
 8	        dst(j,i,l,k) = src(i,j,k,l)
	      ENDIF

	   ELSEIF ( perm(2) .EQ. k_ax ) THEN
	      IF ( perm(3) .EQ. i_ax ) THEN
* JKIL
	        DO 9  l = lol, hil
	        DO 9  k = lok, hik
	        DO 9  j = loj, hij
	        DO 9  i = loi, hii
 9	        dst(j,k,i,l) = src(i,j,k,l)
	      ELSE
* JKLI
	        DO 10  l = lol, hil
	        DO 10  k = lok, hik
	        DO 10  j = loj, hij
	        DO 10  i = loi, hii
 10	        dst(j,k,l,i) = src(i,j,k,l)
	      ENDIF

	   ELSEIF ( perm(2) .EQ. l_ax ) THEN
	      IF ( perm(3) .EQ. i_ax ) THEN
* JLIK
	        DO 11 l = lol, hil
	        DO 11 k = lok, hik
	        DO 11 j = loj, hij
	        DO 11 i = loi, hii
 11	        dst(j,l,i,k) = src(i,j,k,l)
	      ELSE
* JLKI
	        DO 12 l = lol, hil
	        DO 12 k = lok, hik
	        DO 12 j = loj, hij
	        DO 12 i = loi, hii
 12	        dst(j,l,k,i) = src(i,j,k,l)
	      ENDIF
	   ENDIF

**************
	ELSEIF ( perm(1) .EQ. k_ax ) THEN
	   IF ( perm(2) .EQ. i_ax ) THEN
	      IF ( perm(3) .EQ. j_ax ) THEN
* KIJL
	        DO 13 l = lol, hil
	        DO 13 k = lok, hik
	        DO 13 j = loj, hij
	        DO 13 i = loi, hii
 13	        dst(k,i,j,l) = src(i,j,k,l)
	      ELSE
* KILJ
	        DO 14 l = lol, hil
	        DO 14 k = lok, hik
	        DO 14 j = loj, hij
	        DO 14 i = loi, hii
 14	        dst(k,i,l,j) = src(i,j,k,l)
	      ENDIF

	   ELSEIF ( perm(2) .EQ. j_ax ) THEN
	      IF ( perm(3) .EQ. i_ax ) THEN
* KJIL
	        DO 15 l = lol, hil
	        DO 15 k = lok, hik
	        DO 15 j = loj, hij
	        DO 15 i = loi, hii
 15	        dst(k,j,i,l) = src(i,j,k,l)
	      ELSE
* KJLI
	        DO 16  l = lol, hil
	        DO 16  k = lok, hik
	        DO 16  j = loj, hij
	        DO 16  i = loi, hii
 16	        dst(k,j,l,i) = src(i,j,k,l)
	      ENDIF

	   ELSEIF ( perm(2) .EQ. l_ax ) THEN
	      IF ( perm(3) .EQ. i_ax ) THEN
* KLIJ
	        DO 17 l = lol, hil
	        DO 17 k = lok, hik
	        DO 17 j = loj, hij
	        DO 17 i = loi, hii
 17	        dst(k,l,i,j) = src(i,j,k,l)
	      ELSE
* KLJI
	        DO 18 l = lol, hil
	        DO 18 k = lok, hik
	        DO 18 j = loj, hij
	        DO 18 i = loi, hii
 18	        dst(k,l,j,i) = src(i,j,k,l)
	      ENDIF
	   ENDIF

**************
	ELSEIF ( perm(1) .EQ. l_ax ) THEN
	   IF ( perm(2) .EQ. i_ax ) THEN
	      IF ( perm(3) .EQ. j_ax ) THEN
* LIJK
	        DO 19 l = lol, hil
	        DO 19 k = lok, hik
	        DO 19 j = loj, hij
	        DO 19 i = loi, hii
 19	        dst(l,i,j,k) = src(i,j,k,l)
	      ELSE
* LIKJ
	        DO 20 l = lol, hil
	        DO 20 k = lok, hik
	        DO 20 j = loj, hij
	        DO 20 i = loi, hii
 20	        dst(l,i,k,j) = src(i,j,k,l)
	      ENDIF

	   ELSEIF ( perm(2) .EQ. j_ax ) THEN
	      IF ( perm(3) .EQ. i_ax ) THEN
* LJIK
	        DO 21 l = lol, hil
	        DO 21 k = lok, hik
	        DO 21 j = loj, hij
	        DO 21 i = loi, hii
 21	        dst(l,j,i,k) = src(i,j,k,l)
	      ELSE
* LJKI
	        DO 22  l = lol, hil
	        DO 22  k = lok, hik
	        DO 22  j = loj, hij
	        DO 22  i = loi, hii
 22	        dst(l,j,k,i) = src(i,j,k,l)
	      ENDIF

	   ELSEIF ( perm(2) .EQ. k_ax ) THEN
	      IF ( perm(3) .EQ. i_ax ) THEN
* LKIJ
	        DO 23 l = lol, hil
	        DO 23 k = lok, hik
	        DO 23 j = loj, hij
	        DO 23 i = loi, hii
 23	        dst(l,k,i,j) = src(i,j,k,l)
	      ELSE
* LKJI
	        DO 24 l = lol, hil
	        DO 24 k = lok, hik
	        DO 24 j = loj, hij
	        DO 24 i = loi, hii
 24	        dst(l,k,j,i) = src(i,j,k,l)
	      ENDIF
	   ENDIF

	ENDIF

	RETURN
	END 
