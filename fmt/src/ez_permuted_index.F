	INTEGER FUNCTION EZ_PERMUTED_INDEX( )

* work routine for EZ_READ when /ORDER is not the default X-Y-Z-T
* Determine the single-dimensioned array index equivalent to the 4D
* (i,j,k,l) of the "next" word to be stored

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Thermal Modeling and Analysis Program

* revision 0.00 -  3/4/94 *sh*

* calling arguments:
* None - all indexing is done in COMMON to maximize speed (and because
* algorithm is based on incrementing the last position rather than 
* recomputing cold from arguments

* internal variable declarations:

#ifdef unix
        include 'tmap_format/xez_order.cmn_text'
#else
	INCLUDE 'TMAP_FORMAT:XEZ_ORDER.CMN'
#endif

* convenience equivalences
	INTEGER	nx, ny, nz,
     .		 i,  j,  k,  l,
     .		o1, o2, o3, o4
	EQUIVALENCE (eznn(1),nx), (eznn(2),ny), (eznn(3),nz),
     .		    (ezii(1),i),(ezii(2),j),(ezii(3),k),(ezii(4),l),
     .		    (ezorder(1),o1), (ezorder(2),o2), 
     .		    (ezorder(3),o3), (ezorder(4),o4) 


* local variable definitions:

* increment all 4 subscripts in appropriate permutation
	ezii(o1) = ezii(o1) + 1
	IF ( ezii(o1) .GE. eznn(o1) ) THEN
	   ezii(o1) = 0
	   ezii(o2) = ezii(o2) + 1
	   IF ( ezii(o2) .GE. eznn(o2) ) THEN
	      ezii(o2) = 0
	      ezii(o3) = ezii(o3) + 1
	      IF ( ezii(o3) .GE. eznn(o3) ) THEN
	         ezii(o3) = 0
	         ezii(o4) = ezii(o4) + 1
	      ENDIF
	   ENDIF
	ENDIF

* compute the 1D index
	EZ_PERMUTED_INDEX = ( nx*(ny*(nz*l + k) + j) + i ) + 1

	RETURN

      END
