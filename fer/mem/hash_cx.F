	INTEGER FUNCTION HASH_CX( cx )

* generate a pseudo-random pile number between 1 and pmv_npiles based on
* information in the passed context

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 7/21/88
* V200:  7/12/89 - special treatment for cat_temp_var


#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xvariables.cmn'	! for pmv_npiles
	include 'ferret_cmn/xcontext.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'	! for pmv_npiles
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
#endif

* calling argument declarations:
	INTEGER		cx

* internal variable declarations:
	INTEGER		idim, nsum, cat

* initialize
	cat = cx_category( cx )

* special Q&D sol'n for temporary variables, etc.
	IF ( cat .EQ. cat_constant 
     .  .OR. cat .EQ. cat_temp_var
     .  .OR. cat .EQ. cat_pseudo_var ) THEN
	   HASH_CX = -1
	   RETURN
	ENDIF	   
	
* arbitrary number generated from context information
	nsum = cx_data_set( cx )
     .	     + cx_variable( cx ) * 7
     .	     + cat * 13
	DO 100 idim = 1, 4
	   IF ( cx_trans( idim, cx ) .NE. trans_no_transform )
     .							nsum = nsum + idim*idim	
 100	CONTINUE

* randomize the number and return a NEGATIVE pile number
* ( corresponding to the negative subscripts of the hash pile pointers )
	HASH_CX = -1 * ( MOD( 2107*nsum, pmv_npiles ) + 1 )

	RETURN
	END
