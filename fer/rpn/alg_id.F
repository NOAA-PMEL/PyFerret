	SUBROUTINE ALG_ID( text, natom, at_start, at_end,
     .			   at_type, at_id, status )

* identify the atoms from the text

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* V200: 12/21/89
* Unix/RISC port - 1/91 - cannot "//" with CHAR*(*) variables
* V450: 12/2/96 - check that functions are followed by left parens.
*		  Else, they are variable names.
*		- Also, id grid-changing fcns as category alg_grid_chg_fcn
*		- and check for negative constants 
*		==> Note that this may modify natom
* V495: 7/7/98 - allow variable names in single quotes - bypass name check
*     : 9/30/98 - NAG F90 (linux) received error on above change.  
*                 need to put single  tick between double quotes  *kob*

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/rpn.parm'
	include 'ferret_cmn/xalgebra.cmn'
	include 'ferret_cmn/xrisc.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:RPN.PARM'
	INCLUDE 'FERRET_CMN:XALGEBRA.CMN'
	INCLUDE 'FERRET_CMN:XRISC.CMN'
#endif

* calling argument declarations:
	INTEGER		natom, at_start(maxatoms), at_end(maxatoms),
     .			at_type(maxatoms), at_id(maxatoms), status
	CHARACTER*(*)	text

* internal variable declarations:
	LOGICAL		TM_DIGIT, TM_LEGAL_NAME
	INTEGER		STR_MATCH, GCF_FIND_FCN,
     .			imatch, i, i2, brac_pos, aend

	DO 100 i = 1, natom

* operator ?
	   imatch = STR_MATCH (	text(at_start(i):at_end(i)),
     .				alg_op, num_operators )
	   IF ( imatch .NE. atom_not_found ) THEN
	      at_type(i) = alg_operator
	      at_id(i)   = imatch
	      GOTO 100
	   ENDIF

* punctuation?
	   imatch = STR_MATCH (	text(at_start(i):at_end(i)),
     .				alg_punct, num_punctuations )
	   IF ( imatch .NE. atom_not_found ) THEN
	      at_type(i) = alg_punctuation
	      at_id(i)   = imatch
	      GOTO 100
	   ENDIF

* function ?
	   imatch = STR_MATCH (	text(at_start(i):at_end(i)),
     .				alg_fcn, num_functions )
	   IF ( imatch .NE. atom_not_found ) THEN
	      at_type(i) = alg_function
	      at_id(i)   = imatch
	      GOTO 100
	   ENDIF

* grid-changing function ?
	   imatch = GCF_FIND_FCN( text(at_start(i):at_end(i)) )
	   IF ( imatch .NE. unspecified_int4 ) THEN
	      at_type(i) = alg_grid_chg_fcn
	      at_id(i)   = imatch
	      GOTO 100
	   ENDIF

* logic structure ?
	   imatch = STR_MATCH (	text(at_start(i):at_end(i)),
     .				alg_struct, num_log_structs )
	   IF ( imatch .NE. atom_not_found ) THEN
	      at_type(i) = alg_log_struct
	      at_id(i)   = imatch
	      GOTO 100
	   ENDIF

* constant ?
	  IF ( TM_DIGIT( text(at_start(i):at_start(i)) ) ) THEN
	      at_type(i) = alg_constant
	      at_id(i)   = unspecified_int4
	      GOTO 100
	   ENDIF


* pseudo-variable ?
* (look for "[" as name end first)
	   brac_pos = INDEX( text(at_start(i):at_end(i)), '[' )
	   IF ( brac_pos .GT. 0 ) THEN
	      aend = at_start(i) + brac_pos - 2
	   ELSE
	      aend = at_end(i)
	   ENDIF
	   imatch = STR_MATCH (	text(at_start(i):aend),
     .				alg_pvar, num_pseudo_vars )
	   IF ( imatch .NE. atom_not_found ) THEN
	      at_type(i) = alg_pseudo_var
	      at_id(i)   = imatch
	      GOTO 100
	   ENDIF

* cant identify it - we will call it an variable for now
* *kob* NAG f90 gave error on statement with three tick marks
#ifdef FORTRAN_90
	   IF (text(at_start(i):at_start(i)) .NE. "'" ) THEN
#else
	   IF (text(at_start(i):at_start(i)) .NE. '\'' ) THEN
#endif
* (7/98) only check name if it is not in quotation marks
	     IF ( .NOT.TM_LEGAL_NAME(text(at_start(i):aend)) ) GOTO 5100
	   ENDIF
	   at_type(i) = alg_variable
	   at_id(i)   = unspecified_int4

 100	CONTINUE

* A function name not followed by a left paren is actually a variable (12/96)
	DO 200  i = 1, natom
	   IF ( at_type(i) .EQ. alg_function
     .	   .OR. at_type(i) .EQ. alg_grid_chg_fcn ) THEN
	      IF (i .EQ. natom) THEN
	          at_type(i) = alg_variable
	          at_id(i)   = unspecified_int4
	      ELSEIF ( at_type(i+1) .NE. alg_punctuation
     .		  .OR. at_id  (i+1) .NE. left_paren ) THEN
	          at_type(i) = alg_variable
	          at_id(i)   = unspecified_int4
	      ENDIF
	   ENDIF
 200	CONTINUE

* 12/96 kludge ...
* In ALG_BREAK_UP the operators EQ, LT, GT, ... are just names so that routine
* misinterprets the negative in  "a EQ -1". Correct that here.
 290	DO 310 i = 3, natom
	   IF ( at_type(i) .EQ. alg_constant ) THEN
	      IF ( at_type(i-1) .EQ. alg_operator
     .	     .AND. at_id  (i-1) .EQ. 2
     .	     .AND. at_type(i-2) .EQ. alg_operator
     .	     .AND. at_end (i-1) .EQ. at_start(i)-1    ) THEN
* ... this minus sign actually belongs to the constant
	         at_start(i) = at_start(i-1)
	         DO 300 i2 = i, natom 	   
	            at_start(i2-1) = at_start(i2)
	            at_end  (i2-1) = at_end  (i2)
	            at_type (i2-1) = at_type (i2)
	            at_id   (i2-1) = at_id   (i2)
 300	         CONTINUE
	         natom = natom - 1
	         GOTO 290
	      ENDIF
	   ENDIF
 310	CONTINUE

* insert flag indicating end of atom list
	at_type(natom+1)  = alg_finished
	at_start(natom+1) = 1
	at_end(natom+1)   = 1

* successful completion
	status = ferr_ok
	RETURN

* error exits
 5000	RETURN
 5100	risc_buff = text(at_start(i):aend)
	CALL ERRMSG( ferr_syntax, status,
     .		     'illegal name: '//risc_buff(:at_start(i)-aend+1), *5000 )
	END
