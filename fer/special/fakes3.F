* V530 10/00 *sh* - added COMPLEX declarations to force link FORTRAN complex
* variable code for use in external functions
   
	subroutine fakes

#ifdef unix
	include	'errmsg.parm'
#else
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
#endif

	INTEGER status
	COMPLEX a

        entry	odr_op
        entry	odr_quit
        entry	movie_close
        entry	odr_init
        entry	movie_open
	CALL ERRMSG( ferr_not_implemented, status,
     .				'Unix-based laser disk movies', *20 )
 20     write (0,*)  'Use the HDF-based on-screen animations'
        write (0,*)  'or use the VMS version of FERRET'
	a=CMPLX(0.,1.)
	a=AIMAG(a) 
	RETURN

        entry	epic_list
	CALL ERRMSG( ferr_not_implemented, status,
     .				'EPIC output', *30 )
 30     write (0,*)  'EPIC format is not yet available under Unix'
        write (0,*)  'Use the VMS version of FERRET if available'
        RETURN

	entry	 gks_open_ws
	stop	'gks_open_ws not available'
	entry	 is_autogrid
	stop	'is_autogrid not available'
	end





