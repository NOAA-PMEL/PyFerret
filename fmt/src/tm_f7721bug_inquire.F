	LOGICAL FUNCTION TM_F7721BUG_INQUIRE( fpath )

* identical function to the INQUIRE EXIST=... statement from FORTRAN
* this routine is provided only to compensate for a bug in f772.1

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* revision 0.0 - 10/21/92 - used only on DEC

* calling argument declarations
	CHARACTER*(*)	fpath

* local variable declarations
	INTEGER	lunit

* get free logical unit #
        CALL LIB_GET_LUN (lunit)
                                 
* open the file as a test for its existance
* any error will be interpreted as non-existance
        OPEN ( UNIT = lunit,
     .         STATUS = 'OLD',
     .         FILE = fpath,
     .         ERR = 5000)

* we found it
        CLOSE( UNIT=lunit )
        TM_F7721BUG_INQUIRE = .TRUE.
	RETURN

* file doesn't exist (or other improperly-detected error)
 5000   TM_F7721BUG_INQUIRE = .FALSE.
        RETURN
	END
