	SUBROUTINE CD_STAMP_OUT(append, cdfid, string, status)

* Put a "history" stamp into the current netCDF output file

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 3/92
* 10/96 *kob* - Linux port - had to move variable assignments off of
*		declaration line.

* argument definitions:
*       append  - logical control to create or append
*       cdfid   - returned pointer to open CDF file
*       string  - history string to add
*	status	- return status

* argument declarations:
        LOGICAL  append
	INTEGER	 cdfid, status
        CHARACTER*(*) string

* include files
#ifdef unix
        include 'tmap_format/netcdf.decl'
        include 'tmap_format/netcdf.inc'
        include 'tmap_format/tmap_errors.parm'
        include 'tmap_format/cd_lib.parm'
#else
        INCLUDE 'TMAP_FORMAT:NETCDF.DECL'
        INCLUDE 'TMAP_FORMAT:NETCDF.INC'
        INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
        INCLUDE 'TMAP_FORMAT:CD_LIB.PARM'
#endif

* local variable declarations:
* *kob* explicit definitions not allowed on decl line in linux...
*       moved it to parameter line 10/96
      INTEGER TM_LENSTR1, hlen, slen
      LOGICAL CD_GET_ATTRIB, got_it,
     .        do_warn
      PARAMETER (do_warn = .TRUE. )
      CHARACTER*132 history

* initialize
      slen = MIN( TM_LENSTR1(string), 120 )

* update (or create) history record
      got_it = CD_GET_ATTRIB( cdfid, NCGLOBAL, 'history',
     .                     .NOT.do_warn, ' ', history, 132 )
      hlen = TM_LENSTR1( history )

* is the last history stamp the same as this ?
      IF ( hlen .GE. slen ) THEN
         IF ( history(hlen-slen+1:hlen) .EQ. string(:slen) ) RETURN
      ENDIF

* insert new signature
      CALL CD_WRITE_ATTRIB(cdfid, pcdf_global, 'history', string(:slen),
     .                     append, status )
      IF ( status .NE. merr_ok ) GOTO 5000

* successful completion
      status = merr_ok
*      RETURN

* error exit
 5000 RETURN
	END
