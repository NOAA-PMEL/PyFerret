	SUBROUTINE PROCLAIM ( logical_unit, left_margin )

* introduce the program

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 4/3/86
* V400 - 3/95 - changed "Revision" to "Version"
* V402 - 8/95 - use XREVISION_TYPE to automate version info

#ifdef unix
	include 'ferret_cmn/xrevision.cmn'
	include 'ferret_cmn/xrevision_type.cmn'
	include 'ferret_cmn/xtoday.cmn'
	include 'ferret_cmn/xmake_date.cmn'
#else
	INCLUDE 'FERRET_CMN:XREVISION.CMN'
	INCLUDE 'FERRET_CMN:XREVISION_TYPE.CMN'
	INCLUDE 'FERRET_CMN:XTODAY.CMN'
	INCLUDE 'FERRET_CMN:XMAKE_DATE.CMN'
#endif

* calling argument declarations:
	INTEGER		logical_unit
	CHARACTER*(*)	left_margin

* local variable definitions
	INTEGER TM_LENSTR1

* get revision type info (8/95 - kind of a hack)
	progname_mod = revision_type
	len_progname_mod = TM_LENSTR1(progname_mod)

	WRITE  ( logical_unit, 3000)
     .				left_margin,
     .				left_margin, program_name(1:len_program_name),
     .					     progname_mod(1:len_progname_mod),	
     .				left_margin, revision_level, make_date,
     .				left_margin, today_date, today_time

 3000	FORMAT ( 1X,A,'NOAA/PMEL TMAP'/
     .		 1X,A,'Program ',A,1X,A/
     .		 1X,A,'Version',F5.2,' - ',A/
     .		 1X,2A,1X,A/ )

	RETURN
	END
