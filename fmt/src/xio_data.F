	BLOCK DATA XIO_DATA

* initialize data for indicated COMMON block
* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.00 - 10/08/85

* COMMON/XIO.....logical unit numbers for I/O
#ifdef unix
	include 'tmap_format/xio.cmn_text'
#else
	INCLUDE 'TMAP_FORMAT:XIO.CMN_TEXT'
#endif

	DATA	tohdlu		/20/,
     .		fmhdlu		/21/,
     .		fmhd2lu		/22/,
     .		tooplu		/ 6/,
     .		fmoplu		/ 5/,
     .		lunit_errors	/ 6/

	END
