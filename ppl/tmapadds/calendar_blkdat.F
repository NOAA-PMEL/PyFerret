	BLOCK DATA CALENDAR_BLKDAT

C 	calendar_blkdat.F
C
C	Calendar-specific initialization used in multiple calendar support
C	J Davison 10.99
C	V 1.0
C
C       0.00000000000000000000000000000000000000000000000000000000000000001

#ifdef unix
        include 'calendar.cmn'
#else
        INCLUDE 'TMAP_PPLV11INC:CALENDAR.CMN'
#endif


C	*******************************************************************
C       Supported calendars are those below --
C	*******************************************************************
C	Calendar name as recognized in the PPL+ CALENDAR command:
	data	calendar_names
     .  /
     1	'GREGORIAN',
     2  'COMMON',
     3  'LEAP_YR',
     4  'DAYS_360',
     5	'TMAP_360',
     6  'PECULIAR'
     .  /

C	Calendar index corresponding to calendar names:
	data    gregorian   /1/
        data    common      /2/
        data    leap_yr     /3/
        data    days_360    /4/
        data    tmap_360    /5/
        data    peculiar    /6/

C	*******************************************************************
C	Array values are ordered by calendar indices.  Each row specifies
C	values for a single calendar.  The first row assigns values for
C	calendar 1 (Gregorian), the second for calendar 2 (Common), and
C	so on as declared in the index integers immediately above.

C	Here, each row identifies the number of months in the calendar:
C	*******************************************************************
	data    cals_num_months 
     .	/ 
     1	12,
     2	12,
     3	12,
     4	12,
     5	12,
     6  07    
     .  /

C	*******************************************************************
C	Each row identifies the number of days in the calendar year:
C	*******************************************************************
	data	cals_num_days
     .	/ 
     1	365,
     2  365,  
     5  366,  
     4  360, 
     5  360, 
     6  210    
     .  /

C	*******************************************************************
C	Each row names the months for one calendar.  Row must have 
C	'max_months' values.  Fill with blanks if necessary.
C	*******************************************************************
	data	cals_month_names 
     .	/ 
     1	'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
     .    'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC',

     2	'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
     .    'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC',

     3	'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
     .    'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC',

     4	'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
     .    'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC',

     5	'JER', 'EDH', 'STV', 'KEV', 'JON', 'JOE',
     .    'ANS', 'SIM', 'GAB', 'VAL', 'DEN', 'EDI',

     6	'ALF', 'BET', 'GAM', 'DEL', 'EPS', 'ZET',
     .    'ETA', ' ', ' ', ' ', ' ', ' '
     .  /

C	*******************************************************************
C	Each row identifies the number of days in each month for a
C	single calendar.  Row must have 'max_months' values.  Fill with 
C	zeros if necessary.
C	*******************************************************************
	data	cals_days_in_month
     .	/   
     1	31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,
     2  31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,
     3  31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,
     4  12*30,
     5  12*30,
     6	30, 45, 15, 30, 50, 10, 30, 0, 0, 0, 0, 0
     .  /


C**************************************************************************
C**************************************************************************
C   --> DO NOT CHANGE INITIALIZATIONS BELOW UNLESS GREGORIAN IS NOT DEFAULT
C**************************************************************************
C**************************************************************************

C	*******************************************************************
C       Default calendar (ie calendar 1, Gregorian) data initialization
C	*******************************************************************
	data    cal_name        /'GREGORIAN'/
	data    cal_id          /1/

	data    loaded_calendar /1/
	data    num_months      /12/
	data    num_days        /365/

	data    month_names
     .  /
     1	'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
     .    'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'
     .  /

	data    days_in_month
     .  /
     1	31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31
     .  /
     
C	*******************************************************************
C       These variables are computed for each calendar in tm_load_calendar()
C         but are initialized here for the default calendar (Gregorian).
C	*******************************************************************
	data    days_before_month
     .  /
     1	0,  31,  59,  90, 120, 151, 181, 212, 243, 273, 304, 334
     .  /

	data    month_by_day
     .  /
     1	31*01, 28*02, 31*03, 30*04, 31*05, 30*06,
     .    31*07, 31*08, 30*09, 31*10, 30*11, 31*12, 0
     .  /    

	end



