	BLOCK DATA CALENDAR_BLKDAT

C 	calendar_blkdat.F
C
C	Calendar-specific initialization used in multiple calendar support
C	J Davison 10.99
C	V 1.0
C   12/00 *acm* change the calendar names to match those in the NetCDF standard
C         (plus two of our own, for testing)
C        - also don't initialize "the calendar" -- each time axis may have
C          its own calendar ID; calendar attributes are set for a given
C          calendar as needed.
C

C   5/1/2001  *acm* shorten data statement for cals_month_by_day for some compilers
C  10/1/2001  *acm* change name of 360-day calendar to 360_DAY
*      *acm* 10/01  remove VMS includes
C   v554 2/3/2004  *acm* add an index to identify calendar type of d360
C   v570 5/ 2004   *acm* new calendar all_leap=366_day

        include 'calendar.decl'
        include 'calendar.cmn'

C	*******************************************************************
C       Calendar parameters needed by calendar.cmn
C	*******************************************************************

	DATA max_calendars/mxcals/
        DATA max_months/mxmnths/
        DATA max_days/mxdays/


C	*******************************************************************
C       Supported calendars are those below --
C	*******************************************************************
C	Calendar name as recognized by Ferret (see tm_set_current_calendar,
C       tm_get_calendar_id).   

	data	allowed_calendars
     .  /
     1	'GREGORIAN',
     2  'NOLEAP',
     3  'JULIAN',
     4  '360_DAY',
     5  'ALL_LEAP'
     .  /

C	Calendar index corresponding to calendar names:
	data    gregorian   /1/
        data    noleap      /2/
        data    julian      /3/
        data    d360        /4/
        data    all_leap    /5/

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
     5	12    
     .  /

C	*******************************************************************
C	Each row identifies the number of days in the calendar year:
C	*******************************************************************
	data	cals_num_days
     .	/ 
     1	365,
     2  365,  
     5  365,  
     4  360,  
     5  366   
     .  /

C	*******************************************************************
C	Each row identifies the exact number of days in the calendar year:
C	*******************************************************************
	data	cals_yeardays
     .	/ 
     1	365.2425,
     2  365.,  
     3  365.25,  
     4  360. ,  
     5  366. 
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

     5	'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
     .    'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'

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
     3  31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,
     4  12*30,
     5  31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31
     .  /


C	*******************************************************************
C	Each row identifies the number of days before each month for a
C	single calendar.  Row must have 'max_months' values.  Fill with 
C	zeros if necessary.
C       See output_cals_by.f  to generate this from cals_days_in_month.
C	*******************************************************************
	data	cals_days_before_month
     .	/   
     1   0,  31,  59,  90, 120, 151, 181, 212, 243, 273, 304, 334,
     2   0,  31,  59,  90, 120, 151, 181, 212, 243, 273, 304, 334,
     3   0,  31,  59,  90, 120, 151, 181, 212, 243, 273, 304, 334,
     4   0,  30,  60,  90, 120, 150, 180, 210, 240, 270, 300, 330,
     5   0,  31,  60,  91, 121, 152, 182, 213, 244, 274, 305, 335
     .  /


C	*******************************************************************
C	Each block identifies the month in which the given day appears for
C       a single calendar.  Each must have 'max_days' values.  Fill with 
C	zeros if necessary.  
C       See output_cals_by.f  to generate this from cals_days_in_month
C	*******************************************************************
	data	cals_month_by_day
     .	/   
C Calendar 1, GREGORIAN
     .   31*1,  28*2,  31*3, 30*4, 31*5, 30*6, 31*7, 31*8, 30*9,
     .   31*10, 30*11, 31*12, 0,
C Calendar 2, NOLEAP
     .   31*1,  28*2,  31*3, 30*4, 31*5, 30*6, 31*7, 31*8, 30*9,
     .   31*10, 30*11, 31*12, 0,
C Calendar 3, JULIAN
     .   31*1,  28*2,  31*3, 30*4, 31*5, 30*6, 31*7, 31*8, 30*9,
     .   31*10, 30*11, 31*12, 0,
C Calendar 4, 360_DAY
     .   30*1, 30*2, 30*3, 30*4, 30*5, 30*6, 30*7, 30*8, 30*9,
     .   30*10, 30*11, 30*12, 6*0,
C Calendar 5, ALL_LEAP
     .   31*1,  29*2,  31*3, 30*4, 31*5, 30*6, 31*7, 31*8, 30*9,
     .   31*10, 30*11, 31*12/

      end
