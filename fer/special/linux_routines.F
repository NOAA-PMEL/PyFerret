* Linux replacement for fdate/date routines.  
*
* kob 3.11.97

	SUBROUTINE FDATE(date)

	CHARACTER*(*) date

	CHARACTER*9 linux_date
	CHARACTER*2 month
	character*3 mmm

	CALL DATE_AND_TIME(linux_date)

	month = linux_date(5:6)
	if (month .eq. '01') then
		mmm = "Jan"
	else if (month .eq. '02') then
		mmm = "Feb"
	else if (month .eq. '03') then
		mmm = "Mar"
	else if (month .eq. '04') then
		mmm = "Apr"
	else if (month .eq. '05') then
		mmm = "May"
	else if (month .eq. '06') then
		mmm = "Jun"
	else if (month .eq. '07') then
		mmm = "Jul"
	else if (month .eq. '08') then
		mmm = "Aug"
	else if (month .eq. '09') then
		mmm = "Sep"
	else if (month .eq. '10') then
		mmm = "Oct"
	else if (month .eq. '11') then
		mmm = "Nov"
	else if (month .eq. '12') then
		mmm = "Dec"
	endif

	date = linux_date(7:8)//'-'//mmm//'-'//linux_date(3:4)

	return
	end

************************TODAY'S TIME******************************

	SUBROUTINE FTIME(time)

	CHARACTER*(*) time

	character*9 linux_date
	character*10 linux_time

	
	call DATE_AND_TIME(linux_date, linux_time)

	time = linux_time(1:2)//':'//linux_time(3:4)//':'//linux_time(5:6)

	return
	end

