/*
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*
*  Access and use of this software shall impose the following
*  obligations and understandings on the user. The user is granted the
*  right, without any fee or cost, to use, copy, modify, alter, enhance
*  and distribute this software, and any derivative works thereof, and
*  its supporting documentation for any purpose whatsoever, provided
*  that this entire notice appears in all copies of the software,
*  derivative works and supporting documentation.  Further, the user
*  agrees to credit NOAA/PMEL/TMAP in any publications that result from
*  the use of this software or in any product that includes this
*  software. The names TMAP, NOAA and/or PMEL, however, may not be used
*  in any advertising or publicity to endorse or promote any products
*  or commercial entity unless specific written permission is obtained
*  from NOAA/PMEL/TMAP. The user also understands that NOAA/PMEL/TMAP
*  is not obligated to provide the user with any support, consulting,
*  training or assistance of any kind with regard to the use, operation
*  and performance of this software nor to provide the user with any
*  updates, revisions, new versions or "bug fixes".
*
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP "AS IS" AND ANY EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE.  
*
*/




/*******************************************************************************
	Utility.c

       Associated Header file: Utility.h
*******************************************************************************/

#include <stdio.h>
#include <Xm/Xm.h>
#include <Xm/MwmUtil.h>
#include <Xm/MenuShell.h>
#include "UxXt.h"

#include <X11/Shell.h>

/*******************************************************************************
       Includes, Defines, and Global variables from the Declarations Editor:
*******************************************************************************/

#include <Xm/CascadeB.h>
#include "ferret_structures.h"
#include "ferret.h"
#include "ferret_shared_buffer.h"

typedef struct _menu_item {
    char        label[64];         /* the label for the item */
    WidgetClass *class;         /* pushbutton, label, separator... */
    void       (*callback)();   /* routine to call; NULL if none */
    char    callback_data[32]; /* client_data for callback() */
    struct _menu_item *subitems; /* pullright menu items, if not NULL */
} MenuItem;

extern int ferret_query(int query, smPtr sBuffer, char *tag,
		 char *arg1, char *arg2, char *arg3, char *arg4 );


char *CollectToReturn(char *targetStr, char *subStr);
char *CollectDateItem(char *targetStr, char *subStr);
void AllCaps(Widget wid, XtPointer client_data,
	       XtPointer cbs);
char *FormatFloatStr(double inNum);
double Mod(double x,double  y);
double Ceiling(double x);
double Floor(double x);
double AbsVal(double x);
double RoundItOff(double x);
int PadOrTrunc(char *inString, int numChars);
void TimeToFancyDate(double *val, char *outDate);
double XtoLon(int x);
double YtoLat(int y);
int LonToX(double inLon, int lOrR);
int LatToY(double inLat);
char *MmmToMonth(int mmm);
int LeapYear(int year);
double DateToSecs(char *inDate, int hasYear);
int FixDate(char *inDate);
int DaysUptoIntMonth(int *inMonth, int *inYear);
int DaysUptoMonth(char *inMonth, int inYear);
void my_secs_to_date(double *val, char *outDate);
int JulianToMonth(int inDay);
int ConvertMonth(char *inMonth);

extern enum {tAxisIsCalendar, tAxisIsDerivedCalendar, tAxisIsClimatology, tAxisIsRaw, tAxisIsIndex} tAxisState;

#ifndef FULL_GUI_VERSION
#ifdef NO_ENTRY_NAME_UNDERSCORES
double tm_secs_from_bc(int *gStartYear, int *month, int *day, int *hour, int *minute, int *second);
#else
double tm_secs_from_bc_(int *gStartYear, int *month, int *day, int *hour, int *minute, int *second);
#endif
#endif


static	swidget	UxParent;

#define CONTEXT_MACRO_ACCESS 1
#include "Utility.h"
#undef CONTEXT_MACRO_ACCESS

Widget	Utility;

/*******************************************************************************
Auxiliary code from the Declarations Editor:
*******************************************************************************/

#define NUMLATDEGS 180.0
#define NUMLONDEGS 720.0
#define NUMLATPIXELS 216.0
#define NUMLONPIXELS 627.0
#define LEFT 1
#define RIGHT 2
#define LL 1
#define Cl 2
#define LR 3
#define CR 4
#define UR 5
#define CU 6
#define UL 7
#define CL 8
#define CC 9
#define MINLTMAX
#define SECSINDAY 86400

void my_secs_to_date(val, outDate)
double *val;
char *outDate;
{
	int yr, mm, dd, hh, mmm, ss;
	double myVal = *val;

	/* subtract out the number of years */
	yr = myVal/(365 * SECSINDAY);
	myVal -= yr * 365 * SECSINDAY;

	/* subtract out julian days in year */
	dd = myVal/SECSINDAY;
	myVal -= dd * SECSINDAY;

	/* get month from julian day */
	mm = JulianToMonth(dd);

	/* get the true day of the month */
	dd = dd - DaysUptoIntMonth(&mm, &yr);

	/* what's left in val is a fraction of a day */
	hh = myVal/3600;
	myVal -= myVal * 3600;
	mmm = myVal/60;
	myVal -= myVal * 60;
	ss = myVal;

	/* format the output string */
	sprintf(outDate, "%d-%s-%02d:%02d:%02d:%02d", yr, MmmToMonth(mm), dd, hh, mmm, ss);
}

int JulianToMonth(inDay)
{
	if (inDay <= 31)
		return 1;
	if (inDay <= 59)
		return 2;
	if (inDay <= 90)
		return 3;
	if (inDay <= 120)
		return 4;
	if (inDay <= 151)
		return 5;
	if (inDay <= 181)
		return 6;
	if (inDay <= 212)
		return 7;
	if (inDay <= 243)
		return 8;
	if (inDay <= 273)
		return 9;
	if (inDay <= 304)
		return 10;
	if (inDay <= 334)
		return 11;
	if (inDay <= 365)
		return 12;
}

int DaysUptoIntMonth(inMonth, inYear)
int *inMonth, *inYear;
{
	int days;

	if (*inMonth == 1)
		return 0;

	if (*inMonth == 2)
		return 31;

	if (*inMonth == 3)
		days = 31 + 28;

	if (*inMonth == 4)
		days = 31 + 28 + 31;

	if (*inMonth == 5)
		days = 31 + 28 + 31 + 30;

	if (*inMonth == 6)
		days = 31 + 28 + 31 + 30 + 31;

	if (*inMonth == 7)
		days = 31 + 28 + 31 + 30 + 31 + 30;

	if (*inMonth == 8)
		days = 31 + 28 + 31 + 30 + 31 + 30 + 31;

	if (*inMonth == 9)
		days = 31 + 28 + 31 + 30 + 31 + 30 + 31 + 31;

	if (*inMonth == 10)
		days = 31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30;

	if (*inMonth == 11)
		days = 31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31;

	if (*inMonth == 12)
		days = 31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31 + 30;

	if (LeapYear(*inYear))
		days++;
	return days;
}

#ifndef FULL_GUI_VERSION

#ifdef NO_ENTRY_NAME_UNDERSCORES
double tm_secs_from_bc(year, month, day, hour, minute, second)
#else
double tm_secs_from_bc_(year, month, day, hour, minute, second)
#endif
int *year, *month, *day, *hour, *minute, *second;
{
	double refSecs=0;

	/* days in this month */
	refSecs += (double)*day * SECSINDAY;

	/* months and year */
	refSecs += (double)DaysUptoIntMonth(month, year) * SECSINDAY;
	refSecs += (double)*year * 365 * SECSINDAY;

	/* now add in number of leap years befor year */
	refSecs += (*year - 1)/4 * SECSINDAY;
	
	/* hours */
	refSecs += (double)*hour * 3600;

	/* minutes */
	refSecs += (double)*minute * 60;

	/* seconds */
	refSecs += (double)*second;
}
#endif

double DateToSecs(inDate, hasYear)
char *inDate;
int hasYear;
{
	/* accepts a syntactically correct date (long or short form and return
	   number of secs since year 0000 */
	char *tDate=(char *)XtMalloc(32), datePart[32], mText[32];
	int dd=0, yy=0, mmm=0, hh=0, mm=0, ss=0;
	double refSecs=0;

	strcpy(tDate, inDate);

	/* day */
	tDate = CollectDateItem(tDate, datePart);
	sscanf(datePart, "%d", &dd);

	/* month */
	tDate = CollectDateItem(tDate, mText);
	mmm = ConvertMonth(mText);

	if (!mmm) {
		/* an error occurred */
		XtFree(tDate);	
		return INTERNAL_ERROR;
	}

	/* year */
	if (hasYear) {
		tDate = CollectDateItem(tDate, datePart);
		sscanf(datePart, "%d", &yy);
	}

	/* hours */
	tDate = CollectDateItem(tDate, datePart);
	sscanf(datePart, "%d", &hh);

	/* minutes */
	tDate = CollectDateItem(tDate, datePart);
	sscanf(datePart, "%d", &mm);

	/* seconds */
	tDate = CollectDateItem(tDate, datePart);
	sscanf(datePart, "%d", &ss);
	
#ifdef NO_ENTRY_NAME_UNDERSCORES
	refSecs = tm_secs_from_bc(&yy, &mmm, &dd, &hh, &mm, &ss);
#else
	refSecs = tm_secs_from_bc_(&yy, &mmm, &dd, &hh, &mm, &ss);
#endif
	XtFree(tDate);
	return refSecs;
}

int LeapYear(year)
int year;
{
	if ((year - 1900) & 4)
		return 0;
	else
		return 1;
}
 
int DaysUptoMonth(inMonth, inYear)
char *inMonth;
int inYear;
{
	int days;

	if (strstr(inMonth, "JAN"))
		return 0;

	if (strstr(inMonth, "FEB"))
		return 31;

	if (strstr(inMonth, "MAR"))
		days = 31 + 28;

	if (strstr(inMonth, "APR"))
		days = 31 + 28 + 31;

	if (strstr(inMonth, "MAY"))
		days = 31 + 28 + 31 + 30;

	if (strstr(inMonth, "JUN"))
		days = 31 + 28 + 31 + 30 + 31;

	if (strstr(inMonth, "JUL"))
		days = 31 + 28 + 31 + 30 + 31 + 30;

	if (strstr(inMonth, "AUG"))
		days = 31 + 28 + 31 + 30 + 31 + 30 + 31;

	if (strstr(inMonth, "SEP"))
		days = 31 + 28 + 31 + 30 + 31 + 30 + 31 + 31;

	if (strstr(inMonth, "OCT"))
		days = 31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30;

	if (strstr(inMonth, "NOV"))
		days = 31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31;

	if (strstr(inMonth, "DEC"))
		days = 31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31 + 30;

	if (LeapYear(inYear))
		days++;
	return days;
}

int ConvertMonth(inMonth)
char *inMonth;
{
	if (strstr(inMonth, "JAN"))
		return 1;

	if (strstr(inMonth, "FEB"))
		return 2;

	if (strstr(inMonth, "MAR"))
		return 3;

	if (strstr(inMonth, "APR"))
		return 4;

	if (strstr(inMonth, "MAY"))
		return 5;

	if (strstr(inMonth, "JUN"))
		return 6;

	if (strstr(inMonth, "JUL"))
		return 7;

	if (strstr(inMonth, "AUG"))
		return 8;

	if (strstr(inMonth, "SEP"))
		return 9;

	if (strstr(inMonth, "OCT"))
		return 10;

	if (strstr(inMonth, "NOV"))
		return 11;

	if (strstr(inMonth, "DEC"))
		return 12;

	return 0;
}

char *MmmToMonth(mmm)
int mmm;
{
	char tText[4];

	switch (mmm) {
		case 1:	strcpy(tText, "JAN");
			break;
		case 2:	strcpy(tText, "FEB");
			break;
		case 3:	strcpy(tText, "MAR");
			break;
		case 4:	strcpy(tText, "APR");
			break;
		case 5:	strcpy(tText, "MAY");
			break;
		case 6:	strcpy(tText, "JUN");
			break;
		case 7:	strcpy(tText, "JUL");
			break;
		case 8:	strcpy(tText, "AUG");
			break;
		case 9:	strcpy(tText, "SEP");
			break;
		case 10:strcpy(tText, "OCT");
			break;
		case 11:strcpy(tText, "NOV");
			break;
		case 12:strcpy(tText, "DEC");
			break;
	}
	return tText;
}

int FixDate(inDate)
char *inDate;
{
	char *tDate, datePart[32], tText[32], tText2[32], retDate[32], retTime[32];
	int dd, yy, mmm, hh, mm, ss;

	tDate = (char *)&inDate[0];
	strcpy(retDate, "");

	/* climatology dates have blank years--test for this and remove extra blanks */
	if (strstr(inDate, "  ")) {
		/* date has extra blanks */
		sscanf(inDate, "%s%s", tText, tText2);
		strcpy(tDate, "");
		sprintf(tDate, "%s-0000 %s", tText, tText2);
	}

	/* day */
	tDate = CollectDateItem(tDate, datePart);
	if (strlen(datePart) < 2) {
		/* reformat day */
		sscanf(datePart, "%d", &dd);
		sprintf(tText, "%02d-", dd);
		strcat(retDate, tText);
	}
	else {
		strcat(retDate, datePart);
		strcat(retDate, "-");
	}

	/* month */
	tDate = CollectDateItem(tDate, datePart);
	if (strlen(datePart) < 3) {
		/* see if this is a numeric month */
		sscanf(datePart, "%d", &mmm);
		if (mmm > 0 & mmm <= 12) {
			/* code date as abbreviated date */
			strcat(retDate, MmmToMonth(mmm));
			strcat(retDate, "-");
		}
		else {
			/* error */
			return INTERNAL_ERROR;
		}
	}
	else {
		strcat(retDate, datePart);
		strcat(retDate, "-");
	}

	/* year */
	tDate = CollectDateItem(tDate, datePart);
	sscanf(datePart, "%d", &yy);
	if (strlen(datePart) < 4) {
		/* numeric year? */
		if (yy < 99 && yy >= 0) {
			yy = 1900 + yy;
			sprintf(tText, "%4d", yy);
		}
		else {
			/* error */
			XtFree(tDate);
			return INTERNAL_ERROR;
		}
	}
	else {
		if (yy != 0) {
			strcat(retDate, datePart);
		}
		else {
			/* this is a year 0000 climatology--don't need the year or time */
			retDate[strlen(retDate)-1] = '\0';
			strcpy(inDate, retDate);
		}
	}

	/* now do the time part */
	if (!tDate) {
		/* no time component */
		strcpy(inDate, retDate);
		return;
	}

	strcpy(retTime, ":");

	/* hour */
	tDate = CollectDateItem(tDate, datePart);
	if (strlen(datePart) == 0) {
		/* no hour component--ignore time part */
		strcpy(inDate, retDate);
		return;
	}
	else if (strlen(datePart) < 2) {
		/* reformat hour */
		sscanf(datePart, "%d", &hh);
		sprintf(tText, "%02d:", hh);
		strcat(retTime, tText);
	}
	else {
		strcat(retTime, datePart);
		strcat(retTime, ":");
	}

	/* minute */
	tDate = CollectDateItem(tDate, datePart);
	if (strlen(datePart) == 0) {
		/* no minute component--ignore time part */
		strcpy(inDate, retDate);
		return;
	}
	else if (strlen(datePart) < 2) {
		/* reformat min */
		sscanf(datePart, "%d", &mm);
		sprintf(tText, "%02d:", mm);
		strcat(retTime, tText);
	}
	else {
		strcat(retTime, datePart);
		strcat(retTime, ":");
	}

	/* seconds */
	tDate = CollectDateItem(tDate, datePart);
	if (strlen(datePart) == 0) {
		/* no minute component--ignore time part */
		strcpy(inDate, retDate);
		return;
	}
	else if (strlen(datePart) < 2) {
		/* reformat secs */
		sscanf(datePart, "%d", &ss);
		sprintf(tText, "%02d", ss);
		strcat(retTime, tText);
	}
	else 
		strcat(retTime, datePart);

	/* if we get here we made a valid date and time string--put together before exiting */
	strcpy(inDate, retDate);
	strcat(inDate, retTime);
	return;	
}

char *CollectDateItem(targetStr, subStr)
char *targetStr, *subStr;
{
	while ((*targetStr != '-') && (*targetStr != ':') && (*targetStr != '/') && (*targetStr != ' ') && (*targetStr != 0))
		*subStr++ = *targetStr++;
	*subStr++ = '\0';
	if (*targetStr != 0)
		return(++targetStr);
	else
		return(targetStr);
}

void AllCaps(wid, clientData, callData)
Widget wid;
XtPointer clientData, callData;
{
	int len;
	XmTextVerifyCallbackStruct *cbs = (XmTextVerifyCallbackStruct *)callData;

	for (len=0; len<cbs->text->length; len++)
		if (islower(cbs->text->ptr[len]))
			cbs->text->ptr[len] = toupper(cbs->text->ptr[len]);
}

char *CollectToReturn(targetStr, subStr)
char *targetStr, *subStr;
{
	while ((*targetStr != '\n') && (*targetStr != '\r') && (*targetStr != 0))
		*subStr++ = *targetStr++;
	*subStr++ = '\0';
	if (*targetStr != 0)
		return(++targetStr);
	else
		return(targetStr);
}

/* from Cx interface */

void TimeToFancyDate(val, date)
double *val;
char *date;
{
	char fDate[80], str1[20], str2[20];

#ifdef FULL_GUI_VERSION
	/* get the "fortran date" */
#ifdef NO_ENTRY_NAME_UNDERSCORES
	secs_to_date_c(val, fDate);
#else
	secs_to_date_c_(val, fDate);
#endif
#else
	my_secs_to_date(val, fDate);
#endif
	FixDate(fDate);
	strcpy(date, fDate);
	return;
}

int PadOrTrunc(inStr, numChars)
char *inStr;
int numChars;
{
 	register int i, len;

	len = strlen(inStr);
	if (len > numChars) {
		/* truncate */
 		inStr[numChars-3] = '.';
		inStr[numChars-2] = '.';
		inStr[numChars-1] = '.';
		inStr[numChars] = '\0';
	}
	else if (len < numChars) {
		/* pad */
 		for (i=len; i<numChars-1; i++)
			inStr[i] = ' ';
		inStr[numChars-1] = '\0';
	}
}

double XtoLon(x)
int x;
{
	double val;

	val = (float)x * (NUMLONDEGS/NUMLONPIXELS);
	return val;
}

double YtoLat(y)
int y;
{
	double val;

	val = -(float)y * (NUMLATDEGS/NUMLATPIXELS) + 90.0;
	return val;
}

int LonToX(inLon, lOrR)
double inLon;
int lOrR;
{
	/*if (inLon >= 0 && inLon <= 20 && lOrR == RIGHT)
		return (int)((inLon+360) * NUMLONPIXELS/NUMLONDEGS);
	else if (inLon > 360 && inLon <= 380 && lOrR == RIGHT)
		return (int)((inLon) * NUMLONPIXELS/NUMLONDEGS);
	else */if (inLon >= 0/* && inLon <= 720*/)
		return (int)(inLon * NUMLONPIXELS/NUMLONDEGS);
	else if (inLon < 0)
		return (int)((inLon+360) * NUMLONPIXELS/NUMLONDEGS);
}

int LatToY(inLat)
double inLat;
{
	int val;

	val = -((inLat-90) * NUMLATPIXELS/NUMLATDEGS);
	return val;
}

char *FormatFloatStr(inNum)
double inNum;
{
	char tText[64];
	register int i;

	sprintf(tText, "%.2f", inNum);
	return tText;
/*
	i = strlen(tText) - 1;
	while (i >= 0) {
		if (tText[i] == '0')
			tText[i] = '\0';
		else if (tText[i] == '.') {
			tText[i] = '\0';
			break;
		}
		else
			break;
		i--;
	}
	return tText;
*/
}

double AbsVal(x)
double x;
{
	if (x < 0)
		return -x;
	else
		return x;
}

double Floor(x)
double x;
{
	int anInt;

	anInt = (int)x;
	return (double)anInt;
}

double Ceiling(x)
double x;
{
	int anInt;

	anInt = (int)x;
	if (x - anInt == 0) return
		(double)anInt;
	else
		return (double)anInt + 1;
}

double Mod(x, y)
double x, y;
{
	if (x == 0 || y == 0) return x;

	return x - y * Floor(x/y);
}

double RoundItOff(x)
double x;
{
	int ix;

	ix = (int)x;
	if (x - (double)ix > 0.5)
		return (double)ix + 1;
	else
		return (double)ix;
}

/*******************************************************************************
       The following are callback functions.
*******************************************************************************/

/*******************************************************************************
       The 'build_' function creates all the widgets
       using the resource values specified in the Property Editor.
*******************************************************************************/

static Widget	_Uxbuild_Utility()
{
	Widget		_UxParent;


	/* Creation of Utility */
	_UxParent = UxParent;
	if ( _UxParent == NULL )
	{
		_UxParent = UxTopLevel;
	}

	Utility = XtVaCreatePopupShell( "Utility",
			topLevelShellWidgetClass,
			_UxParent,
			XmNwidth, 269,
			XmNheight, 469,
			XmNx, 504,
			XmNy, 357,
			NULL );



	return ( Utility );
}

/*******************************************************************************
       The following is the 'Interface function' which is the
       external entry point for creating this interface.
       This function should be called from your application or from
       a callback function.
*******************************************************************************/

Widget	create_Utility( swidget _UxUxParent )
{
	Widget                  rtrn;

	UxParent = _UxUxParent;

	rtrn = _Uxbuild_Utility();

	return(rtrn);
}

/*******************************************************************************
       END OF FILE
*******************************************************************************/

