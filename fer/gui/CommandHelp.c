
/*******************************************************************************
	CommandHelp.c

       Associated Header file: CommandHelp.h
*******************************************************************************/

#include <stdio.h>
#include <Xm/Xm.h>
#include <Xm/MwmUtil.h>
#include <Xm/MenuShell.h>
#include "UxXt.h"

#include <Xm/PushB.h>
#include <Xm/Text.h>
#include <Xm/Form.h>
#include <Xm/ScrolledW.h>
#include <X11/Shell.h>

/*******************************************************************************
       Includes, Defines, and Global variables from the Declarations Editor:
*******************************************************************************/


/* prototypes */
swidget create_CommandHelp(swidget UxParent);
extern void ferret_command(char *cmdText, int cmdMode);
extern int ferret_send(char *cmdText);

/* globals */
swidget gSavedCommandHelp = NULL;
swidget CommandHelp;

static void InitText(void);


static	Widget	scrolledWindowText3;
static	Widget	form15;
static	Widget	scrolledWindowText4;
static	Widget	scrolledText3;
static	Widget	pushButton3;
static	swidget	UxParent;

#define CONTEXT_MACRO_ACCESS 1
#include "CommandHelp.h"
#undef CONTEXT_MACRO_ACCESS

Widget	CommandHelp;

/*******************************************************************************
Auxiliary code from the Declarations Editor:
*******************************************************************************/

static void InitText()
{
	char cmdText[5000];

	strcpy(cmdText, "Commands in Program FERRET   version 4.0:\n");
 	strcat(cmdText, "SET\n");
  	strcat(cmdText, "SET WINDOW/SIZE/NEW/LOCATION/ASPECT/CLEAR\n");
  	strcat(cmdText, "SET REGION/I/J/K/L/X/Y/Z/T/DX/DY/DZ/DT/DI/DJ/DK/DL\n");
  	strcat(cmdText, "SET VIEWPORT\n");
  	strcat(cmdText, "SET EXPRSION\n");
  	strcat(cmdText, "SET LIST/PRECISON/FILE/FORMAT/APPEND/HEADING\n");
  	strcat(cmdText, "SET DATA/EZ/VARIABLE/TITLE/FORMAT/GRID/SKIP/COLUMNS/SAVE/RESTORE/ORDER\n");
  	strcat(cmdText, "SET MODE/LAST\n");
  	strcat(cmdText, "SET MOVI/FILE/COMPRESS/LASER/START\n");
  	strcat(cmdText, "SET VARIABLE/TITLE/UNIT/GRID/BAD\n");
  	strcat(cmdText, "SET GRID/SAVE/RESTORE\n");
  	strcat(cmdText, "SET AXIS/MODULO\n");
  	strcat(cmdText, "SHOW/ALL\n");
  	strcat(cmdText, "SHOW WINDOW/ALL\n");
  	strcat(cmdText, "SHOW REGION/ALL\n");
  	strcat(cmdText, "SHOW AXIS/ALL\n");
  	strcat(cmdText, "SHOW EXPRSION/ALL\n");
  	strcat(cmdText, "SHOW LIST/ALL\n");
  	strcat(cmdText, "SHOW DATA/ALL/BRIEF/FULL/VARIABLE/FILE\n");
  	strcat(cmdText, "SHOW MODE/ALL\n");
  	strcat(cmdText, "SHOW MOVIE/ALL\n");
  	strcat(cmdText, "SHOW VARIABLE/ALL/DIAG/USER\n");
  	strcat(cmdText, "SHOW COMMANDS/ALL\n");
  	strcat(cmdText, "SHOW MEMORY/ALL/TEMPORY/PERMANT/FREE\n");
  	strcat(cmdText, "SHOW GRID/ALL/I/J/K/L/X/Y/Z/T\n");
  	strcat(cmdText, "SHOW VIEWPORT/ALL\n");
  	strcat(cmdText, "SHOW TRANFORM/ALL\n");
  	strcat(cmdText, "SHOW ALIAS/ALL\n");
  	strcat(cmdText, "SHOW QUERIES/ALL\n");
  	strcat(cmdText, "CANCEL\n");
  	strcat(cmdText, "CANCEL WIND/ALL\n");
  	strcat(cmdText, "CANCEL REGION/ALL/I/J/K/L/X/Y/Z/T\n");
  	strcat(cmdText, "CANCEL MEMORY/ALL/TEMPORY/PERMANT\n");
  	strcat(cmdText, "CANCEL EXPRSION/ALL\n");
  	strcat(cmdText, "CANCEL LIST/ALL/PRECSION/FILE/FORMAT/HEADING/APPEND\n");
  	strcat(cmdText, "CANCEL DATA/ALL/NOERROR\n");
  	strcat(cmdText, "CANCEL MODE\n");
  	strcat(cmdText, "CANCEL MOVIE/ALL\n");
  	strcat(cmdText, "CANCEL VIEWPORT\n");
  	strcat(cmdText, "CANCEL VARIABLE/ALL\n");
  	strcat(cmdText, "CANCEL AXIS/MODULO\n");
  	strcat(cmdText, "CANCEL ALIAS/ALL\n");
  	strcat(cmdText, "CONTOUR/I/J/K/L/X/Y/Z/T/OVERLAY/SET_UP/FRAME/D/TRANPOSE/FILL/LINE/NOLABEL\n");
        strcat(cmdText, "	/LEVELS/KEY/NOKEY/PALETTE/XLIMITS/YLIMITS/TITLE/PEN\n");
  	strcat(cmdText, "LIST/I/J/K/L/X/Y/Z/T/D/HEADING/NOHEAD/SINGLE/FILE/APPEND/ORDER/FORMAT/RIGID\n");
  	strcat(cmdText, "PLOT/I/J/K/L/X/Y/Z/T/OVERLAY/SET_UP/FRAME/D/TRANPOSE/VS/SYMBOL/NOLABEL\n");
        strcat(cmdText, "	/LINE/XLIMITS/YLIMITS/TITLE\n");
  	strcat(cmdText, "GO\n");
  	strcat(cmdText, "HELP\n");
  	strcat(cmdText, "LOAD/TEMPORY/PERMANT/I/J/K/L/X/Y/Z/T/D/NAME\n");
  	strcat(cmdText, "DEFINE\n");
  	strcat(cmdText, "DEFINE REGION/I/J/K/L/X/Y/Z/T/DEFAULT/DX/DY/DZ/DT/DI/DJ/DK/DL\n");
  	strcat(cmdText, "DEFINE GRID/X/Y/Z/T/FILE/LIKE\n");
  	strcat(cmdText, "DEFINE VARIABLE/TITLE/UNITS/QUIET\n");
  	strcat(cmdText, "DEFINE AXIS/X/Y/Z/T/FILE/UNIT/T0/NAME/FROMDATA/DEPTH/MODULO/NPOINTS\n");
  	strcat(cmdText, "DEFINE VIEWPORT/TEXT/XLIMITS/YLIMITS/SIZE/ORIGIN/CLIP\n");
  	strcat(cmdText, "DEFINE ALIAS\n");
  	strcat(cmdText, "EXIT/COMMAND\n");
  	strcat(cmdText, "MESSAGE/CONTINUE/QUIET\n");
  	strcat(cmdText, "VECTOR/I/J/K/L/X/Y/Z/T/OVERLAY/SET_UP/FRAME/D/TRANPOSE/ASPECT/NOLABEL/LENGTH\n");
  	strcat(cmdText, "        /XSKIP/YSKIP/XLIMITS/YLIMITS/TITLE/PEN\n");
  	strcat(cmdText, "PPLUS/RESET\n");
  	strcat(cmdText, "FRAME\n");
  	strcat(cmdText, "REPEAT/I/J/K/L/X/Y/Z/T\n");
  	strcat(cmdText, "STAT/I/J/K/L/X/Y/Z/T/D\n");
  	strcat(cmdText, "SHADE/I/J/K/L/X/Y/Z/T/OVERLAY/SET_UP/FRAME/D/TRANPOSE/LINE/NOLABEL/LEVELS\n");
  	strcat(cmdText, "       /KEY/NOKEY/PALETTE/XLIMITS/YLIMITS/TITLE\n");
  	strcat(cmdText, "SPAWN\n");
  	strcat(cmdText, "USER/OPT1/OPT2/COMMAND/I/J/K/L/X/Y/Z/T/D/FILE/FORMAT\n");
 	strcat(cmdText, "WIRE/I/J/K/L/X/Y/Z/T/OVERLAY/SET_UP/FRAME/D/VIEWPOIN/ZLIMITS/TRANPOSE/NOLABEL\n");
 	strcat(cmdText, "       /ZSCALE/TITLE\n");
  	strcat(cmdText, "QUERY/ALL/FILE/IGNORE\n");
   	strcat(cmdText, " Alias       Command\n");
   	strcat(cmdText, " -----       -------\n");
   	strcat(cmdText, "LET         DEFINE VARIABLE  \n");                                               
   	strcat(cmdText, "FILE        SET DATA/EZ   \n");                                                  
    	strcat(cmdText, "QUIT        EXIT          \n");                                                  
   	strcat(cmdText, "REGION      SET REGION     \n");                                                 
    	strcat(cmdText, "SAY         MESSAGE/CONTINUE \n");                                               
    	strcat(cmdText, "FILL        CONTOUR/FILL   \n");                                                 
    	strcat(cmdText, "ALIAS       DEFINE ALIAS    \n");                                                
    	strcat(cmdText, "UNALIAS     CANCEL ALIAS    \n");                                                
    	strcat(cmdText, "USE         SET DAT/FORM=CDF  \n");                                              
    	strcat(cmdText, "SAVE        LIST/FORMAT=CDF    \n");                                             
    	strcat(cmdText, "PALETTE     PPL SHASET SPECTRUM=  \n");                                          
    	strcat(cmdText, "LABEL       PPL %LABEL        \n");                                              
    	strcat(cmdText, "ANIMATE     SPAWN xds&   \n");
	XmTextSetString(UxGetWidget(scrolledText3), cmdText);
}

/*******************************************************************************
       The following are callback functions.
*******************************************************************************/

static	void	destroyCB_CommandHelp(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	gSavedCommandHelp = NULL;
}

static	void	activateCB_pushButton3(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	extern swidget CommandHelp;
	
	XtPopdown(UxGetWidget(CommandHelp));
	}
}

/*******************************************************************************
       The 'build_' function creates all the widgets
       using the resource values specified in the Property Editor.
*******************************************************************************/

static Widget	_Uxbuild_CommandHelp()
{
	Widget		_UxParent;


	/* Creation of CommandHelp */
	_UxParent = UxParent;
	if ( _UxParent == NULL )
	{
		_UxParent = UxTopLevel;
	}

	CommandHelp = XtVaCreatePopupShell( "CommandHelp",
			topLevelShellWidgetClass,
			_UxParent,
			XmNwidth, 441,
			XmNheight, 409,
			XmNx, 560,
			XmNy, 260,
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNiconName, "Ferret Command Help",
			XmNtitle, "Command Help",
			XmNallowShellResize, TRUE,
			NULL );
	XtAddCallback( CommandHelp, XmNdestroyCallback,
		(XtCallbackProc) destroyCB_CommandHelp,
		(XtPointer) NULL );



	/* Creation of scrolledWindowText3 */
	scrolledWindowText3 = XtVaCreateManagedWidget( "scrolledWindowText3",
			xmScrolledWindowWidgetClass,
			CommandHelp,
			XmNscrollingPolicy, XmAPPLICATION_DEFINED,
			XmNvisualPolicy, XmVARIABLE,
			XmNscrollBarDisplayPolicy, XmSTATIC,
			XmNx, 23,
			XmNy, 11,
			XmNunitType, XmPIXELS,
			NULL );


	/* Creation of form15 */
	form15 = XtVaCreateManagedWidget( "form15",
			xmFormWidgetClass,
			scrolledWindowText3,
			XmNwidth, 305,
			XmNheight, 306,
			XmNresizePolicy, XmRESIZE_NONE,
			XmNx, 34,
			XmNy, 37,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );


	/* Creation of scrolledWindowText4 */
	scrolledWindowText4 = XtVaCreateManagedWidget( "scrolledWindowText4",
			xmScrolledWindowWidgetClass,
			form15,
			XmNscrollingPolicy, XmAPPLICATION_DEFINED,
			XmNvisualPolicy, XmVARIABLE,
			XmNscrollBarDisplayPolicy, XmSTATIC,
			XmNx, 24,
			XmNy, 17,
			XmNtopOffset, 10,
			XmNtopAttachment, XmATTACH_FORM,
			XmNleftOffset, 10,
			XmNleftAttachment, XmATTACH_FORM,
			XmNrightOffset, 10,
			XmNrightAttachment, XmATTACH_FORM,
			XmNbottomOffset, 50,
			XmNbottomAttachment, XmATTACH_FORM,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );


	/* Creation of scrolledText3 */
	scrolledText3 = XtVaCreateManagedWidget( "scrolledText3",
			xmTextWidgetClass,
			scrolledWindowText4,
			XmNwidth, 382,
			XmNheight, 334,
			XmNcursorPositionVisible, FALSE,
			XmNeditMode, XmMULTI_LINE_EDIT ,
			XmNeditable, FALSE,
			XmNfontList, UxConvertFontList( "-adobe-courier-bold-r-normal--12-120-75-75-m-70-iso8859-1" ),
			RES_CONVERT( XmNhighlightColor, "gray80" ),
			XmNrows, 100,
			RES_CONVERT( XmNbackground, "gray75" ),
			RES_CONVERT( XmNforeground, "black" ),
			XmNvalue, "",
			NULL );


	/* Creation of pushButton3 */
	pushButton3 = XtVaCreateManagedWidget( "pushButton3",
			xmPushButtonWidgetClass,
			form15,
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-140-*" ),
			RES_CONVERT( XmNlabelString, "Cancel" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNleftPosition, 42,
			XmNleftOffset, 0,
			XmNleftAttachment, XmATTACH_POSITION,
			RES_CONVERT( XmNforeground, "black" ),
			XmNtopOffset, 10,
			XmNtopWidget, scrolledWindowText4,
			XmNtopAttachment, XmATTACH_WIDGET,
			NULL );
	XtAddCallback( pushButton3, XmNactivateCallback,
		(XtCallbackProc) activateCB_pushButton3,
		(XtPointer) NULL );




	return ( CommandHelp );
}

/*******************************************************************************
       The following is the 'Interface function' which is the
       external entry point for creating this interface.
       This function should be called from your application or from
       a callback function.
*******************************************************************************/

Widget	create_CommandHelp( swidget _UxUxParent )
{
	Widget                  rtrn;

	UxParent = _UxUxParent;

	{
		if (gSavedCommandHelp == NULL) {
		rtrn = _Uxbuild_CommandHelp();

		gSavedCommandHelp = rtrn;
			InitText();
		}
		else
			rtrn = gSavedCommandHelp;
		
		XtPopup(UxGetWidget(rtrn), no_grab);
		
		return(rtrn);
	}
}

/*******************************************************************************
       END OF FILE
*******************************************************************************/

