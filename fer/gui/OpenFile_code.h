/* 
 * OpenFile_code.h
 *
 * John Osborne
 * Jonathan Callahan
 * April 4th 1996
 *
 * This file contains the necessary header information which is included by
 * OpenFile.h.
 *
 */

/*.....     includes     .....*/

#include "ferret_structures.h"
#include "JC_Utility.h"

/*.....     defines     .....*/
#define charset	XmSTRING_DEFAULT_CHARSET

/*.....     variables     .....*/
/* increase DSText to MAX_NAME_LENGTH  3/99 *kob* */
static char	DSText[MAX_NAME_LENGTH];
swidget		OpenFile;
swidget		gSavedOpenFile = NULL;

extern LIST *GLOBAL_DatasetNameList;
extern swidget	Open_Save_dset, FerretMainWd;
extern Boolean	gHiRez;

/*.....     functions     .....*/
static void	ActivateCB(Widget wid, XtPointer clientData, XtPointer callData);
static void	CancelOpen(void);
static void	InitialList(void);
static void	ListBrowserCB(Widget UxWidget, XtPointer UxClientData, XtPointer UxCallbackArg);
static void	MaintainBtns(void);
static void	OpenOK(void);
swidget		create_OpenFile(swidget UxParent);

extern swidget	create_Open_Save_dset(swidget UxParent);
extern void	ferret_command(char *cmdText, int cmdMode);
extern int	JC_II_Synchronize(swidget caller_id);

