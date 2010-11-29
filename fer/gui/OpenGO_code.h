/* 
 * OpenGO_code.h
 *
 * John Osborne
 * Jonathan Callahan
 * Nov 25th 1996
 *
 * This file contains the necessary header information which is included by
 * OpenGO.h.
 *
 */

/*.....     includes     .....*/

#include "ferret_structures.h"

/*.....     defines     .....*/
#define charset	XmSTRING_DEFAULT_CHARSET

/*.....     variables     .....*/
swidget OpenGO;
swidget gSavedOpenGO = NULL;
static char GOText[80];

extern Boolean gHiRez;
extern swidget OpenGOFile, FerretMainWd;

/*.....     functions     .....*/
static char *CollectToSpace(char *targetStr,char * subStr);
static void ActivateCB(Widget wid, XtPointer clientData, XtPointer callData);

swidget create_OpenGO(swidget UxParent);
void GOCancelOpen(void);
void GOOpenOK(void);
void ListBrowserCB(Widget UxWidget, XtPointer UxClientData, XmListCallbackStruct *UxCallbackArg);
void InitialList(void);
void MaintainBtns(void);

extern void ferret_command(char *cmdText, int cmdMode);
extern int JC_II_Synchronize( swidget caller_id );
extern swidget create_OpenGOFile(swidget UxParent);
#ifdef NO_ENTRY_NAME_UNDERSCORES
extern void ferret_list_in_window(char *text, int mode);
#else
extern void ferret_list_in_window_(char *text, int mode);
#endif
