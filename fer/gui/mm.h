#include <stdio.h>
#include "ferret_structures.h"

/* globals */
extern int gMacroIsRecording, gMMIsOpen;
swidget gSavedMacroManager = NULL;
extern swidget Open_Save_jnl, fileSelectionBox4;
swidget MacroManager;
static int macroIsDirty = 0;
static int macroHasText = 0;
static int macroSavedOnce = 0;
static char defaultPath[256];


/* prototypes */
swidget create_MacroManager(swidget UxParent);
static void InitialState(void);
static void ClearRadioGroup1(void);
extern void MMSaveOK(Widget UxWidget, XtPointer UxClientData, XtPointer UxCallbackArg);
extern void MMSaveAsOK(Widget UxWidget, XtPointer UxClientData, XtPointer UxCallbackArg);
extern void MMOpenOK(Widget UxWidget, XtPointer UxClientData, XtPointer UxCallbackArg);
extern void MMCancel(Widget UxWidget, XtPointer UxClientData, XtPointer UxCallbackArg);
static void SaveMMFile(void);
static void SaveAsMMFile(void);
static void OpenMMFile(void);
void DisplayMacroBuffer(void);