static int numOptions=0;
static float lows[100], highs[100], deltas[100];

static void InitialState()
{
	ClearWindow();
	MaintainAddRemove();
	ClearOptions();
}

static void ClearContourDisplay()
{
	/*char *nullStr

	nullStr = '\0';*/
	/* clear the user contour field */
	XtVaSetValues(UxGetWidget(scrolledText5),
		XmNvalue, "",
		NULL);
	
	/* set save button to be insensitive */
	XtSetSensitive(UxGetWidget(pushButton13), False);
	DisableAddRemove();
}

static void DisableAddRemove()
{
	XtSetSensitive(UxGetWidget(pushButton15), False);
	XtSetSensitive(UxGetWidget(pushButton17), False);
}

static void EnableAddRemove()
{
	XtSetSensitive(UxGetWidget(pushButton15), True);
	XtSetSensitive(UxGetWidget(pushButton17), True);
}

static void MaintainAddRemove()
{
	char *testStr = (char *)XtMalloc(32);

	strcpy(testStr, "");

	/* check whether the low, high, delta are filled in and if so, enable ADD/REMOVE */
	testStr = XmTextFieldGetString(UxGetWidget(textField34));
	if (strlen(testStr) == 0) {
		DisableAddRemove();
		XtFree(testStr);
		return;
	}

	testStr = XmTextFieldGetString(UxGetWidget(textField36));
	if (strlen(testStr) == 0) {
		DisableAddRemove();
		XtFree(testStr);
		return;
	}

	testStr = XmTextFieldGetString(UxGetWidget(textField37));
	if (strlen(testStr) == 0) {
		DisableAddRemove();
		XtFree(testStr);
		return;
	}
	EnableAddRemove();
	XtFree(testStr);
}

static void ClearOptions()
{
	char nullStr[2];
	
	nullStr[0] = '\0';

	/* clear the levels flds */
	XtVaSetValues(UxGetWidget(textField34),
		XmNvalue, nullStr,
		NULL);
	XtVaSetValues(UxGetWidget(textField36),
		XmNvalue, nullStr,
		NULL);
	XtVaSetValues(UxGetWidget(textField37),
		XmNvalue, nullStr,
		NULL);

	/* clear the prefix and suffix flds */
	XtVaSetValues(UxGetWidget(textField38),
		XmNvalue, nullStr,
		NULL);
	XtVaSetValues(UxGetWidget(textField39),
		XmNvalue, nullStr,
		NULL);

	/* clear and unmap the decimal fld */
	XtVaSetValues(UxGetWidget(textField35),
		XmNvalue, nullStr,
		NULL);
	XtUnmapWidget(UxGetWidget(textField35));

	/* reset the digit toggles */
	XtVaSetValues(UxGetWidget(toggleButton33),
		XmNset, False,
		NULL);
	XtVaSetValues(UxGetWidget(toggleButton34),
		XmNset, False,
		NULL);
	XtVaSetValues(UxGetWidget(toggleButton35),
		XmNset, False,
		NULL);
	XtVaSetValues(UxGetWidget(rowColumn11),
/*50*/		XmNmenuHistory, "",
		NULL);
	digitState = 0;

	/* reset the style toggles */
	XtVaSetValues(UxGetWidget(toggleButton29),
		XmNset, False,
		NULL);
	XtVaSetValues(UxGetWidget(toggleButton30),
		XmNset, False,
		NULL);
	XtVaSetValues(UxGetWidget(toggleButton31),
		XmNset, False,
		NULL);
	XtVaSetValues(UxGetWidget(rowColumn10),
		XmNmenuHistory, "",
		NULL);
	styleState = 0;

	/* reset color menu to black */
}

static void ClearWindow()
{
	ClearContourDisplay();
	ClearOptions();
}

/* read the digits and get decimals if needed */
static void GetDigit(digitText)
char *digitText;
{
	Widget activeButton = NULL;
	XmString buttonLabel = XmStringCreate("Button", XmSTRING_DEFAULT_CHARSET), 
		noneLabel = XmStringCreate("None", XmSTRING_DEFAULT_CHARSET),
		intLabel = XmStringCreate("Integer", XmSTRING_DEFAULT_CHARSET),
		decLabel = XmStringCreate("Decimals:", XmSTRING_DEFAULT_CHARSET);
	char decText[3];

	strcpy(decText, "");
	strcpy(digitText, "");
	if (digitState) {
		/* get the toggle button */
		XtVaGetValues(UxGetWidget(rowColumn11),
			XmNmenuHistory, &activeButton,
			NULL);
		XtVaGetValues(activeButton,
			XmNlabelString, &buttonLabel,
			NULL);
		if (XmStringCompare(buttonLabel, noneLabel))
			strcpy(digitText, "None");
		else if (XmStringCompare(buttonLabel, intLabel))
			strcpy(digitText, "Integer");
		else if (XmStringCompare(buttonLabel, decLabel)) {
			/* get the decimal field */
			XtVaGetValues(UxGetWidget(textField34),
				XmNvalue, decText,
				NULL);
			strcpy(digitText, decText);
			strcat(digitText, " Places");
		}
	}

	XmStringFree(buttonLabel);
	XmStringFree(noneLabel);
	XmStringFree(intLabel);
	XmStringFree(decLabel);
}

/* read the color */
static void GetColor(colorText)
char *colorText;
{
	strcpy(colorText, "Black");
}

/* read the style */
static void GetStyle(styleText)
char *styleText;
{
	Widget activeButton = NULL;
	XmString buttonLabel = XmStringCreate("Button", XmSTRING_DEFAULT_CHARSET), 
		dashLabel = XmStringCreate("Dash", XmSTRING_DEFAULT_CHARSET),
		darkLabel = XmStringCreate("Dark", XmSTRING_DEFAULT_CHARSET),
		solidLabel = XmStringCreate("Solid", XmSTRING_DEFAULT_CHARSET); 

	strcpy(styleText, "");
	if (styleState) {
		XtVaGetValues(UxGetWidget(rowColumn10),
			XmNmenuHistory, &activeButton,
			NULL);
		/* get the togglebutton title */
		XtVaGetValues(activeButton,
			XmNlabelString, &buttonLabel,
			NULL);
		if (XmStringCompare(buttonLabel, dashLabel))
			strcpy(styleText, "Dash");
		else if (XmStringCompare(buttonLabel, darkLabel))
			strcpy(styleText, "Dark");
		else if (XmStringCompare(buttonLabel, solidLabel))
			strcpy(styleText, "Solid");
	}

	XmStringFree(buttonLabel);
	XmStringFree(dashLabel);
	XmStringFree(darkLabel);
	XmStringFree(solidLabel);
}

static void CreateContours(mode)
int mode;
{
	char *currContents, *numStr, *toTextBuffer, digitText[15], colorText[10], 
		styleText[6], *preText, *sufText, tempText[512], decText[10];
	int numLines = 0;
	register int i;
	float low, high, delta, lineValues[100], newVal;

	toTextBuffer = (char *)XtMalloc(5000 * sizeof(char));
	currContents = (char *)XtMalloc(10000 * sizeof(char));
	preText = (char *)XtMalloc(16 * sizeof(char));
	sufText = (char *)XtMalloc(16 * sizeof(char));
	numStr = (char *)XtMalloc(32 * sizeof(char));

	strcpy(toTextBuffer, "");
	strcpy(currContents, "");
	strcpy(preText, "");
	strcpy(sufText, "");
	strcpy(numStr, "");
	strcpy(digitText, "");
	strcpy(colorText, "");
	strcpy(styleText, "");
	strcpy(decText, "");
	strcpy(tempText, "");

	/* get low, high, delta */
	numStr = XmTextFieldGetString(UxGetWidget(textField34));
	sscanf(numStr, "%f", &low);

	numStr = XmTextFieldGetString(UxGetWidget(textField36));
	sscanf(numStr, "%f", &high);

	numStr = XmTextFieldGetString(UxGetWidget(textField37));
	sscanf(numStr, "%f", &delta);

	/* should do some error testing here */

	/* create the line values */
	newVal = low;
	while (1) {
		newVal = low + (numLines * delta);
		if (newVal > high) break;
		lineValues[numLines++] = newVal;
	}

	/* read the digits and get decimals if needed */
	GetDigit(digitText);

	/* read the color */
	GetColor(colorText); 

	/* read the style */
	GetStyle(styleText);

	/* read the prefix */
	preText = XmTextFieldGetString(UxGetWidget(textField38));

	/* read the suffix */
	sufText = XmTextFieldGetString(UxGetWidget(textField39));

	/* now build a string for display in the contour display */
	for (i=0;i<numLines;i++) {
		strcpy(tempText, "");
		if (mode == CREATE)
			sprintf(tempText, "(+) %.3f %8s %8s %6s %10s %10s\n",  lineValues[i], 
				digitText, colorText, styleText, preText, sufText);
		else
			sprintf(tempText, "(-) %.3f %8s %8s %6s %10s %10s\n",  lineValues[i], 
				digitText, colorText, styleText, preText, sufText);
		strcat(toTextBuffer, tempText);
	}

	/* append this to the textfield */
	/* first get contents 
	currContents = XmTextGetString(scrolledText5);*/
	XtVaGetValues(UxGetWidget(scrolledText5),
		XmNvalue, &currContents,
		NULL);

	strcat(currContents, toTextBuffer);
	XtVaSetValues(UxGetWidget(scrolledText5),
		XmNvalue, currContents,
		NULL);

	ClearOptions();

	/* set save to sensitive */
	XtSetSensitive(UxGetWidget(pushButton13), True);

	XtFree(numStr);
	XtFree(preText);
	XtFree(sufText);
	XtFree(toTextBuffer);
	XtFree(currContents);
}

/* ok and cancel callbacks for fileSelectionBox2 */

extern void SaveOK(UxWidget, UxClientData, UxCallbackArg)
Widget UxWidget;
XtPointer UxClientData, UxCallbackArg;
{
	XmSelectionBoxCallbackStruct *cbInfo = (XmSelectionBoxCallbackStruct *)UxCallbackArg;
	char *pathName, *contents;
	FILE *outFile;
	int io;

	pathName = (char *)malloc(cbInfo->length);
	strcpy(pathName, "");
	XmStringGetLtoR(cbInfo->value, XmSTRING_DEFAULT_CHARSET, &pathName);

	/* open the file */
	outFile = fopen(pathName, "w");
	
	/* get a pointer to contour text */
	contents = (char *)malloc(5000);
	strcpy(contents, "");
	XtVaGetValues(UxGetWidget(scrolledText5),
		XmNvalue, &contents,
		NULL);	

	/* write text to file */
	io = fwrite(contents, sizeof(char), strlen(contents), outFile);

	/* close file */
	io = fclose(outFile);

	/* pop down the interface */
	XtPopdown(UxGetWidget(Open_Save_ctl));

	XtFree(pathName);
	XtFree(contents);
}

extern void OpenOK(UxWidget, UxClientData, UxCallbackArg)
Widget UxWidget;
XtPointer UxClientData, UxCallbackArg;
{
	XmSelectionBoxCallbackStruct *cbInfo = (XmSelectionBoxCallbackStruct *)UxCallbackArg;
	char *pathName, *fileContents, *currContents;
	FILE *inFile;
	int io;

	pathName = (char *)malloc(cbInfo->length);
	strcpy(pathName, "");
	XmStringGetLtoR(cbInfo->value, XmSTRING_DEFAULT_CHARSET, &pathName);

	/* open the file */
	inFile = fopen(pathName, "r");
	
	/* create a buffer to store contour text */
	fileContents = (char *)malloc(5000);
	strcpy(fileContents, "");
/*300 */
	/* read text from file */
	io = fread(fileContents, sizeof(char), 5000, inFile);

	/* close file */
	io = fclose(inFile); 

	/* append this the contents of field */
	currContents = (char *)malloc(10000);
	strcpy(currContents, "");
	XtVaGetValues(UxGetWidget(scrolledText5),
		XmNvalue, currContents,
		NULL);

	strcat(currContents, fileContents);

	XtVaSetValues(UxGetWidget(scrolledText5),
		XmNvalue, currContents,
		NULL);

	/* pop down the interface */
	XtPopdown(UxGetWidget(Open_Save_ctl));

	XtFree(pathName);
	XtFree(fileContents);
	XtFree(currContents);
}
extern void Cancel(UxWidget, UxClientData, UxCallbackArg)
Widget UxWidget;
XtPointer UxClientData, UxCallbackArg;
{
	/* pop down the interface */
	XtPopdown(UxGetWidget(Open_Save_ctl));
}

static void SaveCTLFile()
{
	XmString dirMask;

	/* see if the interface has been created */
	Open_Save_ctl = create_Open_Save_ctl(NO_PARENT);
	
	XtVaSetValues(UxGetWidget(Open_Save_ctl),
		XmNtitle, "Save Contour Levels",
		NULL); 

	XtAddCallback(UxGetWidget(fileSelectionBox2),
		XmNokCallback,
		SaveOK,
		NULL);

	XtAddCallback(UxGetWidget(fileSelectionBox2),
		XmNcancelCallback,
		Cancel,
		NULL);

	dirMask = XmStringCreateSimple("*.ctl");

	XtVaSetValues(UxGetWidget(fileSelectionBox2),
		XmNdirMask, dirMask,
		NULL);

	/* apply the mask */
	XmFileSelectionDoSearch((Widget)UxGetWidget(fileSelectionBox2), 
		dirMask);

	/* popup Open file */
	XtPopup(UxGetWidget(Open_Save_ctl), XtGrabNone);
}

static void OpenCTLFile()
{
	XmString dirMask;

	/* see if the interface has been created */
	Open_Save_ctl = create_Open_Save_ctl(NO_PARENT);

	XtAddCallback(UxGetWidget(fileSelectionBox2),
		XmNokCallback,
		OpenOK,
		NULL);
	XtAddCallback(UxGetWidget(fileSelectionBox2),
		XmNcancelCallback,
		Cancel,
		NULL);

	dirMask = XmStringCreateSimple("*.ctl");

	XtVaSetValues(UxGetWidget(fileSelectionBox2),
		XmNdirMask, dirMask,
		NULL);

	/* apply the mask */
	XmFileSelectionDoSearch((Widget)UxGetWidget(fileSelectionBox2), 
		dirMask);

	XtVaSetValues(UxGetWidget(Open_Save_ctl),
		XmNtitle, "Open Contour Levels File",
		NULL);

	/* popup Open file */
	XtPopup(UxGetWidget(Open_Save_ctl), XtGrabNone);

	XmStringFree(dirMask);
}
