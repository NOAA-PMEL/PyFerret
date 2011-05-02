#! /bin/csh -f
# Fpatch
# Untar's the file fer_patch.tar.Z to patch in additions to the current version
# of ferret and its support files
# J Davison 6.94/8.28.95
# NOAA PMEL TMAP
echo " This script patches the current version of FERRET with enhancements to"
echo " GO scripts, documentation and other support files. "

### Print menu and act on choice 
menu:
echo " "
echo " Enter your choice:"
echo " (1) Install the patch, (2) Exit and do nothing"
echo -n " (1 or 2) --> "
set choice = $<

if ($choice == 1) goto install_patch
if ($choice == 2) exit
goto menu

### Install the patch ###################################################
install_patch:
echo " "
echo " Install patch..."
echo " "

### Get FER_DIR value
if ($?FER_DIR) then
	set fer_dir = $FER_DIR
	echo " The environment variable FER_DIR is currently defined as"
	echo " $FER_DIR." 
	echo " This is the directory where the 'fer_environment' tar file was installed." 
	echo -n " Is that correct and acceptable (y/n) [y] "
	set ans = $<
	if ($ans != 'N' && $ans != 'n' ) goto get_tarloc
        echo " "
endif

echo " Enter the complete path of the directory where the 'fer_environment'"
echo " tar file was installed (FER_DIR). The location recommended"
echo " in the FERRET installation guide was '/usr/local/ferret'. "

getfer_dir:
echo -n " FER_DIR --> "
set fer_dir = $<
set fer_dir = $fer_dir

echo $fer_dir | grep '^/' > /dev/null
if ($status != 0) then
        echo " Sorry, you can't use relative pathnames..."
        goto getfer_dir
endif

if (! -d $fer_dir) then
        echo " '$fer_dir' is not a directory..."
        goto getfer_dir
endif

if (`find $fer_dir/bin -name Fgo -print` == "") then
        echo " The FERRET environment files are not in $fer_dir..."
        goto getfer_dir
endif

### Get directory where ferret patch tar file is supposed to be
get_tarloc:
echo " "
echo " Enter the complete path of the directory where you put the"
echo " 'patch' tar file."

getferpatch_dir:
echo -n " patch tar file location --> "
set ferpatch_dir = $<
set ferpatch_dir = $ferpatch_dir

echo $ferpatch_dir | grep '^/' > /dev/null
if ($status != 0) then
        echo " Sorry, you can't use relative pathnames..."
        goto getferpatch_dir
endif

if (! -d $ferpatch_dir) then
        echo " '$ferpatch_dir' is not a directory..."
        goto getferpatch_dir
endif

### Get name of patch tar file
get_patchname:
echo " "
echo " Enter the name of this 'patch' tar file, e.g. patch_28aug95.tar.Z"
echo -n " patch tar file name --> "

set patchname = $<
set patchname = $patchname

if (! -e $ferpatch_dir/$patchname) then
        echo " '$patchname' is not in $ferpatch_dir..."
        goto get_patchname
endif

### Move to $FER_DIR directory and begin work
pushd $fer_dir > /dev/null
echo " "
echo " Moving to $cwd..."

### Untar the patch tar file
echo " "
zcat $ferpatch_dir/$patchname | tar xvpf -
if ($status != 0) goto patch_err
echo " "
echo " Extracted ${patchname}..."

echo " "
echo "This FERRET patch installed `date +'%d%h%y'|tr '[A-Z]' '[a-z]'` by $user" >! $fer_dir/${patchname}_README
if ($status != 0) goto patch_err
echo " Created $fer_dir/${patchname}_README log file..."

echo
pushd > /dev/null
echo " "
echo " Returning to $cwd..."
goto menu

### There was a problem installing the patch
patch_err:
pushd > /dev/null
echo " "
echo "There's a problem manipulating files in $fer_dir.  Check your"
echo "privileges to change files in that directory and try again."
exit (1)
