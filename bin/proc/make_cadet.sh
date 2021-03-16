#! /bin/bash
set -e

if [ $# -lt 1 ]; then
    echo "missing argument: input_csvs"
    echo "USAGE: make_cadet.sh <input_dir>"
    exit;
fi;

CADET_PREFIX=alpha
PYTHONBIN=/mnt/c/Python38/python.exe
echo "Making cadet model"
echo "Input dir: $1"
echo "cadet prefix: ${CADET_PREFIX}"

INPUT_DIR=$1
# csv -> pickles
echo "Converting csv to python pickles"

if [ ! -d $INPUT_DIR ]; then
    echo "[ERROR] Cannot find directory: ${INPUT_DIR}" 
    exit;
fi;

for csv in $INPUT_DIR/*.csv; do          
    pkl_path=${csv/.csv/.pkl}        
    if [ ! -f $pkl_path ]; then
        $PYTHONBIN make_threads.py $csv
    else
        echo "$pkl_path already exists, skipped"
    fi;
done;

# merge pickles
pkl_files="$INPUT_DIR/*.pkl"
merged_file="$INPUT_DIR/threads_merged.pkl"
if [ ! -f $merged_file ]; then
    $PYTHONBIN merge_threads.py "$pkl_files" $merged_file
else
    echo "threads already merged, skipped"
fi;

# building model

echo "Building cadet model"
$PYTHONBIN build_cadet.py "$merged_file" --prefix cadet/$CADET_PREFIX

echo "NOTE: cadet model is built under <CASA_ROOT>/data/cadet/$CADET_PREFIX."
echo "      Cadet still need the seed table, seed.csv."
echo "      Please place the appropriate copy into cadet/$CADET_PREFIX"
