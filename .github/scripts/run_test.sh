#!/bin/sh

DUMP_DIR="$1"
shift

echo "Running test: $*"
OUTPUT="$(echo "$*" && echo "Test output:" && "$@" 2>&1)"
EC=$?
if [ $EC -eq 0 ]; then
    [ "$GC_DUMP_IR" = "1" ] && echo "$OUTPUT"
    exit 0
fi

DUMP_FILE="$DUMP_DIR/$(echo "$*" | sha1sum | cut -d' ' -f1).txt"
mkdir -p "$DUMP_DIR"
echo "Test '$*' failed with exit code $EC. Saving output to $DUMP_FILE."
echo "$OUTPUT" > "$DUMP_FILE"
[ "$GC_DUMP_IR" = "1" ] && echo "$OUTPUT" && exit $EC

echo "Running test again with IR dumps enabled:" >> "$DUMP_FILE"
GC_DUMP_IR=1 GC_FILE_CACHE_OFF=1 OV_MLIR_DEBUG=1 "$@" >> "$DUMP_FILE" 2>&1
cat "$DUMP_FILE"
exit $EC
