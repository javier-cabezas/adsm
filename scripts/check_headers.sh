#!/bin/bash

SRC_HEADERS=`find src/ -name *.h` 

for HEADER in $SRC_HEADERS
do
    NAME=`basename $HEADER`
    DIR=`dirname $HEADER`
    GUARDS=`grep "#define" $HEADER | cut -f2 -d" "`

    GUARD_ERROR_STRING=""
    LICENSE_ERROR_STRING=""
    for GUARD in $GUARDS
    do
        i=0
        if [ -n "$GUARD" ]
        then
            RESULTS=`grep "#ifndef $GUARD" $HEADER`
            if [ -n "$RESULTS" ]
            then
                CORRECT_DIR=${DIR/src\//}
                CORRECT_DIR=${CORRECT_DIR//\//_}
                CORRECT_DIR=${CORRECT_DIR^^[a-z]}
                CORRECT_NAME=`echo $NAME | cut -f 1 -d "."`
                CORRECT_NAME=${CORRECT_NAME//-/_}
                CORRECT_NAME=${CORRECT_NAME^^[a-z]}
                CORRECT="GMAC_${CORRECT_DIR}_${CORRECT_NAME}_H_"
                
                if [ $CORRECT != $GUARD ]
                then
                    GUARD_ERROR_STRING="Bad guard. Current: $GUARD Correct: $CORRECT"
                fi
                i=$(( i + 1 ))
            fi
        fi
        if [ $i -gt 1 ]
        then
            echo Script internal error
            exit
        fi
    done

    LICENSE=`grep "Permission is hereby granted, free of charge" $HEADER`
    if [ ! -n "$LICENSE" ]
    then
        LICENSE_ERROR_STRING="Bad license"
    fi
    if [ -n "$GUARD_ERROR_STRING" -o -n  "$LICENSE_ERROR_STRING" ]
    then
        echo "Error in file $HEADER"
        
        if [ -n "$GUARD_ERROR_STRING" ]
        then
            echo "$GUARD_ERROR_STRING"
        fi
        if [ -n "$LICENSE_ERROR_STRING" ]
        then
            echo "$LICENSE_ERROR_STRING"
        fi

    fi
done

# vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab:
