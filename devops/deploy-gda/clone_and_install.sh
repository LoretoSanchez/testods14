#!/bin/bash
INSTALL_DIR="/databricks/data"
DATALAKE_BASE_DIR_PROD="/dbfs/mnt/refined/modelos/DCH/minco/kedro_dch_plantas"
DATALAKE_BASE_DIR_TEST="/dbfs/mnt/refined/modelos/DCH/minco/kedro_dch_plantas_test"
DATALAKE_BASE_DIR_DEV="/dbfs/mnt/refined/modelos/DCH/minco/kedro_dch_plantas"

ENV=$1

PROD="prod"
TEST="test"
DEV="dev"

if [[ "$ENV" == "$PROD" ]]; then
    BASE_DIR=$DATALAKE_BASE_DIR_PROD
elif [[ "$ENV" == "$TEST" ]]; then
    BASE_DIR=$DATALAKE_BASE_DIR_TEST
else
    BASE_DIR=$DATALAKE_BASE_DIR_DEV
fi

cd $INSTALL_DIR/repo
pip install -r src/requirements.txt
cp data/09_data_dictionaries/* $BASE_DIR/data/09_data_dictionaries/
rm -rf data
ln -s $BASE_DIR/data .
