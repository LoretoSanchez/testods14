import itertools as it
import logging
import time
from datetime import datetime
from typing import Any, Dict
from uuid import uuid4
import pandas as pd
import numpy as np
import os
import pytz

import mlflow
import mlflow.sklearn
import requests
import simplejson as json
from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog
from kedro.pipeline.node import Node

from project_clisham.utils import alarms

logger = logging.getLogger(__name__)


class MlFlowHooks:

    namespaces = [
        "s0",
        "s1",
        "s2",
        "s3",
        "s4",
        "s5",
        "s6",
        "s7",
        "s8",
        "s9",
        "s10",
        "s11",
        "s12",
        "s13",
        "s14",
        "s15",
        "s16",
        "s17",
        "fa0l1",
        "fa0l2",
        "fa1l1",
        "fa1l2",
        "fa2l1",
        "fa2l2",
        "fa2l3",
    ]

    def __init__(self):
        self.target_runid = {}

    @hook_impl
    def after_node_run(
        self, node: Node, inputs: Dict[str, Any], outputs: Dict[str, Any]
    ):
        if node.namespace in self.namespaces:
            target = node.namespace
            run_id = self.target_runid.get(target)
            with mlflow.start_run(run_id=run_id) as run:
                self.target_runid[target] = run.info.run_id
                mlflow.set_tag("model", node.namespace)
                if node.name.endswith("load_regressor"):
                    mlflow.log_params(
                        {
                            "train_model_params": inputs[
                                f"params:{target}.train_model"
                            ]["regressor"]
                        }
                    )

                elif node.name.endswith("train_tree_model"):
                    model = outputs[f"{target}.train_model"]
                    mlflow.sklearn.log_model(model, f"{target}.model")
                    mlflow.xgboost.autolog(model)

                elif node.name.endswith("create_predictions"):
                    metrics = outputs[f"{target}.test_set_metrics"].to_dict()
                    met_values = metrics["opt_perf_metrics"]
                    for key in met_values:
                        mlflow.log_metrics({key: met_values[key]})


class SendToAPIHook:
    namespaces = ["ca1xl", "ca2xl", "ma2", "ca0xl"]
    nodes_names = ["generate_sensitivity_data"]

    @staticmethod
    @hook_impl
    def after_node_run(
        node: Node,
        catalog: DataCatalog,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
    ):
        envs = [
            "cloud_recommend_dev",
            "cloud_recommend_prod",
            "cloud_recommend_test",
            "dev",
        ]
        KEDRO_ENV = catalog.load("params:KEDRO_ENV")
        if KEDRO_ENV in envs:
            # Debug prints
            logger.info(f"ENTER API HOOK")
            logger.info(f"NODE NAME: {node.name}")
            logger.info(f"NODE NAMESPACE: {node.namespace}")
            alarm_params = catalog.load("params:alarm_config")
            ns_list = SendToAPIHook.namespaces
            nodes = SendToAPIHook.nodes_names
            full_name = [".".join(x) for x in list(it.product(ns_list, nodes))]
            if node.namespace in ns_list and node.name in full_name:
                # Set the namespace
                ns = node.namespace
                # Retrieve parameters from the catalog
                params = catalog.load("params:backend_api")

                # Create the URLS for the API
                analytics_url = params["REACT_APP_OPTIMUS_ANALYTICS_API"]
                data_insight_url = params["REACT_APP_OPTIMUS_DATA_INSIGHT_API"]
                # Map the correct endpoint for each model.
                analytics_url += params["model_to_api_map"][ns]
                data_insight_url += params["model_to_api_map"][ns]
                # Debug info
                logger.info(f"Analytics URL: {analytics_url}")
                logger.info(f"Data Insight URL: {data_insight_url}")
                # Prepare the data to send
                recomm = catalog.load(f"{ns}.recommendations")
                sensi = outputs[f"{ns}.sensitivity_plot_df"]
                # retrieve the run_id to get the sens data.
                run_id = recomm["run_id"].values[0]

                json_data = [recomm.loc[0].to_dict()]
                # This is to remove the context
                json_data[0].pop("context")
                # let get the controls tags to filter on the sensitivity data.
                control_tags = list(json_data[0]["controls"].keys())
                # sensi data treatment
                sensi_data = sensi.query("run_id == @run_id").copy()
                sensi_data_grp = sensi_data.groupby("control_tag_id")
                # add sensi data to a new dict
                new_dict = dict()
                for name, group in sensi_data_grp:
                    if name in control_tags:
                        group["target_value"].to_list()
                        new_dict[name] = {
                            "target_value": group["target_value"].to_list(),
                            "control_value": group["control_value"].to_list(),
                        }
                # paste sensi data to json
                json_data[0]["sensitivity"] = new_dict
                # dump it into a string
                fixed_data = json.dumps(json_data, ignore_nan=True)
                logger.info(fixed_data)
                # Send the Data
                try:
                    headers = {
                        "Content-type": "application/json",
                        "Accept": "text/plain",
                    }
                    logger.debug(fixed_data)
                    response = requests.post(
                        analytics_url, data=fixed_data, headers=headers
                    )
                    logger.info(f"Analytics API Response:\t{response.content}")
                    if response.status_code != 201:
                        alarms.create_json_msg(
                            msg=response.content, group="data_team", params=alarm_params
                        )
                    # Wait a little.
                    time.sleep(10)
                    response = requests.post(
                        data_insight_url, data=fixed_data, headers=headers
                    )
                    logger.info(f"Data Insight API Response:\t{response.content}")
                    if response.status_code != 201:
                        alarms.create_json_msg(
                            msg=response.content, group="data_team", params=alarm_params
                        )
                except ConnectionRefusedError as c:
                    logger.error("Could not send the recommendation to the Backend.")
                    logger.error(c)
                except Exception as e:
                    logger.warning(e)
                    raise e

def get_right_path(KEDRO_ENV):
    """
    Provide the right DBFS path for storing
    objects using hooks.
    """
    if KEDRO_ENV=="cloud_recommend_test":
        return "/dbfs/mnt/refined/modelos/DCH/minco/kedro_dch_plantas_test"
    elif KEDRO_ENV=='cloud_recommend_prod':
        return "/dbfs/mnt/refined/modelos/DCH/minco/kedro_dch_plantas"
    elif KEDRO_ENV=='cloud_recommend_dev':
        return "/dbfs/mnt/refined/modelos/DCH/minco/kedro_dch_plantas"
    else:
        return None

class GeneralAlarm:
    @staticmethod
    @hook_impl
    def on_pipeline_error(
        error: Exception, 
        run_params: Dict, 
        catalog: DataCatalog
    ) -> None:
        """
        This method is in charge of sending an Alarm email when the 
        pipeline fails. It grabs automatically the error exception and 
        generate a predefined message after the retry 3. By now, this number 
        is hardcoded. The logic for verifying if the run belongs to the 
        same set of retry is also hardcoded, and it looks if the latest failed 
        run was 3 hours ago or less.
        It also store a historic logs files with the same information.
        Args:
            error: Error Exception for better tracking.
            run_params: Parameters of the running instance.
            catalog: Context DataCatalog.
        Returns:
            None
        """
        KEDRO_ENV = catalog.load("params:KEDRO_ENV")
        logger.info(f"Working in: {KEDRO_ENV}")

        datetime = get_local_time().strftime("%d-%m-%Y %H:%M:%S")

        base_dir = get_right_path(KEDRO_ENV)
        parent_dir = f"{base_dir}/data/run_logs/{run_params['pipeline_name']}"

        historic_path = f'{parent_dir}/historic_logs.csv'
        latest_path = f'{parent_dir}/latest_logs.csv'
        
        if base_dir and not os.path.exists(parent_dir):
          logger.info(f"Trying to create {parent_dir}")
          os.makedirs(parent_dir)

        allowed_envs = [
            "cloud_recommend_dev",
            "cloud_recommend_prod",
            "cloud_recommend_test",
        ]
        
        # hardcoding this due the first run need it
        try:
            latest_run = pd.read_csv(latest_path)
            latest_run.run_id = pd.to_datetime(latest_run.run_id)
        except:
            latest_run = None
        
        if KEDRO_ENV in allowed_envs:
            
            data = pd.DataFrame.from_dict({'message':['Failed run'],
                    'run_id':[datetime],
                    'enviroment':[run_params['env']],
                    'error':[error],
                    'retry':[0]}
                    )

            data.run_id = pd.to_datetime(data.run_id)

            if type(latest_run)!=type(None):

                delta = np.timedelta64(3,'h')
                diff = data.run_id.values[0] - delta
                
                if latest_run.run_id.values[0] > diff:
                    # if the run is in range, add 1 to the retry
                    retry_num = latest_run.retry.values[0] + 1
                    data.retry = retry_num
                else:
                    # if not, start a new counter
                    retry_num = 1
                    data.retry = retry_num
                
                if retry_num > 3:
                    # if we are reaching the limit, send the alert
                    msg = f"Pipeline Error: {error}\n"
                    msg += f"RUN: {datetime}\n"
                    msg += f"ENV: {run_params['env']}\n"
                    msg += f"Pipeline: {run_params['pipeline_name']}\n"

                    logger.info(msg)
                    html_ = fill_html_report(error, run_params['pipeline_name'], datetime)
                    html_ = html_.replace('%%','%')
                    alarm_params = catalog.load("params:alarm_config")

                    alarms.create_json_msg(
                    msg=html_, group="data_team", params=alarm_params, severity="High"
                    )

                data.to_csv(latest_path, index = False)

                # append values to historic
                historic_logs = pd.read_csv(historic_path)
                historic_logs = historic_logs.append(data)
                historic_logs.to_csv(historic_path, index = False)

            else:
                data.retry = 1
                data.to_csv(latest_path, index = False)
                data.to_csv(historic_path, index = False)

class SuccessLog:
    @staticmethod
    @hook_impl
    def after_pipeline_run(
        run_params: Dict, 
        catalog: DataCatalog
    ) -> None:
        """
        This method is in charge of saving a new row in the historic
        log file if the pipeline end successfully. 
        Args:
            run_params: Parameters of the running instance.
            catalog: Context DataCatalog.
        Returns:
            None
        """
        KEDRO_ENV = catalog.load("params:KEDRO_ENV")
        logger.info(f"Working in: {KEDRO_ENV}")
        
        datetime = get_local_time().strftime("%d-%m-%Y %H:%M:%S")
        base_dir = get_right_path(KEDRO_ENV)
        parent_dir = f"{base_dir}/data/run_logs/{run_params['pipeline_name']}"

        historic_path = f'{parent_dir}/historic_logs.csv'

        if base_dir and not os.path.exists(parent_dir):
          logger.info(f"Trying to create {parent_dir}")
          os.makedirs(parent_dir)

        allowed_envs = [
            "cloud_recommend_dev",
            "cloud_recommend_prod",
            "cloud_recommend_test",
        ]
        
        if KEDRO_ENV in allowed_envs:
            
            data = pd.DataFrame.from_dict({'message':['Success run'],
                    'run_id':[datetime],
                    'enviroment':[run_params['env']],
                    'error':['NA'],
                    'retry':['NA']}
                    )

            data.run_id = pd.to_datetime(data.run_id)

            # append values to historic
            if os.path.exists(historic_path):
                historic_logs = pd.read_csv(historic_path)
                historic_logs = historic_logs.append(data)
                historic_logs.to_csv(historic_path, index = False)
            else:
                data.to_csv(historic_path, index = False)

# TODO: To deprecate
#def _clean_run_id(run_id):
#    """
#    Helper to parse run id
#    """
#    run_id = run_id.split('T')
#    date = run_id[0]
#    time = run_id[1]
#    
#    time = time.split('.')[:-1]
#    time = ':'.join(time)
#    return f'{date} {time}'

def get_local_time():
  local_tz = pytz.timezone('America/Santiago')
  local_dt = datetime.utcnow().replace(tzinfo=pytz.utc).astimezone(local_tz)
  return local_tz.normalize(local_dt)

def from_utc_to_local_time(utc_date):
  local_tz = pytz.timezone('America/Santiago')
  local_dt = utc_date.replace(tzinfo=pytz.utc).astimezone(local_tz)
  return local_dt

def fill_html_report(error, pipeline, timestamp):
    return """
    <!DOCTYPE html>
    <html lang="en" xmlns="http://www.w3.org/1999/xhtml" xmlns:o="urn:schemas-microsoft-com:office:office">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width,initial-scale=1">
      <meta name="x-apple-disable-message-reformatting">
      <title></title>
      <link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.3.1/css/all.css">
      <!--[if mso]>
      <style>
        table {border-collapse:collapse;border-spacing:0;border:none;margin:0;}
        div, td {padding:0;}
        div {margin:0 !important;}
      </style>
      <noscript>
        <xml>
          <o:OfficeDocumentSettings>
            <o:PixelsPerInch>96</o:PixelsPerInch>
          </o:OfficeDocumentSettings>
        </xml>
      </noscript>
      <![endif]-->
      <style>
        table, td, div, h1, p {
          font-family: Arial, sans-serif;
        }
        @media screen and (max-width: 530px) {
          .unsub {
            display: block;
            padding: 8px;
            margin-top: 14px;
            border-radius: 6px;
            background-color: #d6d6d6;
            text-decoration: none !important;
            font-weight: bold;
          }
          .col-lge {
            max-width: 100%% !important;
          }
        }
        @media screen and (min-width: 531px) {
          .col-sml {
            max-width: 27%% !important;
          }
          .col-lge {
            max-width: 73%% !important;
          }
        }

        /*------*/
    
      .table-responsive table {
        border-collapse: collapse;
        border-width: 0;
        /* Making the table self-contain itself on overflow */
        display: block;
        overflow: auto;
      }

    .table-responsive td,
      th {
        /* Spacing and border for Table cells */
        border: 1px solid #ebebeb;
        padding: .5em 1em;
      }



      </style>
    </head>
    <body style="margin:0;padding:0;word-spacing:normal;background-color:#374752;">
      <div role="article" aria-roledescription="email" lang="en" style="text-size-adjust:100%%;-webkit-text-size-adjust:100%%;-ms-text-size-adjust:100%%;background-color:#374752;">
        <table role="presentation" style="width:100%%;border:none;border-spacing:0;">
          <tr>
            <td align="center" style="padding:0;">
              <!--[if mso]>
              <table role="presentation" align="center" style="width:600px;">
              <tr>
              <td>
              <![endif]-->
              <table role="presentation" style="width:94%%;max-width:600px;border:none;border-spacing:0;text-align:left;font-family:Arial,sans-serif;font-size:16px;line-height:22px;color:#363636;">
                <tr>
                  <td style="padding:40px 30px 20px 30px;text-align:center;font-size:24px;font-weight:bold;">
                    <a href="http://www.codelco.com/" style="text-decoration:none;"><img src="https://www.codelco.com/prontus_codelco/site/artic/20151113/imag/foto_0000000120151113170607.png" width="165" alt="Logo" style="width:80%%;max-width:165px;height:auto;border:none;text-decoration:none;color:#ffffff;"></a>
                  </td>
                </tr>

                <tr>
                  <td style="padding:30px;background-color:#0098AA;">
                    <h1 style="margin-top:0;margin-bottom:0px;font-size:26px;line-height:32px;font-weight:bold;letter-spacing:-0.02em;color: #f0f0f5;">¡Notificación de Alarma de Kedro!</h1>
                    </td>
                </tr>
                <tr>
                  <td style="padding-left:30px;padding-top:5px;background-color:#C4C1A0;">
                    <h1 style="margin-top:10;margin-bottom:10px;;margin-top:5px;font-size:18px;line-height:20px;font-weight:bold;letter-spacing:-0.02em;color: #f0f0f5;">Modelo "Molienda"</h1>
                    </td>
                </tr>
                <tr>
                  <td style="padding:35px 30px 11px 30px;font-size:0;background-color:#ffffff;border-bottom:1px solid #f0f0f5;border-color:rgba(201,201,207,.35);">
                    <!--[if mso]>
                    <table role="presentation" width="100%%">
                    <tr>
                    <td style="width:145px;" align="left" valign="top">
                    <![endif]-->
                    <div class="col-sml" style="display:inline-block;width:100%%;max-width:145px;vertical-align:top;text-align:left;font-family:Arial,sans-serif;font-size:14px;color:#363636;">

                    <h2 style="width:100%%;max-width:135px;margin-bottom:20px; color:#F4AA00; border-right: 1px dashed rgb(213, 213, 213);">¡ATENCIÓN!</h2>

                    </div>
                    <!--[if mso]>
                    </td>
                    <td style="width:395px;padding-bottom:20px;" valign="top">
                    <![endif]-->
                    <div class="col-lge" style="display:inline-block;width:100%%;max-width:395px;vertical-align:top;padding-bottom:20px;font-family:Arial,sans-serif;font-size:16px;line-height:22px;color:#363636;">
                      <h3 style="margin-top:0;margin-bottom:0px;font-size:24px;line-height:32px;font-weight:bold;letter-spacing:-0.02em;color: #444444;"> %s </br > <span style="font-size:20px;color:#919191;">cloud_prod Pipeline: %s</span></h3>
                     </div>
                    <!--[if mso]>
                    </td>
                    </tr>
                    </table>
                    <![endif]-->
                  </td>
                </tr>

                <tr>
                  <td style="padding:30px;background-color:#ffffff;">
                    <p style="margin:0; font-size: 12px; font-weight: bold;">* Ejecución: %s</p>
                  </td>
                </tr>
                <tr>
                  <td style="padding:30px;text-align:center;font-size:12px;background-color:#404040;color:#cccccc;">

                    <p style="margin:0;font-size:14px;line-height:20px;">
                      &reg; GDAA - 2021
                    </p>
                  </td>
                </tr>
              </table>
              <!--[if mso]>
              </td>
              </tr>
              </table>
              <![endif]-->
            </td>
          </tr>
        </table>
      </div>
    </body>
    </html>
    """ %(error, pipeline, timestamp)