from azure.servicebus import ServiceBusClient, ServiceBusMessage
import json
from typing import Dict


def send_msg_to_servicebus(msg: Dict, params: Dict) -> None:
    """
    Method for sending a message to a ServiceBus queue.
    Args:
        msg: String message.
        params: Context params with service information.
    Returns:
        None
    """

    conn = params["service_conn"]
    queue = params["queue_name"]
    if conn and queue:
        with ServiceBusClient.from_connection_string(conn) as client:
            sender = client.get_queue_sender(queue_name=queue)
            message = ServiceBusMessage(str(msg), content_type="application/json")
            sender.send_messages(message)
    return None


def create_json_msg(
    msg: str = None, group: str = None, params: Dict = None, severity: str = None
) -> None:
    """
    Generate a JSON-dict like object for sending an alarm message to
    ServiceBus queue for sendind an email to an specific group and 
    specific severity.
    Args:
        msg: String message.
        group: Messaging group, created and managed directly in Azure.
        params: Context parameters.
        severity: Severity of the message. This is reflected in the inbox
        of the reciever.
    Returns:
        None
    """
    system = params['system']
    alarm = dict()
    alarm["command"] = "SendEmail"
    alarm["body"] = msg if msg is not None else "Generic Alarm"
    alarm["group"] = group if group is not None else "all"
    alarm["system"] = system if system is not None else "dch_kedro_molienda"
    alarm["severity"] = severity if severity is not None else "High"
    send_msg_to_servicebus(msg=alarm, params=params)
    return None
