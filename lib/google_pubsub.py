import os
from fastapi.exceptions import HTTPException
from google.cloud import pubsub_v1
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class GcpPubSub:
    def __init__(self, project_id: Optional[str]):

        self.description = "This is the base class for Google Publisher Subscriber send and receive methods"
        if project_id is None:
            project_id = os.getenv("GOOGLE_CLOUD_PROJECT")

    def create_publisher(self, topic_name):
        self.publisher = pubsub_v1.PublisherClient()
        self.topic_name = "projects/{project_id}/topics/{topic}".format(
            project_id=self.project_id, topic=topic_name
        )
        self.publisher.create_topic(name=self.topic_name)

    def create_subscriber(self, subscriber_name):
        self.subscriber = pubsub_v1.SubscriberClient()
        self.subscription_name = "projects/{project_id}/subscriptions/{sub}".format(
            project_id=self.project_id, sub=subscriber_name
        )

    def callback(self, message):
        logger.info(message.data)
        # Acknowledge the message
        message.ack()

    def publish_message(self, message):
        message = message.encode("utf-8")
        streaming_put = self.publisher.publish(
            self.topic_name, message, origin="sample_app"
        )
        logging.info("Publish result:{}".format(streaming_put.result()))
        logging.info("Published message to {}".format(self.topic_name))

    def receive_message(self, message, timeout=5.0):
        streaming_pull = self.subscriber.subscribe(
            self.subscription_name, callback=self.callback
        )
        logging.info(
            "Listening for messages on {} .. \n\n".format(self.subscription_name)
        )
        with self.subscriber:
            try:
                return streaming_pull.result(timeout=timeout)
            except Exception as e:
                logging.info("Exception Caugh:{}".format(e))
                streaming_pull.cancel()
                streaming_pull.result()
                return 500
